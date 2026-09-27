"""Slurm job preemption: who may preempt whom, which jobs are chosen, and how.

Off by default (`PreemptType=preempt/none`, Slurm's default), in which case
nothing in this module is consulted and every result is unchanged.

Modelled: `PreemptType=preempt/partition_prio` and `preempt/qos`;
`PreemptMode=CANCEL` and `REQUEUE`, with the `WITHIN` and `PRIORITY` flags;
`GraceTime` (per partition for partition_prio, per QOS for qos);
`PreemptExemptTime` (global and per QOS); `PreemptParameters=youngest_first`,
`reorder_count` and `strict_order`; `JobRequeue`; and the `requeue_delay`
SchedulerParameter. Not modelled: `SUSPEND` (and its alias `ON`), refused
rather than approximated, and the `GANG` flag (time-slicing needs the gang
scheduler), parsed and reported as not modelled. Also not modelled:
licences, reservations, `min_exempt_priority`, heterogeneous jobs.

Where every rule comes from. Documentation is slurm.conf(5) and sacctmgr(1)
for Slurm 26.05 and https://slurm.schedmd.com/preempt.html, read 2026-09-26.
Source is SchedMD/slurm at 9f9da53b4a7bc5b56062bc357f4f94e0d19c71cc:

* **Eligibility** (`preempt_p_preemptable()` in
  src/plugins/preempt/partition_prio/preempt_partition_prio.c and
  src/plugins/preempt/qos/preempt_qos.c). partition_prio: the preemptee's
  partition must have a strictly lower `PriorityTier` and must not have
  `PreemptMode=OFF`. qos: the preemptor's QOS must list the preemptee's QOS in
  `Preempt`, or the two share a QOS and `WITHIN` is set, in which case the
  preemptor's job priority must be higher. With the `PRIORITY` flag the
  preemptor's job priority must not be lower (partition_prio) or must be
  higher (qos). "The Priority of a QOS is NOT related to QOS preemption"
  (sacctmgr(1)).
* **Candidates** (`slurm_find_preemptable_jobs()`, src/interfaces/preempt.c):
  running jobs that pass eligibility, whose resolved PreemptMode is not OFF,
  and that have run for at least `PreemptExemptTime` (a QOS value overrides
  the global one; `acct_policy_get_preemptable_time()`, acct_policy.c; -1,
  INFINITE and UNLIMITED mean none, i.e. 0).
  Sorted ascending by `preempt_p_get_prio()` — `PriorityTier` (or QOS
  priority) in the upper 16 bits and node count in the lower 16, "so we can
  preempt smaller jobs rather than larger jobs" — or, with `youngest_first`,
  latest start first. Slurm's sort is not stable; ties here go to the lower
  job id (then, for jobs sharing an id, the earlier original submit), a
  choice of this model.
* **Selection** (`_run_now()`, src/plugins/select/cons_tres/job_test.c):
  remove candidates in order from a copy of the node state until the job
  fits; then (up to `reorder_count` times) re-sort to try to preempt fewer
  jobs and search again; finally preempt the removed jobs that sit on the
  nodes chosen. Node choice inside that search is this simulator's first-fit,
  not cons_tres's, so the preempted set can differ from Slurm's on the same
  state.
* **Effect** (`select_nodes()` and `_preempt_jobs()`, node_scheduler.c): the
  preemptor does not start in that pass (`ESLURM_NODES_BUSY`); it starts in a
  later pass once the resources are free. A preemptor that preempted within
  `KillWait + MessageTimeout` seconds (defaults 30 + 10) does not preempt more
  jobs yet.
* **GraceTime** (`_job_check_grace_internal()`, preempt.c): "Once a job has
  been selected for preemption, its end time is set to the current time plus
  GraceTime" (slurm.conf(5)), and it keeps its allocation until then. "A job
  selected for preemption that exits before GraceTime expires will be handled
  according to PreemptMode regardless of why it exited" — so under REQUEUE, a
  job that happens to finish inside its grace period is requeued and runs
  again. Slurm enforces the new end time in its periodic time-limit check, so
  a real release can lag it; the model releases exactly on time.
* **Requeue** (`batch_requeue_fini()`, job_mgr.c): the job returns to
  PENDING with `submit_time = now`, `begin_time = now + requeue_delay + 1`
  (`requeue_delay` defaults to `AuthInfo=cred_expire`, whose default is
  120 s, per slurm.conf(5)), and `accrue_time = 0`, so its age factor
  restarts from the new begin time and its priority is recomputed. A job that
  cannot be requeued (`JobRequeue=0`) is cancelled instead ("Preempts jobs by
  requeuing them (if possible) or canceling them").
* **Queue order** (`sort_job_queue2()`, job_scheduler.c; slurm.conf(5)):
  "1. Jobs that can preempt [...] 3. Partition PriorityTier 4. Job priority
  5. Job submit time 6. Job ID". The first rule is a pairwise test, not a
  key. With preempt lists that do not form a ladder (c preempts b, b
  preempts a, c does not preempt a) it is not a total order, in Slurm or
  here, and a job can come after one it may preempt: a main pass can start
  a job and then preempt it for a later job in the same pass. The simulator
  handles that. Lists that form a *loop* are refused, as slurmdbd refuses
  them (`_preempt_loop`).

Two things are the lab's own. `checkpoint_fraction` (default 0) is the share
of a preempted run's progress, up to its selection, that a requeued job keeps.
Slurm restarts the batch script and knows nothing of application checkpoints.
The job keeps running through its grace period in Slurm, but the model credits
no progress to that time: it is grace-locked, as in the k8s lab's accounting. And preemption is
triggered by the main scheduler only: Slurm's backfill scheduler can also
start a preemptor, but here backfill plans a preemptor around running jobs
like any other job.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

from .model import Cluster, Job

PreemptType = Literal["preempt/none", "preempt/partition_prio", "preempt/qos"]
PREEMPT_TYPES: tuple[str, ...] = get_args(PreemptType)

BaseMode = Literal["OFF", "CANCEL", "REQUEUE"]

#: `usable_nodes` marker in `_run_now()` for the job that completed a fit.
_COMPLETED_FIT = 99_999


@dataclass(frozen=True)
class PreemptMode:
    """A `PreemptMode` value: one mechanism plus flags."""

    base: BaseMode = "OFF"
    within: bool = False
    priority: bool = False
    #: Parsed so a config can be read, never modelled (see module docstring).
    gang: bool = False

    @classmethod
    def parse(cls, text: str) -> PreemptMode:
        """`REQUEUE`, `CANCEL,WITHIN`, `OFF`, ... (case-insensitive, comma list).

        The rules of `preempt_mode_num()` (src/common/slurm_protocol_defs.c
        L2746-L2803, SchedMD/slurm@9f9da53), whose NO_VAL16 result slurmctld
        turns into "PreemptMode=%s invalid" and does not start
        (read_config.c L4693-L4698):

        * at most one base mode: OFF, CLUSTER (the same value as OFF), CANCEL,
          REQUEUE, or SUSPEND and its alias ON. "Only one mode value may be
          set, optionally combined with GANG or WITHIN", so `CANCEL,REQUEUE`
          is refused rather than read as whichever came last;
        * `GANG,WITHIN` and `GANG,PRIORITY` are invalid combinations.

        SUSPEND (and so ON) is refused for another reason: it needs the gang
        scheduler, which is not modelled.
        """
        base: BaseMode = "OFF"
        within = priority = gang = False
        bases: list[str] = []
        for raw in text.split(","):
            token = raw.strip().upper()
            if not token:
                continue  # strtok_r() skips empty tokens
            if token in {"OFF", "CLUSTER"}:
                bases.append(token)
                base = "OFF"
            elif token in {"CANCEL", "REQUEUE"}:
                bases.append(token)
                base = token  # type: ignore[assignment]
            elif token in {"SUSPEND", "ON"}:
                raise ValueError(
                    f"PreemptMode={text!r}: {token} needs the gang scheduler, which is not "
                    "modelled"
                )
            elif token == "WITHIN":
                within = True
            elif token == "PRIORITY":
                priority = True
            elif token == "GANG":
                gang = True
            else:
                raise ValueError(f"PreemptMode={text!r}: unknown value {token!r}")
        if len(bases) > 1:
            raise ValueError(
                f"PreemptMode={text!r} invalid: only one mode may be set ({', '.join(bases)}), "
                "optionally combined with GANG or WITHIN (preempt_mode_num())"
            )
        if gang and (within or priority):
            raise ValueError(
                f"PreemptMode={text!r} invalid: GANG cannot be combined with "
                f"{'WITHIN' if within else 'PRIORITY'} (preempt_mode_num())"
            )
        return cls(base, within, priority, gang)

    @property
    def is_unset(self) -> bool:
        """True for the value 0: OFF (or CLUSTER) with no flag.

        That is the only value Slurm treats as "not set". A QOS PreemptMode
        of 0 defers to the cluster's (`preempt_p_get_mode()`, preempt_qos.c
        L66-L80); any other value, a bare `WITHIN` included, replaces it.
        """
        return self.base == "OFF" and not (self.within or self.priority or self.gang)

    def __str__(self) -> str:
        parts: list[str] = [self.base]
        parts += [f for f, on in (("WITHIN", self.within), ("PRIORITY", self.priority)) if on]
        parts += ["GANG"] if self.gang else []
        return ",".join(parts)


@dataclass(frozen=True)
class PartitionSpec:
    """The partition settings preemption and queue order read.

    Every partition spans every node in this simulator (there are no
    per-partition node lists), which is the overlapping-partition layout
    preempt/partition_prio needs.
    """

    name: str
    #: slurm.conf(5) default 1 (`conf_part->priority_tier = 1`, read_config.c).
    priority_tier: int = 1
    #: Seconds; used by preempt/partition_prio only.
    grace_time: float = 0.0
    #: None means the cluster-wide PreemptMode.
    preempt_mode: PreemptMode | None = None


@dataclass(frozen=True)
class QOSSpec:
    """The QOS settings preemption reads (sacctmgr(1))."""

    name: str
    #: Only orders candidates for preemption; it does not grant the right.
    priority: int = 0
    #: Names of the QOSes this one may preempt. slurmdbd refuses a list that
    #: makes a loop (see `PreemptionConfig`).
    preempt: frozenset[str] = frozenset()
    #: None is `CLUSTER` (the default): use the cluster-wide PreemptMode. So
    #: is a bare `OFF`, which Slurm stores as the same 0. Any other value
    #: replaces the cluster's, flags and all: a bare `WITHIN` resolves to
    #: OFF, and that QOS's jobs cannot be preempted (see `mode_of`).
    preempt_mode: PreemptMode | None = None
    grace_time: float = 0.0
    #: None is -1: defer to the global `PreemptExemptTime`. So are a negative
    #: value and infinity: in `acct_policy_get_preemptable_time()` INFINITE
    #: (the stored -1) "means none" and falls through to the next level.
    preempt_exempt_time: float | None = None


def _preempt_loop(qos: Mapping[str, QOSSpec]) -> list[str] | None:
    """A loop in the QOS `Preempt` graph, as a path of names, or None.

    slurmdbd will not store one. `_preemption_loop()` in
    src/plugins/accounting_storage/mysql/as_mysql_qos.c (SchedMD/slurm@9f9da53
    L77-L126, called from the QOS modify path at L925) rejects a QOS whose own
    Preempt list names it ("has an internal loop"), or whose preemptees,
    followed transitively, lead back to it ("has a loop at QOS"). Since every
    change is checked, a database it accepted has no cycle anywhere, so any
    cycle is refused here. Names absent from `qos` grant nothing and are
    skipped.

    Why it matters beyond fidelity: with a loop, "can preempt" stops being a
    partial order at all, and the queue order built on it (`queue_order`)
    lets jobs preempt each other in turn without end.
    """
    done: set[str] = set()
    path: list[str] = []

    def visit(name: str) -> list[str] | None:
        path.append(name)
        for nxt in sorted(qos[name].preempt):
            if nxt not in qos or nxt in done:
                continue
            if nxt in path:
                return [*path[path.index(nxt) :], nxt]
            found = visit(nxt)
            if found is not None:
                return found
        path.pop()
        done.add(name)
        return None

    for name in sorted(qos):
        if name not in done:
            found = visit(name)
            if found is not None:
                return found
    return None


@dataclass(frozen=True)
class PreemptionConfig:
    """Everything preemption depends on. The default disables it."""

    preempt_type: PreemptType = "preempt/none"
    mode: PreemptMode = field(default_factory=PreemptMode)
    partitions: Mapping[str, PartitionSpec] = field(default_factory=dict)
    qos: Mapping[str, QOSSpec] = field(default_factory=dict)
    #: `PreemptExemptTime`, seconds. slurm.conf(5): "A time of -1 disables
    #: the option, equivalent to 0"; `time_str2secs()` reads `-1`, `INFINITE`
    #: and `UNLIMITED` all as INFINITE, and `acct_policy_get_preemptable_time()`
    #: (acct_policy.c L5203-L5226) treats INFINITE as none. So any negative
    #: or infinite value is stored as 0 here, whatever path set it.
    exempt_time: float = 0.0
    youngest_first: bool = False
    #: `PreemptParameters=reorder_count`, default 1.
    reorder_count: int = 1
    strict_order: bool = False
    #: `JobRequeue`, default 1. With 0, REQUEUE falls back to cancelling.
    job_requeue: bool = True
    #: SchedulerParameters `requeue_delay`; default `cred_expire` = 120 s.
    requeue_delay: float = 120.0
    #: `KillWait` and `MessageTimeout` (defaults 30 and 10, read_config.h):
    #: together the window in which one preemptor does not preempt again.
    kill_wait: float = 30.0
    message_timeout: float = 10.0
    #: Lab-only: share of a preempted run's progress a requeued job keeps.
    checkpoint_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.preempt_type not in PREEMPT_TYPES:
            raise ValueError(f"PreemptType={self.preempt_type!r} is not one of {PREEMPT_TYPES}")
        # read_config.c refuses both mismatches; slurmctld would not start.
        # For preempt/none it masks only GANG (`preempt_mode &
        # ~PREEMPT_MODE_GANG`, L4711-L4719), so WITHIN or PRIORITY there is
        # refused too.
        if self.preempt_type == "preempt/none" and (
            self.mode.base != "OFF" or self.mode.within or self.mode.priority
        ):
            raise ValueError("PreemptType and PreemptMode values incompatible")
        if self.preempt_type != "preempt/none" and self.mode.base == "OFF":
            raise ValueError(f"{self.preempt_type} is not compatible with PreemptMode=OFF")
        if not 0.0 <= self.checkpoint_fraction < 1.0:
            raise ValueError("checkpoint_fraction must be in [0, 1)")
        if self.reorder_count < 0:
            raise ValueError("reorder_count must be >= 0")
        for value, name in (
            (self.requeue_delay, "requeue_delay"),
            (self.kill_wait, "KillWait"),
            (self.message_timeout, "MessageTimeout"),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for spec in self.partitions.values():
            if spec.grace_time < 0:
                raise ValueError(f"partition {spec.name}: GraceTime must be >= 0")
        for q in self.qos.values():
            if q.grace_time < 0:
                raise ValueError(f"QOS {q.name}: GraceTime must be >= 0")
        if not (0.0 <= self.exempt_time < math.inf):
            # -1, INFINITE, UNLIMITED: "disables the option, equivalent to 0".
            object.__setattr__(self, "exempt_time", 0.0)
        loop = _preempt_loop(self.qos)
        if loop is not None:
            raise ValueError(
                "QOS Preempt lists form a loop (" + " -> ".join(loop) + "); slurmdbd "
                "refuses that (_preemption_loop(), as_mysql_qos.c)"
            )

    @property
    def enabled(self) -> bool:
        return self.preempt_type != "preempt/none"

    # ── Lookups ─────────────────────────────────────────────────────────────

    def partition(self, job: Job) -> PartitionSpec:
        return self.partitions.get(job.partition) or PartitionSpec(job.partition)

    def qos_of(self, job: Job) -> QOSSpec | None:
        return self.qos.get(job.qos) if job.qos is not None else None

    def tier(self, job: Job) -> int:
        return self.partition(job).priority_tier

    def mode_of(self, job: Job) -> BaseMode:
        """The mechanism used on `job` if it is preempted (`preempt_p_get_mode`)."""
        if self.preempt_type == "preempt/partition_prio":
            part = self.partition(job).preempt_mode
            return (part or self.mode).base
        if self.preempt_type == "preempt/qos":
            q = self.qos_of(job)
            # `preempt_p_get_mode()` (preempt_qos.c L66-L80): a non-zero QOS
            # value replaces the cluster's before GANG, PRIORITY and WITHIN
            # are stripped. A bare OFF is 0, the same as CLUSTER (sacctmgr(1)),
            # and defers; `OFF,WITHIN` or a bare `WITHIN` does not, and
            # resolves to OFF, which `_is_job_preempt_exempt()` (preempt.c
            # L164) reads as "may not be preempted".
            if q is not None and q.preempt_mode is not None and not q.preempt_mode.is_unset:
                return q.preempt_mode.base
            return self.mode.base
        return "OFF"

    def grace_of(self, job: Job) -> float:
        """`preempt_p_get_grace_time()`: the preemptee's partition or QOS."""
        if self.preempt_type == "preempt/partition_prio":
            return self.partition(job).grace_time
        if self.preempt_type == "preempt/qos":
            q = self.qos_of(job)
            return q.grace_time if q is not None else 0.0
        return 0.0

    def exempt_until(self, job: Job) -> float:
        """Earliest time `job` may be preempted (`acct_policy_get_preemptable_time`).

        That function (acct_policy.c L5203-L5226) never looks at PreemptType:
        the job's QOS value, then the global one, applies under every plugin,
        partition_prio included. (Its partition-QOS level is not modelled:
        partitions here have no QOS.)
        """
        assert job.start_time is not None
        q = self.qos_of(job)
        own = q.preempt_exempt_time if q is not None else None
        if own is not None and 0.0 <= own < math.inf:
            return job.start_time + own
        return job.start_time + self.exempt_time

    def _priority_flag(self, preemptor: Job) -> bool:
        if self.mode.priority:
            return True
        if self.preempt_type == "preempt/partition_prio":
            part = self.partition(preemptor).preempt_mode
            return part is not None and part.priority
        q = self.qos_of(preemptor)
        return q is not None and q.preempt_mode is not None and q.preempt_mode.priority

    # ── Who may preempt whom ────────────────────────────────────────────────

    def preemptable(self, preemptee: Job, preemptor: Job) -> bool:
        """`preempt_p_preemptable(preemptee, preemptor)` for the active plugin."""
        if self.preempt_type == "preempt/partition_prio":
            if self._priority_flag(preemptor) and preemptor.priority < preemptee.priority:
                return False
            ee = self.partition(preemptee)
            if ee.priority_tier >= self.tier(preemptor):
                return False
            return not (ee.preempt_mode is not None and ee.preempt_mode.base == "OFF")
        if self.preempt_type == "preempt/qos":
            q_ee, q_or = self.qos_of(preemptee), self.qos_of(preemptor)
            if q_ee is None or q_or is None:
                return False
            if q_or.name == q_ee.name:
                within = self.mode.within or (
                    q_or.preempt_mode is not None and q_or.preempt_mode.within
                )
                return within and preemptor.priority > preemptee.priority
            if q_ee.name not in q_or.preempt:
                return False
            if self._priority_flag(preemptor):
                return preemptor.priority > preemptee.priority
            return True
        return False

    def sorts_before(self, a: Job, b: Job) -> bool:
        """Queue order's "jobs that can preempt" test (`preempt_g_job_preempt_check`).

        For qos it is `preemptable(b, a)`. For partition_prio it is
        `preempt_p_job_preempt_check()`: the tier comparison without the
        preemptee's `PreemptMode=OFF` test.
        """
        if self.preempt_type == "preempt/qos":
            return self.preemptable(b, a)
        if self.preempt_type == "preempt/partition_prio":
            if self._priority_flag(a) and a.priority < b.priority:
                return False
            return self.tier(a) > self.tier(b)
        return False

    def preempt_prio(self, job: Job) -> int:
        """`preempt_p_get_prio()`: lower is preempted first."""
        if self.preempt_type == "preempt/qos":
            q = self.qos_of(job)
            upper = min(q.priority, 0xFFFF) if q is not None else 0
        else:
            upper = min(self.tier(job), 0xFFFF)
        return (upper << 16) + min(job.nodes, 0xFFFF)

    def candidates(self, preemptor: Job, running: Iterable[Job], now: float) -> list[Job]:
        """Running jobs `preemptor` could preempt now, in preemption order."""
        found = [
            j
            for j in running
            if j is not preemptor
            and j.start_time is not None
            and self.preemptable(j, preemptor)
            and self.mode_of(j) in ("CANCEL", "REQUEUE")
            and now >= self.exempt_until(j)
        ]
        # Slurm's sort is not stable and its job ids are unique. Here ties go
        # to the lower id and then, for jobs sharing an id (sacct array
        # tasks, a hand-made trace), to the earlier original submit, so the
        # order never depends on the order of the `running` list.
        if self.youngest_first:
            found.sort(key=lambda j: (-(j.start_time or 0.0), j.job_id, j.submit_time))
        else:
            found.sort(key=lambda j: (self.preempt_prio(j), j.job_id, j.submit_time))
        return found


def queue_order(pending: Sequence[Job], config: PreemptionConfig) -> list[Job]:
    """Slurm's queue order when partitions or preemption are configured.

    Jobs that can preempt, then `PriorityTier`, then priority, then submit
    time, then job id. `simulate.queue_order()` uses the plain key when no
    config is given, which gives the same order when every tier is equal.
    """

    def cmp(a: Job, b: Job) -> int:
        if config.enabled:
            if config.sorts_before(a, b):
                return -1
            if config.sorts_before(b, a):
                return 1
        ta, tb = config.tier(a), config.tier(b)
        if ta != tb:
            return -1 if ta > tb else 1
        # Job id is Slurm's last rule; ids are unique there. The original
        # submit time after it only orders jobs sharing an id (see candidates).
        ka = (-a.priority, a.sched_submit_time, a.job_id, a.submit_time)
        kb = (-b.priority, b.sched_submit_time, b.job_id, b.submit_time)
        return -1 if ka < kb else (1 if ka > kb else 0)

    return sorted(pending, key=functools.cmp_to_key(cmp))


def _first_fit(job: Job, free: list[list[int]]) -> list[int] | None:
    chosen: list[int] = []
    for nid, (cpus, gpus) in enumerate(free):
        if cpus >= job.cpus_per_node and gpus >= job.gpus_per_node:
            chosen.append(nid)
            if len(chosen) == job.nodes:
                return chosen
    return None


def select_preemptees(
    preemptor: Job,
    candidates: Sequence[Job],
    cluster: Cluster,
    config: PreemptionConfig,
) -> tuple[list[Job], list[int]] | None:
    """Which running jobs to preempt so `preemptor` fits, or None if none do.

    A port of the REQUEUE/CANCEL branch of `_run_now()` in cons_tres's
    job_test.c, with this simulator's first-fit standing in for the select
    plugin's node test. Returns the preemptees and the nodes the preemptor
    would get in the hypothetical state. Those nodes are not reserved: the
    preemptor starts later, on whatever a later pass finds free.
    """
    order = [c for c in candidates if config.mode_of(c) in ("CANCEL", "REQUEUE")]
    if not order:
        return None
    # Keyed by the job itself, not its id: ids need not be unique, and two
    # candidates sharing one would overwrite each other's entry here, which
    # made the final loop stop at the first victim and preempt nothing.
    usable: dict[Job, int] = {}
    pass_count = 0
    count = len(order)
    while True:
        free = [[n.free_cpus, n.free_gpus] for n in cluster.nodes]
        hit: tuple[int, list[int]] | None = None
        for i, cand in enumerate(order):
            for nid in cand.allocated:
                free[nid][0] += cand.cpus_per_node
                free[nid][1] += cand.gpus_per_node
            nodes = _first_fit(preemptor, free)
            usable[cand] = 0
            if nodes is not None:
                hit = (i, nodes)
                break
        if hit is None:
            return None
        i, nodes = hit
        previous = pass_count
        pass_count += 1
        if previous > config.reorder_count or count <= pass_count:
            for later in order[i + 1 :]:
                usable[later] = 1
            break
        # Re-sort and search again, trying to preempt fewer jobs.
        last = order[i]
        if config.strict_order:
            order = [last] + [j for j in order if j is not last]
        else:
            chosen = set(nodes)
            usable[last] = _COMPLETED_FIT
            for j in order[:i]:
                usable[j] = len(chosen.intersection(j.allocated))
            for j in order[i + 1 :]:
                usable[j] = 0
            # Python's sort is stable; Slurm's is not (ties are unordered there).
            order = sorted(order, key=lambda j: -usable[j])

    # `_foreach_run_now_preemptee()`: removed jobs on the chosen nodes, stopping
    # at the first job the final search never needed.
    chosen = set(nodes)
    preemptees: list[Job] = []
    for j in order:
        if not chosen.intersection(j.allocated):
            continue
        if usable.get(j, 0):
            break
        preemptees.append(j)
    return preemptees, nodes


def qos_ladder(priorities: Mapping[str, int], grace_time: float = 0.0) -> dict[str, QOSSpec]:
    """One QOS per level; each may preempt every QOS with a lower priority.

    The shape Kubernetes' default `PreemptLowerPriority` policy implies, and
    what the S0 adapter builds from k8s priorities.
    """
    return {
        name: QOSSpec(
            name=name,
            priority=prio,
            preempt=frozenset(n for n, p in priorities.items() if p < prio),
            grace_time=grace_time,
        )
        for name, prio in priorities.items()
    }


def partition_ladder(tiers: Mapping[str, int], grace_time: float = 0.0) -> dict[str, PartitionSpec]:
    """One partition per level, `PriorityTier` = the level, all over every node."""
    return {
        name: PartitionSpec(name=name, priority_tier=tier, grace_time=grace_time)
        for name, tier in tiers.items()
    }
