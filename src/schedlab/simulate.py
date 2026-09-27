"""Discrete-event scheduling simulator: EASY or Slurm-like conservative backfill.

The scheduler is only ever allowed to read `job.time_limit`, never
`job.duration`. That asymmetry is deliberate and is the point of the whole
exercise: backfill plans against what users *request*, so systematically
over-requesting wall-clock degrades everyone's queue even when the cluster is
half idle. Letting the simulator peek at real runtimes would hide that.

Exactly one function reads `duration`: `_world_end_time()`, which plays the
part of the job itself finishing. It schedules the completion *event*; the
scheduler learns the job has ended only when that event fires. A job whose
runtime exceeds its own `time_limit` is refused there: Slurm would kill it at
the limit (state TIMEOUT), which is not modelled, and a plan that believes the
job has ended while it still holds its nodes cannot be trusted.

There is no wall clock and no time compression. Simulated time jumps from one
event to the next — arrivals, completions, and the periodic timers Slurm runs
on (`PriorityCalcPeriod`, `sched_interval`, `bf_interval`), which are events
here too. Every interval is therefore in simulated seconds by construction and
needs no scaling (contrast the sibling k8s lab's `--speedup`).

Two backfill modes:

``easy`` — the textbook model (Lifka, 1995) that 0.1.0 shipped. The
algorithm below is 0.1.0's; what changed around it since 0.1.0: priorities
come from `PriorityCalcPeriod` snapshots (and are Slurm's truncated integers),
the half-life default is 7 days, partition `PriorityTier` orders the queue
when partitions are configured, and the shadow time no longer mis-projects a
job that started at exactly t=0 (see `_shadow_time`):

  1. At every event, sort pending jobs by priority.
  2. Walk the queue; start whatever fits.
  3. At the first job that does not fit, compute its *shadow time* — the
     earliest moment enough resources free up, assuming every running job runs
     to its full time limit — and reserve for it.
  4. Any lower-priority job may still start, provided it either finishes before
     the shadow time or fits in resources the reservation does not need.

  Accounting is aggregate (total CPUs/GPUs), and it runs on every event with
  no depth limit — more thorough and more responsive than a real controller.

``conservative`` — Slurm's `sched/backfill` shape (see `backfill.py`):

  * an event-triggered main pass on each submit and completion, in strict
    priority order, which stops at the first job that cannot start (see
    `SlurmScheduler.main_pass` for why that is the whole pass, not just its
    partition), testing at most `default_queue_depth + 1` jobs;
  * a full main pass every `sched_interval` (at every event when it is 0;
    `sched_interval=-1` disables the main scheduler outright, leaving only
    backfill to start jobs);
  * a backfill cycle every `bf_interval` that plans every tested job into a
    per-node timeline and starts only what delays no higher-priority job's
    planned start — bounded by `bf_max_job_test`, `bf_window`, and friends.

  With `backfill=False` only the main passes run: `sched/builtin`.

  Preemption (`preempt.py`) is available in this mode only and is off by
  default. A preempted job keeps its allocation until its grace deadline, and
  that deadline is a second, cancellable completion event: whichever of the
  job's own end and the deadline comes first releases it.

Sources: https://slurm.schedmd.com/sched_config.html and slurm.conf(5).
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, get_args, overload

from . import preempt as preempt_mod
from .accounting import FairshareSnapshot, PriorityEngine
from .backfill import BackfillCycle, BackfillStats, NodeSpace
from .model import Cluster, Job, RunRecord, State
from .params import SchedulerParameters
from .preempt import PreemptionConfig, select_preemptees
from .priority import FairshareTree, PriorityWeights

BackfillMode = Literal["easy", "conservative"]
BACKFILL_MODES: tuple[str, ...] = get_args(BackfillMode)

#: Skip re-evaluating a backfill cycle when nothing it depends on has changed
#: since the last evaluated one. That needs three things:
#:
#: 1. the same queue in the same order;
#: 2. no job started, finished or was selected for preemption (the state
#:    version) — every running job's planned end is at or after its real
#:    release, so a planned start can only be reached through such a change;
#: 3. no job that was beyond `bf_window` has come inside it. The plan depends
#:    on `now` through `horizon = now + bf_window`: a job whose earliest start
#:    was beyond the window gets no reservation, but once the window reaches
#:    that start it does, and its new reservation can move a lower-priority
#:    job's slot or free the gap a sub-node job needed. So each evaluated
#:    cycle records the earliest start of any job it left beyond the window,
#:    and a later cycle is replayed only while `now + bf_window` is short of
#:    it. (Before this third condition, replay changed start times on
#:    sub-node workloads with a short window.)
#:
#: A replayed cycle would plan exactly what the evaluated one did, so its
#: statistics are copied. `test_replayed_cycles_match_full_evaluation` pins the
#: equivalence (start times, first planned starts and per-cycle statistics);
#: set False to evaluate every cycle.
REPLAY_UNCHANGED_BACKFILL_CYCLES = True

#: Within one evaluated cycle, treat a job as beyond `bf_window` without
#: planning it when an earlier job of no larger shape already was (see
#: `SlurmScheduler.backfill_cycle`). Exact, and pinned against the unpruned
#: planner by `test_replayed_cycles_match_full_evaluation`; set False to plan
#: every tested job.
PRUNE_DOMINATED_BEYOND_WINDOW = True

#: Completion-heap entry kinds. The job's own end, and a preemption deadline.
_WORLD_END = 0
_DEADLINE = 1


@dataclass
class Reservation:
    """The EASY reservation held for the highest-priority blocked job."""

    job_id: int
    shadow_time: float
    cpus: int
    gpus: int


@dataclass(frozen=True)
class PreemptionRecord:
    """One preemption, from selection to release. Post-hoc data only."""

    job_id: int
    #: The pending job whose scheduling attempt selected this one.
    preemptor_id: int
    start_time: float
    #: When the job was selected (Slurm's `preempt_time`).
    preempt_time: float
    #: When its resources were released: the grace deadline, or its own end if
    #: that came first.
    release_time: float
    #: What happened: "REQUEUE" or "CANCEL" (REQUEUE falls back to CANCEL when
    #: `JobRequeue=0`).
    mode: str
    #: The job's own end came before the deadline. Slurm still applies
    #: PreemptMode ("regardless of why it exited").
    exited_in_grace: bool
    cpus: int
    gpus: int
    #: Nodes the job held: the k8s lab counts one preemption per pod.
    pods: int = 1
    #: Seconds of progress kept by the checkpoint fraction (REQUEUE only).
    kept_seconds: float = 0.0

    @property
    def run_seconds(self) -> float:
        """Start to selection: the run preemption interrupted."""
        return self.preempt_time - self.start_time

    @property
    def grace_seconds(self) -> float:
        """Selection to release: nodes held, winding down, no credited progress."""
        return self.release_time - self.preempt_time

    @property
    def lost_seconds(self) -> float:
        """The interrupted run less what a checkpoint kept.

        The k8s lab's split, so S0 can be compared with it: lost is
        `(1 - c) x (selection - start)` and the grace period is counted
        separately as grace-locked, never in both. Slurm keeps running a job
        through its grace period, but the model credits no progress to it.
        """
        return self.run_seconds - self.kept_seconds


class NodeSampleView(Sequence[tuple[float, Mapping[str, int]]]):
    """`[(t, {node: free})]` over the compact per-node samples, built lazily.

    The simulator stores one tuple per state change; a dict per sample would
    cost memory proportional to nodes x events on a long trace.
    """

    def __init__(self, names: tuple[str, ...], samples: list[tuple[float, tuple[int, ...]]]):
        self._names = names
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, index: int) -> tuple[float, Mapping[str, int]]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[tuple[float, Mapping[str, int]]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[float, Mapping[str, int]] | Sequence[tuple[float, Mapping[str, int]]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        t, values = self._samples[index]
        return t, dict(zip(self._names, values, strict=True))


@dataclass
class SimulationResult:
    #: Jobs that ran. Excludes anything in `unschedulable`.
    jobs: list[Job]
    makespan: float
    #: One entry per backfill *start*. EASY: a job that started while a
    #: reservation was held (it jumped it). Conservative: a job started by a
    #: backfill cycle rather than a main pass — what `sdiag` counts as
    #: backfilled, once per start. A requeued job that backfill starts again
    #: appears again; count distinct jobs with `RunRecord.backfilled`.
    backfilled_ids: list[int] = field(default_factory=list)
    #: (time, busy_cpus) sampled at every processed event time.
    utilization_samples: list[tuple[float, int]] = field(default_factory=list)
    #: Jobs whose request exceeds the cluster outright. Slurm rejects these at
    #: submit time; surfacing them is more useful than failing the whole run,
    #: because on a real trace they usually mean the trace and the cluster
    #: definition disagree.
    unschedulable: list[Job] = field(default_factory=list)
    backfill_mode: BackfillMode = "easy"
    #: Conservative mode only: one record per backfill cycle.
    backfill_stats: BackfillStats = field(default_factory=BackfillStats)
    #: One snapshot per `PriorityCalcPeriod` tick, when fairshare is enabled.
    fairshare_snapshots: list[FairshareSnapshot] = field(default_factory=list)
    #: Conservative mode only: the first planned start each job was given by a
    #: backfill cycle — a prediction to compare with `start_time`.
    planned_starts: dict[int, float] = field(default_factory=dict)
    #: Conservative mode only: main-scheduler passes run.
    main_passes: int = 0
    #: The scheduler parameters in force (conservative mode).
    sched_params: SchedulerParameters | None = None
    #: The preemption settings in force, and one record per preemption.
    preemption: PreemptionConfig | None = None
    preemptions: list[PreemptionRecord] = field(default_factory=list)
    #: Node names, in node-id order: the keys of the per-node samples.
    node_names: tuple[str, ...] = ()
    #: `(t, free GPUs per node, free CPUs per node)` at trace t=0 (or the first
    #: submit, if earlier) and then at every event that changed any node. A
    #: step function: each state holds until the next sample. Dropping
    #: unchanged states leaves every left-point integral over it exact.
    node_samples: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )
    #: Simulated time of the last event (the last release), on the trace clock.
    horizon: float = 0.0

    def free_gpu_samples(self) -> NodeSampleView:
        """`[(t, {node: free GPUs})]`, the input Definition C takes."""
        return NodeSampleView(self.node_names, [(t, g) for t, g, _ in self.node_samples])

    def free_cpu_samples(self) -> NodeSampleView:
        return NodeSampleView(self.node_names, [(t, c) for t, _, c in self.node_samples])


def partition_schedulable(jobs: list[Job], cluster: Cluster) -> tuple[list[Job], list[Job]]:
    """Split jobs into those the cluster could ever run, and those it cannot."""
    runnable: list[Job] = []
    impossible: list[Job] = []

    for job in jobs:
        capable = sum(
            1
            for n in cluster.nodes
            if n.cpus >= job.cpus_per_node and n.gpus >= job.gpus_per_node
        )
        (runnable if capable >= job.nodes else impossible).append(job)

    return runnable, impossible


def queue_order(pending: list[Job], preemption: PreemptionConfig | None = None) -> list[Job]:
    """Priority order, then submit time, then job id.

    slurm.conf(5), PriorityType: "jobs are prioritized in the following order:
    1. Jobs that can preempt 2. Jobs with an advanced reservation 3. Partition
    PriorityTier 4. Job priority 5. Job submit time 6. Job ID". Rules 1 and 3
    apply when a `PreemptionConfig` with preemption or partitions is given
    (`preempt.queue_order`); advanced reservations are not modelled. A
    requeued job sorts by its reset submit time, as in Slurm. Ids are unique
    in Slurm; here they need not be, so jobs sharing one fall back to their
    original submit time, never to the order of the `pending` list.
    """
    if preemption is not None and (preemption.enabled or preemption.partitions):
        return preempt_mod.queue_order(pending, preemption)
    return sorted(
        pending, key=lambda j: (-j.priority, j.sched_submit_time, j.job_id, j.submit_time)
    )


def _world_end_time(job: Job, start: float) -> float:
    """When the job will really finish. The one reader of `job.duration`.

    This is the job running, not the scheduler deciding: its result goes
    straight onto the completion heap and is not stored on the job. A
    requeued job resumes from whatever its checkpoints kept.

    A run longer than the job's `time_limit` is refused. Slurm kills such a
    job at its limit (TIMEOUT), which this simulator does not model; running
    it past the limit would leave the scheduler planning around nodes it
    believes are free and crash on them. Every trace loader here produces
    `time_limit >= duration` (sacct keeps the recorded limit and cuts an
    overrun runtime to it; `trace.load_sacct`), so this only fires for
    hand-built jobs.
    """
    remaining = job.duration - job.saved_progress
    if remaining > job.time_limit:
        raise ValueError(
            f"job {job.job_id}: runtime {remaining:g} s exceeds its time limit "
            f"{job.time_limit:g} s; Slurm would kill it at the limit (TIMEOUT), which is "
            "not modelled. Give it time_limit >= duration."
        )
    return start + remaining


def _dispatch(cluster: Cluster, job: Job, node_ids: list[int], now: float) -> None:
    cluster.allocate(job, node_ids)
    job.state = State.RUNNING
    job.start_time = now
    if job.first_start_time is None:
        job.first_start_time = now
    job.dispatch_priority = job.priority
    job.runs.append(RunRecord(start=now, nodes=tuple(node_ids)))


class EasyScheduler:
    """Textbook EASY backfill with aggregate accounting (the 0.1.0 model)."""

    def __init__(
        self,
        cluster: Cluster,
        backfill: bool = True,
        order: PreemptionConfig | None = None,
    ) -> None:
        self.cluster = cluster
        self.backfill = backfill
        self.backfilled_ids: list[int] = []
        #: Partition table for PriorityTier ordering (preemption itself is not
        #: modelled in this mode).
        self.order = order

    # ── Reservation planning ────────────────────────────────────────────────

    def _shadow_time(self, blocked: Job, running: list[Job], now: float) -> float:
        """Earliest time `blocked` can start if every running job uses its full limit.

        Aggregate CPU/GPU accounting rather than per-node, so it can be
        optimistic when free CPUs are scattered across nodes. The conservative
        mode's per-node plan (`backfill.NodeSpace`) is the remedy.
        """
        free_cpus = sum(n.free_cpus for n in self.cluster.nodes)
        free_gpus = sum(n.free_gpus for n in self.cluster.nodes)

        need_cpus, need_gpus = blocked.total_cpus, blocked.total_gpus
        if free_cpus >= need_cpus and free_gpus >= need_gpus:
            return now

        # Release running jobs in projected-completion order until it fits.
        # (0.1.0 wrote `j.start_time or now`, which treats a start at t=0 as
        # "unknown" and slides that job's projected end forward with `now`.)
        def projected_end(j: Job) -> float:
            return (j.start_time if j.start_time is not None else now) + j.time_limit

        for job in sorted(running, key=projected_end):
            free_cpus += job.total_cpus
            free_gpus += job.total_gpus
            if free_cpus >= need_cpus and free_gpus >= need_gpus:
                return projected_end(job)

        # Nothing frees up enough: the job cannot run on this cluster at all.
        return float("inf")

    def _may_backfill(self, job: Job, reservation: Reservation, now: float) -> bool:
        """True if starting `job` now provably cannot delay the reserved job."""
        # Case 1: it finishes before the reservation needs anything.
        if now + job.time_limit <= reservation.shadow_time:
            return True

        # Case 2: it runs past the shadow time, so it must fit in the slack
        # left over after the reservation's resources are set aside.
        free_cpus = sum(n.free_cpus for n in self.cluster.nodes)
        free_gpus = sum(n.free_gpus for n in self.cluster.nodes)
        return (
            free_cpus - reservation.cpus >= job.total_cpus
            and free_gpus - reservation.gpus >= job.total_gpus
        )

    # ── Scheduling pass ─────────────────────────────────────────────────────

    def schedule(self, pending: list[Job], running: list[Job], now: float) -> list[Job]:
        """Start whatever can start. Returns the jobs dispatched this pass.

        Reads each job's current `priority`; it does not compute priorities.
        Started jobs are appended to `running`.
        """
        started: list[Job] = []
        reservation: Reservation | None = None

        for job in queue_order(pending, self.order):
            node_ids = self.cluster.find_nodes(job)

            if node_ids is not None:
                if reservation is not None and not self._may_backfill(job, reservation, now):
                    continue

                _dispatch(self.cluster, job, node_ids, now)
                if reservation is not None:
                    self.backfilled_ids.append(job.job_id)
                    job.runs[-1].backfilled = True
                started.append(job)
                running.append(job)
                continue

            # Does not fit. The first such job earns the reservation.
            if reservation is None:
                if not self.backfill:
                    break  # strict FIFO-by-priority: head-of-line blocking
                shadow = self._shadow_time(job, running, now)
                if shadow == float("inf"):
                    continue  # unschedulable on this cluster; don't block others
                reservation = Reservation(job.job_id, shadow, job.total_cpus, job.total_gpus)

        return started


#: 0.1.0 name, kept so existing imports keep working.
Scheduler = EasyScheduler


class SlurmScheduler:
    """Slurm's main scheduler plus `sched/backfill`, on simulated timers."""

    def __init__(
        self,
        cluster: Cluster,
        params: SchedulerParameters | None = None,
        backfill: bool = True,
        preemption: PreemptionConfig | None = None,
    ) -> None:
        self.cluster = cluster
        self.params = params or SchedulerParameters()
        self.backfill = backfill and self.params.backfill_enabled
        # Read once per run from the module switch, which the equivalence
        # tests flip (there is no second way to set it).
        self.replay_unchanged = REPLAY_UNCHANGED_BACKFILL_CYCLES
        self.preemption = preemption if preemption is not None and preemption.enabled else None
        self.stats = BackfillStats()
        self.backfilled_ids: list[int] = []
        self.first_planned_start: dict[int, float] = {}
        self.main_passes = 0
        #: Replay key of the last evaluated cycle: the queue (by job identity,
        #: so two jobs sharing an id are still told apart) and the state version.
        self._last_key: tuple[tuple[int, ...], int] | None = None
        self._last_cycle: BackfillCycle | None = None
        #: Earliest start of any job the last evaluated cycle left beyond
        #: `bf_window`. Replay stops once `now + bf_window` reaches it.
        self._window_entry = math.inf
        #: Jobs newly selected for preemption; the simulator drains this to
        #: schedule their grace deadlines.
        self.signalled: list[Job] = []
        #: Preemptee -> the job id of the preemptor that selected it. Keyed by
        #: the job itself: ids are not guaranteed unique.
        self.preemptor_of: dict[Job, int] = {}
        #: Preemptor -> when it last preempted (`preempt_start_time`).
        self._preempt_started: dict[Job, float] = {}

    # ── Main scheduler ──────────────────────────────────────────────────────

    def main_pass(
        self,
        queue: list[Job],
        now: float,
        depth: int | None,
        running: list[Job] | None = None,
    ) -> list[Job]:
        """One main-scheduler pass over `queue` (already in priority order).

        Stops at the first job that cannot start. Slurm marks that job's
        partition failed and removes the partition's nodes from the pass —
        "Do not schedule more jobs in this partition or on nodes in this
        partition" (`_schedule()`, job_scheduler.c, SchedMD/slurm@9f9da53
        L2082-L2137). Every partition spans every node in this simulator, so
        nothing later in the pass could start: the pass ends. (Until the
        preemption work this stopped only the blocked job's partition, which
        let a job in another partition start on the same nodes.)

        `depth` is `default_queue_depth` for event-triggered passes and None
        for the periodic `sched_interval` pass. As in `_schedule()`, which
        breaks on `job_depth++ > def_job_limit` (job_scheduler.c, same
        commit, L1578), a pass tests up to `depth + 1` jobs: 101 by default,
        and one with `default_queue_depth=0`. Jobs skipped for
        `partition_job_depth` are skipped before that count, so they do not
        use it up (slurm.conf(5); L1564-L1577).

        With preemption enabled and `running` given, a job that cannot start
        may select running jobs to preempt (`preempt.py`). It still does not
        start in this pass.
        """
        self.main_passes += 1
        started: list[Job] = []
        per_partition: dict[str, int] = {}
        partition_depth = self.params.partition_job_depth
        considered = 0

        for job in queue:
            if depth is not None and considered > depth:
                break
            if partition_depth and per_partition.get(job.partition, 0) >= partition_depth:
                continue
            considered += 1
            per_partition[job.partition] = per_partition.get(job.partition, 0) + 1

            node_ids = self.cluster.find_nodes(job)
            if node_ids is None:
                if self.preemption is not None and running is not None:
                    self._preempt_for(job, [*running, *started], now)
                break
            _dispatch(self.cluster, job, node_ids, now)
            started.append(job)
        return started

    def _preempt_for(self, preemptor: Job, running: list[Job], now: float) -> None:
        """Select and signal preemptees for `preemptor`, as `select_nodes()` does."""
        cfg = self.preemption
        assert cfg is not None
        candidates = cfg.candidates(preemptor, running, now)
        if not candidates:
            return
        picked = select_preemptees(preemptor, candidates, self.cluster, cfg)
        if picked is None or not picked[0]:
            return
        # node_scheduler.c: within KillWait + MessageTimeout of this job's last
        # preemption, "Job preemption may still be in progress, do not cancel
        # or requeue any more jobs yet".
        last = self._preempt_started.get(preemptor)
        if last is not None and last > now - (cfg.kill_wait + cfg.message_timeout):
            return
        self._preempt_started[preemptor] = now
        for victim in picked[0]:
            if victim.preempt_time is not None:
                continue  # already in its grace period: slurm_job_preempt() is a no-op
            victim.preempt_time = now
            victim.preempt_deadline = min(victim.expected_end, now + cfg.grace_of(victim))
            victim.preempt_count += 1
            victim.runs[-1].preempt_time = now
            self.preemptor_of[victim] = preemptor.job_id
            self.signalled.append(victim)

    def drain_signalled(self) -> list[Job]:
        out, self.signalled = self.signalled, []
        return out

    # ── Backfill ────────────────────────────────────────────────────────────

    def backfill_cycle(
        self, queue: list[Job], running: list[Job], now: float, state_version: int
    ) -> list[Job]:
        """One `sched/backfill` cycle. `queue` is in priority order."""
        if not queue:
            return []
        p = self.params
        key = (tuple(id(j) for j in queue), state_version)
        if (
            self.replay_unchanged
            and key == self._last_key
            and self._last_cycle is not None
            and now + p.bf_window < self._window_entry
        ):
            self.stats.cycles.append(replace(self._last_cycle, time=now, replayed=True))
            return []

        plan = NodeSpace(self.cluster, running, p.bf_resolution)
        # backfill.c compares a job's planned start with sched_start +
        # backfill_window (L3700), unquantised.
        horizon = now + p.bf_window
        counts: dict[tuple[str, object], int] = {}
        started: list[Job] = []
        tested = reservations = beyond = skipped = 0
        hit_test = hit_start = False
        entry = math.inf  # earliest start of any job left beyond the window
        # Shapes (cpus/node, gpus/node, nodes, time limit) found beyond the
        # window this cycle. The plan only gains reservations and starts as
        # the cycle goes on, so a job's earliest start can only move later;
        # and a job at least as large in every dimension cannot start
        # earlier than a smaller one. So a job dominating any of these is
        # beyond too, and its earliest start is no earlier than the smaller
        # job's, so it cannot lower `entry` either. Exact, and it saves the
        # planner's most expensive calls on a deep queue.
        beyond_shapes: list[tuple[int, int, int, float]] = []

        for job in queue:
            if tested >= p.bf_max_job_test:
                hit_test = True
                break
            if self._over_limits(job, counts):
                skipped += 1
                continue
            tested += 1

            jc, jg, jn, jl = job.cpus_per_node, job.gpus_per_node, job.nodes, job.time_limit
            if any(c <= jc and g <= jg and n <= jn and t <= jl for c, g, n, t in beyond_shapes):
                found = None
            else:
                found = plan.earliest_start(job, now, horizon)
                if found is None:
                    if PRUNE_DOMINATED_BEYOND_WINDOW:
                        # Keep the list minimal: shapes this one is no larger
                        # than are now redundant.
                        beyond_shapes = [
                            (c, g, n, t)
                            for c, g, n, t in beyond_shapes
                            if not (jc <= c and jg <= g and jn <= n and jl <= t)
                        ]
                        beyond_shapes.append((jc, jg, jn, jl))
                    if self.replay_unchanged:
                        # When would it come inside the window? Only the
                        # smallest such time matters, so search no further.
                        later = plan.earliest_start(job, now, entry)
                        if later is not None:
                            entry = min(entry, later[0])
            if found is None:
                beyond += 1
                job.planned_start = None
                continue
            when, node_ids = found
            if when <= now:
                _dispatch(self.cluster, job, node_ids, now)
                plan.start(job, node_ids, now)
                self.backfilled_ids.append(job.job_id)
                job.runs[-1].backfilled = True
                started.append(job)
                if p.bf_max_job_start and len(started) >= p.bf_max_job_start:
                    hit_start = True
                    break
            else:
                plan.reserve(job, node_ids, when)
                reservations += 1
                job.planned_start = when
                self.first_planned_start.setdefault(job.job_id, when)

        cycle = BackfillCycle(
            time=now,
            queue_length=len(queue),
            jobs_tested=tested,
            jobs_started=len(started),
            reservations=reservations,
            beyond_window=beyond,
            skipped_by_limits=skipped,
            hit_max_job_test=hit_test,
            hit_max_job_start=hit_start,
        )
        self.stats.cycles.append(cycle)
        self._last_key = None if started else key
        self._last_cycle = cycle
        self._window_entry = entry
        return started

    def _over_limits(self, job: Job, counts: dict[tuple[str, object], int]) -> bool:
        """Per-user/partition/association caps on jobs checked per cycle.

        As in `_job_exceeds_max_bf_param()` (backfill.c L1797-L1899): a job over any cap
        is skipped without counting towards `bf_max_job_test`; otherwise every
        applicable counter is incremented. When both `bf_max_job_assoc` and
        `bf_max_job_user` are set, the association cap wins and the per-user
        cap is dropped, as `_load_config()` does (L974-L979: "Both
        bf_max_job_user and bf_max_job_assoc are set: bf_max_job_assoc taking
        precedence"). `SchedulerParameters.parse` already stores it that way
        and warns; this covers parameters built directly.
        """
        p = self.params
        caps: list[tuple[tuple[str, object], int]] = []
        if p.bf_max_job_user_part:
            caps.append((("user_part", (job.owner, job.partition)), p.bf_max_job_user_part))
        if p.bf_max_job_part:
            caps.append((("part", job.partition), p.bf_max_job_part))
        if p.bf_max_job_assoc:
            caps.append((("assoc", (job.owner, job.account)), p.bf_max_job_assoc))
        elif p.bf_max_job_user:
            caps.append((("user", job.owner), p.bf_max_job_user))
        if any(counts.get(k, 0) >= limit for k, limit in caps):
            return True
        for k, _ in caps:
            counts[k] = counts.get(k, 0) + 1
        return False


def _next_on_grid(t: float, origin: float, period: float) -> float:
    """First `origin + k*period` strictly after `t`.

    Computed from `k` rather than by repeated addition so timers never drift,
    and nudged forward because `(origin + k*period - origin) / period` can
    round to just under `k` — which would return `t` itself and stall time.
    """
    k = math.floor((t - origin) / period) + 1
    while origin + k * period <= t:
        k += 1
    return origin + k * period


def _grid_at_or_after(t: float, origin: float, period: float) -> float:
    k = math.ceil((t - origin) / period)
    while origin + k * period < t:
        k += 1
    while k > 0 and origin + (k - 1) * period >= t:
        k -= 1
    return origin + k * period


def simulate(
    jobs: list[Job],
    cluster: Cluster,
    weights: PriorityWeights | None = None,
    fairshare: FairshareTree | None = None,
    partitions: dict[str, float] | None = None,
    backfill: bool = True,
    *,
    backfill_mode: BackfillMode = "conservative",
    sched_params: SchedulerParameters | None = None,
    calendar_offset: float = 0.0,
    preemption: PreemptionConfig | None = None,
) -> SimulationResult:
    """Run `jobs` through `cluster` to completion.

    `backfill_mode` selects the EASY model or the Slurm-like conservative one
    (the default). `sched_params` bounds the conservative mode and is ignored
    by EASY. `calendar_offset` places simulated t=0 relative to a Sunday 00:00
    for DAILY/WEEKLY `PriorityUsageResetPeriod`. `preemption` enables Slurm
    preemption (conservative mode only); its partition table also sets the
    `PriorityTier` queue order.

    Jobs larger than the cluster are set aside rather than raising, and are
    reported on the result. Job ids need not be unique (sacct array tasks
    share a parent id): the simulator tells jobs apart by identity. Only the
    id-keyed views on the result (`planned_starts`, `backfilled_ids`) are
    ambiguous then.
    """
    if backfill_mode not in BACKFILL_MODES:
        raise ValueError(f"backfill_mode must be one of {BACKFILL_MODES}, got {backfill_mode!r}")
    weights = weights or PriorityWeights()
    params = sched_params or SchedulerParameters()
    engine = PriorityEngine(cluster, weights, fairshare, partitions, calendar_offset)
    easy = backfill_mode == "easy"
    preempt_on = preemption is not None and preemption.enabled
    if preempt_on and easy:
        raise ValueError("preemption is modelled in the conservative (Slurm) mode only")
    main_on = params.main_scheduler_enabled
    if not easy and not main_on and not (backfill and params.backfill_enabled):
        # slurmctld would accept this and start no batch job, ever.
        raise ValueError(
            "sched_interval=-1 disables the main scheduler and backfill is off too: "
            "no job could ever start"
        )
    order_cfg = preemption if preemption is not None and (
        preemption.enabled or preemption.partitions
    ) else None
    names = tuple(n.name for n in cluster.nodes)
    if len(set(names)) != len(names):
        raise ValueError("node names must be unique")

    runnable, impossible = partition_schedulable(jobs, cluster)
    if not runnable:
        return SimulationResult(
            jobs=[],
            makespan=0.0,
            unschedulable=impossible,
            backfill_mode=backfill_mode,
            preemption=preemption,
            node_names=names,
        )

    arrivals = sorted(runnable, key=lambda j: (j.submit_time, j.job_id))
    origin = arrivals[0].submit_time
    period = weights.calc_period

    easy_sched = EasyScheduler(cluster, backfill, order_cfg) if easy else None
    slurm = None if easy else SlurmScheduler(cluster, params, backfill, preemption=preemption)

    inf = math.inf
    calc_next = origin if period > 0 else inf
    # sched_interval > 0 is a timer. 0 is not: slurmctld's background loop
    # asks for a full pass whenever `now - last_full_sched_time >=
    # sched_interval` (controller.c L2939), so with 0 every iteration (about
    # once a second) is a full pass. State only changes at events here, so
    # that is a full pass at every event (`every_event_full`).
    sched_next = origin if slurm and params.sched_interval_enabled else inf
    every_event_full = slurm is not None and main_on and params.sched_interval == 0
    bf_next = origin if slurm and slurm.backfill else inf

    pending: list[Job] = []
    running: list[Job] = []
    # Heap of (time, job id, sequence, run epoch, kind, job). A job's entries
    # for an earlier run (a different epoch), or for a run already released
    # by its other entry, are stale and skipped when popped. The sequence is
    # unique, so a tuple comparison never reaches the job.
    completions: list[tuple[float, int, int, int, int, Job]] = []
    sequence = itertools.count()
    # Run epoch per job, keyed by the job itself (identity: `Job` is
    # eq=False). Keyed by id, two jobs sharing one — sacct array tasks — made
    # each other's completions look stale, and the run never ended.
    epochs: dict[Job, int] = {}
    records: list[PreemptionRecord] = []
    samples: list[tuple[float, int]] = []

    def launch(started_now: list[Job], t: float) -> None:
        """Put each just-started run's end on the heap, and start its accrual.

        Called right after the pass that started them, so a job preempted
        later in the same pass (possible when QOS preempt lists do not form a
        ladder, see `preempt.queue_order`) already has its run epoch.
        """
        for job in started_now:
            epoch = epochs[job] = epochs.get(job, 0) + 1
            heapq.heappush(
                completions,
                (_world_end_time(job, t), job.job_id, next(sequence), epoch, _WORLD_END, job),
            )
            engine.job_started(job, t)

    def node_state() -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(n.free_gpus for n in cluster.nodes),
            tuple(n.free_cpus for n in cluster.nodes),
        )

    # Samples start at trace t=0, idle, as the k8s lab's reference model's do.
    last_state = node_state()
    node_samples = [(min(0.0, origin), *last_state)]
    ordered: list[Job] | None = None  # cached queue_order(pending)
    state_version = 0  # bumps on every start, release and preemption
    stalled = 0
    next_arrival = 0
    now = origin
    previous_now = -inf

    def release(job: Job, t: float, kind: int) -> bool:
        """Free `job`'s nodes at `t`. True if it went back to the queue."""
        cluster.release(job)
        running.remove(job)
        engine.job_ended(job, t)
        run = job.runs[-1]
        run.end = t
        if job.preempt_time is None:
            run.outcome = "completed"
            job.state = State.COMPLETED
            job.end_time = t
            return False

        # Selected for preemption: handled per PreemptMode "regardless of why
        # it exited" (slurm.conf(5), GraceTime).
        assert preemption is not None and slurm is not None and job.start_time is not None
        requeue = preemption.mode_of(job) == "REQUEUE" and preemption.job_requeue
        interrupted = job.preempt_time - job.start_time
        kept = preemption.checkpoint_fraction * interrupted if requeue else 0.0
        records.append(
            PreemptionRecord(
                job_id=job.job_id,
                preemptor_id=slurm.preemptor_of.pop(job, -1),
                start_time=job.start_time,
                preempt_time=job.preempt_time,
                release_time=t,
                mode="REQUEUE" if requeue else "CANCEL",
                exited_in_grace=kind == _WORLD_END,
                cpus=job.total_cpus,
                gpus=job.total_gpus,
                pods=job.nodes,
                kept_seconds=kept,
            )
        )
        job.preempt_time = None
        job.preempt_deadline = None
        if not requeue:
            run.outcome = "cancelled"
            job.state = State.PREEMPTED
            job.end_time = t
            return False
        # batch_requeue_fini(), job_mgr.c: a new submit time (never equal to
        # the old one), begin time after requeue_delay, accrual restarted, and
        # the priority recomputed.
        run.outcome = "requeued"
        job.saved_progress += kept
        job.state = State.PENDING
        job.start_time = None
        job.planned_start = None
        job.requeue_count += 1
        job.requeue_submit_time = t if t != job.sched_submit_time else t + 1
        job.eligible_time = t + preemption.requeue_delay + 1
        engine.submit(job, t)
        pending.append(job)
        return True

    while True:
        # ── Events at `now`: releases, then arrivals, then the decay tick.
        completed = False
        requeued = False
        while completions and completions[0][0] <= now:
            end, _, _, epoch, kind, job = heapq.heappop(completions)
            if job.state is not State.RUNNING or epochs.get(job) != epoch:
                continue  # the run this entry belonged to was already released
            requeued = release(job, end, kind) or requeued
            completed = True
            state_version += 1

        arrived = False
        while next_arrival < len(arrivals) and arrivals[next_arrival].submit_time <= now:
            job = arrivals[next_arrival]
            next_arrival += 1
            pending.append(job)
            engine.submit(job, now)
            arrived = True

        # A requeued job's begin time passing is treated like a submit: it
        # triggers an event pass. (Whether slurmctld runs one then, or waits
        # for its periodic pass, is unverified; the difference is bounded by
        # sched_interval.)
        woke = preempt_on and any(
            j.eligible_time is not None and previous_now < j.eligible_time <= now
            for j in pending
        )

        ticked = False
        if calc_next <= now:
            engine.tick(now, pending, running)
            calc_next = _next_on_grid(now, origin, period)
            ticked = True
        elif period == 0 and pending:
            # The idealised mode refreshes before every scheduling decision:
            # event passes, and also the periodic `sched_interval` pass, the
            # backfill cycle and a requeued job's begin time, which run on
            # iterations with no submit or completion (these timers are events
            # here; see the module docstring). Ticking on submits and
            # completions only left those passes sorting by priorities from
            # the last such event. Nothing pending means nothing to order.
            engine.tick(now, pending, running)
            ticked = True
        if arrived or ticked or requeued:
            ordered = None

        # ── Scheduling passes.
        started: list[Job] = []
        passed = False
        bf_deferred = False
        if pending:
            if ordered is None:
                ordered = queue_order(pending, order_cfg)
            if easy_sched is not None:
                if completed or arrived or ticked:
                    started = easy_sched.schedule(ordered, running, now)
                    launch(started, now)
                    passed = True
            elif slurm is not None:
                queue = [j for j in ordered if j.eligible(now)] if preempt_on else ordered
                # sched_interval=-1: `_schedule()` returns before doing
                # anything, for event-triggered and periodic calls alike
                # (job_scheduler.c L1360-L1372), so only backfill starts jobs.
                if main_on and (sched_next <= now or every_event_full):
                    started = slurm.main_pass(queue, now, depth=None, running=running)
                    passed = True
                elif main_on and (completed or arrived or woke):
                    started = slurm.main_pass(
                        queue, now, depth=params.default_queue_depth, running=running
                    )
                    passed = True
                running.extend(started)
                launch(started, now)
                for victim in slurm.drain_signalled():
                    assert victim.preempt_deadline is not None
                    heapq.heappush(
                        completions,
                        (
                            victim.preempt_deadline,
                            victim.job_id,
                            next(sequence),
                            epochs[victim],
                            _DEADLINE,
                            victim,
                        ),
                    )
                    state_version += 1
                    # A zero-grace release is due now. Let it happen (next
                    # loop, same instant) before a backfill cycle plans around
                    # nodes that are about to be free but are not yet.
                    bf_deferred = bf_deferred or victim.preempt_deadline <= now
                if bf_next <= now and not bf_deferred:
                    if started:
                        state_version += 1
                        queue = [j for j in queue if j.state is State.PENDING]
                    more = slurm.backfill_cycle(queue, running, now, state_version)
                    running.extend(more)
                    launch(more, now)
                    started = started + more
                    passed = True

        if sched_next <= now:
            sched_next = _next_on_grid(now, origin, params.sched_interval)
        if bf_next <= now and not bf_deferred:
            bf_next = _next_on_grid(now, origin, params.bf_interval)

        if started:
            state_version += 1
            pending = [j for j in pending if j.state is State.PENDING]
            if ordered is not None:
                ordered = [j for j in ordered if j.state is State.PENDING]

        samples.append((now, cluster.busy_cpus))
        state = node_state()
        if state != last_state:
            node_samples.append((now, *state))
            last_state = state

        # A pass over an idle cluster that starts nothing, repeatedly, is a
        # scheduler bug rather than a property of the workload. Requeued jobs
        # still waiting out their begin time do not count.
        waiting = [j for j in pending if j.eligible(now)] if preempt_on else pending
        if passed and not started and not running and waiting:
            stalled += 1
            if stalled >= 3:
                raise RuntimeError(
                    f"scheduler deadlock: {len(pending)} job(s) pending with an idle cluster"
                )
        elif started or running:
            stalled = 0

        # ── Advance to the next event.
        if next_arrival >= len(arrivals) and not pending and not running:
            break
        candidates = [calc_next]
        if next_arrival < len(arrivals):
            candidates.append(arrivals[next_arrival].submit_time)
        if completions:
            candidates.append(completions[0][0])
        if pending:
            candidates += [sched_next, bf_next]
            if preempt_on:
                candidates += [
                    j.eligible_time
                    for j in pending
                    if j.eligible_time is not None and j.eligible_time > now
                ]
        upcoming = min(candidates)
        if upcoming == inf:
            raise RuntimeError(
                f"scheduler deadlock: {len(pending)} job(s) pending with an idle cluster "
                "and no future events"
            )
        previous_now = now
        now = max(upcoming, now)
        # Timers idle while nothing was pending rejoin their grid.
        if sched_next < now:
            sched_next = _grid_at_or_after(now, origin, params.sched_interval)
        if bf_next < now:
            bf_next = _grid_at_or_after(now, origin, params.bf_interval)

    if node_samples[-1][0] < now:
        node_samples.append((now, *last_state))
    makespan = max((j.end_time or 0.0) for j in runnable) - min(j.submit_time for j in runnable)
    if easy_sched is not None:
        backfilled_ids = easy_sched.backfilled_ids
    else:
        backfilled_ids = slurm.backfilled_ids if slurm is not None else []
    return SimulationResult(
        jobs=runnable,
        makespan=makespan,
        backfilled_ids=backfilled_ids,
        utilization_samples=samples,
        unschedulable=impossible,
        backfill_mode=backfill_mode,
        backfill_stats=slurm.stats if slurm else BackfillStats(),
        fairshare_snapshots=engine.snapshots,
        planned_starts=dict(slurm.first_planned_start) if slurm else {},
        main_passes=slurm.main_passes if slurm else 0,
        sched_params=params if slurm else None,
        preemption=preemption,
        preemptions=records,
        node_names=names,
        node_samples=node_samples,
        horizon=now,
    )
