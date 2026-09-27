"""Core scheduling types: jobs, partitions, and the cluster's resource state.

Resources are modelled as CPUs and GPUs per node. That is enough to reproduce
the two effects that actually drive Slurm queue behaviour — node-level
fragmentation and GPU scarcity — without simulating NUMA. Nodes may differ in
shape (`fleet.py` builds heterogeneous clusters) and carry a *declared* rack
and switch. Only the metrics read those: node selection stays first-fit by
node id and is never topology-aware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    #: Ended by preemption under `PreemptMode=CANCEL` (Slurm has a job state of
    #: the same name). Terminal. A job preempted under REQUEUE goes back to
    #: PENDING instead.
    PREEMPTED = "PREEMPTED"


@dataclass
class RunRecord:
    """One dispatch of a job, kept for post-hoc metrics only.

    A requeued job has one record per run. Nothing on the scheduling path
    reads these. They exist because `Job.allocated` is cleared at release, and
    placement and preempted work have to be measured afterwards.
    """

    start: float
    nodes: tuple[int, ...]
    end: float | None = None
    #: "completed", "requeued" or "cancelled"; None while running.
    outcome: str | None = None
    #: When this run was selected for preemption, if it was.
    preempt_time: float | None = None
    #: Started by backfill (EASY: past a reservation; conservative: by a
    #: backfill cycle) rather than by a main pass. Per run, so a requeued job
    #: backfilled twice counts as one backfilled job with two backfill starts.
    backfilled: bool = False


# eq=False: a job is an entity, not a value. With value equality,
# `list.remove(job)` compared every field of every earlier job — including
# `duration`, the one field the scheduler must never read — and two distinct
# jobs with identical fields would have compared equal.
@dataclass(eq=False)
class Job:
    job_id: int
    account: str
    submit_time: float
    #: Wall-clock the job actually needs. In a replay this comes from the trace;
    #: the scheduler is never allowed to read it (see `time_limit`).
    duration: float
    #: The user's requested limit. Backfill plans against *this*, not `duration`,
    #: which is why over-requesting hurts everyone — the central lesson the
    #: simulator exists to make measurable.
    time_limit: float
    nodes: int = 1
    cpus_per_node: int = 1
    gpus_per_node: int = 0
    qos_factor: float = 0.0
    partition: str = "normal"
    nice: int = 0
    #: Submitting user. Only the per-user backfill limits (`bf_max_job_user`,
    #: `bf_max_job_user_part`) read it; when a trace has no user column the
    #: account stands in for the user (see `owner`).
    user: str | None = None

    state: State = State.PENDING
    start_time: float | None = None
    #: Set when the job *actually* finishes, never earlier. Setting it at
    #: dispatch (as 0.1.0 did) would park the true runtime on the job where
    #: scheduling code could read it.
    end_time: float | None = None
    allocated: list[int] = field(default_factory=list)
    #: The priority the controller currently holds for this job. Like Slurm's
    #: `job_ptr->priority`, it is computed at submit and then only refreshed on
    #: `PriorityCalcPeriod` ticks, so between ticks it is deliberately stale;
    #: and like it, the simulator stores a whole number of at least 1
    #: (`priority.controller_priority`), so near-equal sums tie.
    priority: float = 0.0
    #: Priority at the moment of dispatch, kept for post-hoc analysis.
    dispatch_priority: float = 0.0
    #: Expected start from the most recent conservative backfill cycle that
    #: gave this job a reservation. None when no cycle did: EASY mode, beyond
    #: `bf_window`, or past `bf_max_job_test`. This is narrower than what
    #: `squeue --start` shows. Slurm also keeps the will-run estimate of a
    #: job beyond `bf_window` ("StartTime set to time after current backfill
    #: window. No reservation created", backfill.c L3700-L3716 at
    #: SchedMD/slurm@9f9da53); the model does not compute one, partly because
    #: the dominance pruning in `SlurmScheduler.backfill_cycle` skips those
    #: jobs without planning them. Only jobs never tested have no estimate in
    #: Slurm either.
    planned_start: float | None = None

    # ── Preemption state (see preempt.py). Untouched when preemption is off.
    #: QOS name, for `PreemptType=preempt/qos`. `qos_factor` above stays the
    #: normalised priority factor; this is the identity the preempt lists name.
    qos: str | None = None
    #: First dispatch. `start_time` is the current (finally, the last) run's
    #: start and is cleared on requeue, as Slurm clears `job_ptr->start_time`.
    first_start_time: float | None = None
    #: When the current run was selected for preemption (`job_ptr->preempt_time`).
    preempt_time: float | None = None
    #: The current run's end once selected: `min(start + time_limit,
    #: preempt_time + GraceTime)`, which is what Slurm resets `job_ptr->end_time`
    #: to (`_job_check_grace_internal()`, src/interfaces/preempt.c).
    preempt_deadline: float | None = None
    #: On requeue Slurm resets `details->begin_time` to `requeue time +
    #: requeue_delay + 1`. The job is not eligible before it, and its age
    #: factor accrues from it (`batch_requeue_fini()`, job_mgr.c).
    eligible_time: float | None = None
    #: On requeue Slurm resets `details->submit_time` to the requeue time. Only
    #: the queue order's submit-time tiebreak reads it; metrics keep
    #: `submit_time`, the trace's submission.
    requeue_submit_time: float | None = None
    requeue_count: int = 0
    preempt_count: int = 0
    runs: list[RunRecord] = field(default_factory=list)
    #: World state, not controller state: seconds of work kept by checkpoints
    #: across requeues (`PreemptionConfig.checkpoint_fraction`). Only the
    #: simulator's `_world_end_time()` and post-hoc metrics read it.
    saved_progress: float = 0.0

    @property
    def owner(self) -> str:
        """User for per-user limits: the user if known, else the account."""
        return self.user if self.user is not None else self.account

    @property
    def total_cpus(self) -> int:
        return self.nodes * self.cpus_per_node

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def sched_submit_time(self) -> float:
        """The submit time the controller sorts by (reset on requeue)."""
        if self.requeue_submit_time is not None:
            return self.requeue_submit_time
        return self.submit_time

    @property
    def accrue_start(self) -> float:
        """When the age factor starts accruing: submit, or the post-requeue begin time."""
        return self.eligible_time if self.eligible_time is not None else self.submit_time

    def eligible(self, now: float) -> bool:
        """False while a requeued job waits out its new begin time."""
        return self.eligible_time is None or self.eligible_time <= now

    @property
    def expected_end(self) -> float:
        """Slurm's `job_ptr->end_time` for the current run.

        `start + time_limit`, or the preemption deadline once the job has been
        selected for preemption. Controller knowledge, never the true end.
        """
        if self.start_time is None:
            raise ValueError(f"job {self.job_id} is not running")
        end = self.start_time + self.time_limit
        if self.preempt_deadline is not None and self.preempt_deadline < end:
            return self.preempt_deadline
        return end

    @property
    def wait_time(self) -> float:
        """Submit to the start of the last run: the text report's wait.

        For a requeued job that includes its earlier runs, their grace
        periods and the requeue delay. That is **not** the k8s lab's rule
        under eviction: it counts only the time a job was pending, so the
        time an earlier run was running before its selection is not wait
        there. `parity.pending_wait` ports that rule, and `--json` uses it.
        Without a requeue the two are the same number.
        """
        if self.start_time is None:
            raise ValueError(f"job {self.job_id} never started")
        return self.start_time - self.submit_time

    @property
    def turnaround(self) -> float:
        if self.end_time is None:
            raise ValueError(f"job {self.job_id} never finished")
        return self.end_time - self.submit_time

    def bounded_slowdown(self, threshold: float = 60.0) -> float:
        """Turnaround relative to runtime, floored so short jobs don't dominate.

        The standard metric from the parallel-workload literature: without the
        threshold, a 2-second job waiting 20 seconds reports a slowdown of 11
        and swamps the mean.
        """
        return self.turnaround / max(self.duration, threshold)


@dataclass
class Node:
    node_id: int
    cpus: int
    gpus: int = 0
    free_cpus: int = 0
    free_gpus: int = 0
    #: Stable name: `node-<id>`, unless a fleet file names it `<class>-<i>`.
    name: str = ""
    node_class: str = ""
    #: Declared topology (fleet.py). None means undeclared, and the metrics
    #: then use the k8s lab's flat default: one rack per node class, one switch.
    rack: str | None = None
    switch: str | None = None
    nvlink: bool = False

    def __post_init__(self) -> None:
        self.free_cpus = self.cpus
        self.free_gpus = self.gpus
        if not self.name:
            self.name = f"node-{self.node_id}"

    def fits(self, cpus: int, gpus: int) -> bool:
        return self.free_cpus >= cpus and self.free_gpus >= gpus

    def allocate(self, cpus: int, gpus: int) -> None:
        if not self.fits(cpus, gpus):
            raise ValueError(f"node {self.node_id} cannot fit {cpus}c/{gpus}g")
        self.free_cpus -= cpus
        self.free_gpus -= gpus

    def release(self, cpus: int, gpus: int) -> None:
        self.free_cpus = min(self.cpus, self.free_cpus + cpus)
        self.free_gpus = min(self.gpus, self.free_gpus + gpus)


@dataclass
class Cluster:
    nodes: list[Node]
    name: str = "cluster"
    #: True when a fleet file declared any topology key (fleet.py). False means
    #: every rack and switch is the flat default.
    topology_declared: bool = False

    @classmethod
    def homogeneous(cls, count: int, cpus: int, gpus: int = 0) -> Cluster:
        return cls([Node(i, cpus, gpus) for i in range(count)], name="homogeneous")

    @property
    def total_cpus(self) -> int:
        return sum(n.cpus for n in self.nodes)

    @property
    def total_gpus(self) -> int:
        return sum(n.gpus for n in self.nodes)

    @property
    def busy_cpus(self) -> int:
        return sum(n.cpus - n.free_cpus for n in self.nodes)

    @property
    def max_node_gpus(self) -> int:
        """Largest single-node GPU capacity (the k8s lab's `large_job_gpus`)."""
        return max((n.gpus for n in self.nodes), default=0)

    def domains(self) -> dict[str, dict[str, str]]:
        """`{level: {node name: domain}}` for rack and switch, finest first.

        Undeclared racks default to one per node class, and undeclared
        switches to a single switch. That is the flat layout
        `k8slab.topology.derive()` uses when a fleet declares nothing, so the
        two labs default identically.
        """
        classes: dict[str, int] = {}
        racks: dict[str, str] = {}
        switches: dict[str, str] = {}
        for n in self.nodes:
            index = classes.setdefault(n.node_class, len(classes))
            racks[n.name] = n.rack if n.rack is not None else f"rack-{index}"
            switches[n.name] = n.switch if n.switch is not None else "switch-0"
        return {"rack": racks, "switch": switches}

    def find_nodes(self, job: Job) -> list[int] | None:
        """First-fit node selection. Returns node ids, or None if it won't fit.

        Slurm's real selection is topology-aware; first-fit over a sorted node
        list reproduces the fragmentation behaviour that matters here while
        staying deterministic.
        """
        chosen: list[int] = []
        for node in self.nodes:
            if node.fits(job.cpus_per_node, job.gpus_per_node):
                chosen.append(node.node_id)
                if len(chosen) == job.nodes:
                    return chosen
        return None

    def allocate(self, job: Job, node_ids: list[int]) -> None:
        for nid in node_ids:
            self.nodes[nid].allocate(job.cpus_per_node, job.gpus_per_node)
        job.allocated = list(node_ids)

    def release(self, job: Job) -> None:
        for nid in job.allocated:
            self.nodes[nid].release(job.cpus_per_node, job.gpus_per_node)
        job.allocated = []
