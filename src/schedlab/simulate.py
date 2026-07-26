"""Event-driven scheduling simulator with EASY backfill.

The scheduler is only ever allowed to read `job.time_limit`, never
`job.duration`. That asymmetry is deliberate and is the point of the whole
exercise: backfill plans against what users *request*, so systematically
over-requesting wall-clock degrades everyone's queue even when the cluster is
half idle. Letting the simulator peek at real runtimes would hide that.

Backfill reservation follows the EASY algorithm (Lifka, 1995):

  1. Sort pending jobs by priority.
  2. Walk the queue; start whatever fits.
  3. At the first job that does not fit, compute its *shadow time* — the
     earliest moment enough resources free up, assuming every running job runs
     to its full time limit — and reserve for it.
  4. Any lower-priority job may still start, provided it either finishes before
     the shadow time or fits in resources the reservation does not need.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .model import Cluster, Job, State
from .priority import FairshareTree, PriorityWeights, compute_priority


@dataclass
class Reservation:
    """The EASY reservation held for the highest-priority blocked job."""

    job_id: int
    shadow_time: float
    cpus: int
    gpus: int


@dataclass
class SimulationResult:
    #: Jobs that ran. Excludes anything in `unschedulable`.
    jobs: list[Job]
    makespan: float
    backfilled_ids: list[int] = field(default_factory=list)
    #: (time, busy_cpus) sampled at each scheduling pass.
    utilization_samples: list[tuple[float, int]] = field(default_factory=list)
    #: Jobs whose request exceeds the cluster outright. Slurm rejects these at
    #: submit time; surfacing them is more useful than failing the whole run,
    #: because on a real trace they usually mean the trace and the cluster
    #: definition disagree.
    unschedulable: list[Job] = field(default_factory=list)


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


class Scheduler:
    def __init__(
        self,
        cluster: Cluster,
        weights: PriorityWeights | None = None,
        fairshare: FairshareTree | None = None,
        partitions: dict[str, float] | None = None,
        backfill: bool = True,
    ) -> None:
        self.cluster = cluster
        self.weights = weights or PriorityWeights()
        self.fairshare = fairshare
        self.partitions = partitions or {}
        self.backfill = backfill
        self.backfilled_ids: list[int] = []

    # ── Reservation planning ────────────────────────────────────────────────

    def _shadow_time(self, blocked: Job, running: list[Job], now: float) -> float:
        """Earliest time `blocked` can start if every running job uses its full limit.

        Aggregate CPU/GPU accounting rather than per-node. Slurm's real
        backfill is topology-aware; the aggregate form gives the same ordering
        behaviour and keeps the reservation cheap to compute.
        """
        free_cpus = sum(n.free_cpus for n in self.cluster.nodes)
        free_gpus = sum(n.free_gpus for n in self.cluster.nodes)

        need_cpus, need_gpus = blocked.total_cpus, blocked.total_gpus
        if free_cpus >= need_cpus and free_gpus >= need_gpus:
            return now

        # Release running jobs in projected-completion order until it fits.
        projected = sorted(
            running, key=lambda j: (j.start_time or now) + j.time_limit
        )
        for job in projected:
            free_cpus += job.total_cpus
            free_gpus += job.total_gpus
            if free_cpus >= need_cpus and free_gpus >= need_gpus:
                return (job.start_time or now) + job.time_limit

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
        """Start whatever can start. Returns the jobs dispatched this pass."""
        if self.fairshare:
            self.fairshare.decay(now)

        for job in pending:
            job.dispatch_priority = compute_priority(
                job, now, self.cluster, self.weights, self.fairshare, self.partitions
            )

        # Ties broken by submit order, then id, so runs are reproducible.
        queue = sorted(
            pending, key=lambda j: (-j.dispatch_priority, j.submit_time, j.job_id)
        )

        started: list[Job] = []
        reservation: Reservation | None = None

        for job in queue:
            node_ids = self.cluster.find_nodes(job)

            if node_ids is not None:
                if reservation is not None and not self._may_backfill(job, reservation, now):
                    continue
                if reservation is not None:
                    self.backfilled_ids.append(job.job_id)

                self.cluster.allocate(job, node_ids)
                job.state = State.RUNNING
                job.start_time = now
                job.end_time = now + job.duration
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
                reservation = Reservation(
                    job.job_id, shadow, job.total_cpus, job.total_gpus
                )

        return started


def simulate(
    jobs: list[Job],
    cluster: Cluster,
    weights: PriorityWeights | None = None,
    fairshare: FairshareTree | None = None,
    partitions: dict[str, float] | None = None,
    backfill: bool = True,
) -> SimulationResult:
    """Run `jobs` through `cluster` to completion.

    Jobs larger than the cluster are set aside rather than raising, and are
    reported on the result.
    """
    scheduler = Scheduler(cluster, weights, fairshare, partitions, backfill)

    jobs, impossible = partition_schedulable(jobs, cluster)
    if not jobs:
        return SimulationResult(jobs=[], makespan=0.0, unschedulable=impossible)

    by_submit = sorted(jobs, key=lambda j: (j.submit_time, j.job_id))
    arrivals = list(by_submit)
    pending: list[Job] = []
    running: list[Job] = []
    completions: list[tuple[float, int, Job]] = []  # heap of (end, id, job)
    samples: list[tuple[float, int]] = []

    now = arrivals[0].submit_time if arrivals else 0.0
    next_arrival = 0

    while next_arrival < len(arrivals) or pending or running:
        # Admit everything that has arrived by `now`.
        while next_arrival < len(arrivals) and arrivals[next_arrival].submit_time <= now:
            pending.append(arrivals[next_arrival])
            next_arrival += 1

        for job in scheduler.schedule(pending, running, now):
            pending.remove(job)
            heapq.heappush(completions, (job.end_time, job.job_id, job))
            if fairshare:
                fairshare.charge(job.account, job.total_cpus * job.duration)

        samples.append((now, cluster.busy_cpus))

        # Advance to whichever comes first: the next arrival or the next finish.
        next_times = []
        if next_arrival < len(arrivals):
            next_times.append(arrivals[next_arrival].submit_time)
        if completions:
            next_times.append(completions[0][0])

        if not next_times:
            # Nothing running and nothing arriving. Anything still pending is
            # deadlocked against the current allocation, which would be a bug
            # in the scheduler rather than in the workload.
            if pending:
                raise RuntimeError(
                    f"scheduler deadlock: {len(pending)} job(s) pending with an "
                    "idle cluster and no future arrivals"
                )
            break

        now = max(min(next_times), now)

        # Retire everything that finished at or before the new `now`.
        while completions and completions[0][0] <= now:
            _, _, job = heapq.heappop(completions)
            cluster.release(job)
            job.state = State.COMPLETED
            running.remove(job)

    makespan = max((j.end_time or 0.0) for j in jobs) - min(j.submit_time for j in jobs)
    return SimulationResult(
        jobs=jobs,
        makespan=makespan,
        backfilled_ids=scheduler.backfilled_ids,
        utilization_samples=samples,
        unschedulable=impossible,
    )
