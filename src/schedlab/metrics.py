"""Queue-health metrics.

Every number here answers a question someone actually asks in a cluster review:
"is the hardware busy", "how long do I wait", "are small jobs starving", and
"is one account eating the machine". The k8s lab's definitions, for S0
comparisons, are in `parity.py`. The quantities both reports print (GPU
utilization, work lost and grace-locked GPU-hours, Definition C) are computed
once, by the helpers below that `parity.compute` also calls, so the text
report and the `--json` keys cannot drift apart.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .fragmentation import structural_fragmentation, structural_fragmentation_any
from .model import Cluster, Job, State
from .simulate import PreemptionRecord, SimulationResult

SECONDS_PER_HOUR = 3600.0


@dataclass
class Metrics:
    job_count: int
    makespan: float
    utilization: float
    mean_wait: float
    median_wait: float
    p95_wait: float
    mean_bounded_slowdown: float
    #: Distinct jobs started by backfill at least once (EASY: past a
    #: reservation; conservative: by a backfill cycle). A requeued job that
    #: backfill restarts is still one job here; see `backfill_starts`.
    backfilled: int
    #: Fraction of requested wall-clock actually used: how far the ends
    #: backfill plans with are from the real ones. The tests show one padded
    #: limit forfeiting a backfill slot an exact one takes; on a whole
    #: workload, exact limits are not uniformly better (README, "Time
    #: limits: what accurate requests buy, measured").
    time_limit_accuracy: float
    per_account_wait: dict[str, float]
    #: Runs started by backfill: what `sdiag` reports as "Total backfilled
    #: jobs", which backfill.c increments on every start
    #: (`slurmctld_diag_stats.backfilled_jobs++`, SchedMD/slurm@9f9da53
    #: L4009). Equal to `backfilled` unless preemption requeued a job that
    #: backfill then started again.
    backfill_starts: int = 0
    backfill_mode: str = "easy"
    #: Conservative mode: backfill cycles run with a non-empty queue, and how
    #: many of them were evaluated rather than replayed (see simulate.py).
    bf_cycles: int = 0
    bf_cycles_evaluated: int = 0
    #: Jobs reaching the placement test per cycle — what bf_max_job_test caps.
    bf_mean_tested: float = 0.0
    bf_max_tested: int = 0
    #: Cycles that stopped at bf_max_job_test with jobs still untested. Non-zero
    #: means some of the queue was never considered for backfill in those cycles.
    bf_cycles_hit_max_job_test: int = 0
    #: Cycles that stopped after starting bf_max_job_start jobs. They too can
    #: leave jobs untested; the report prints the count when it is non-zero.
    bf_cycles_hit_max_job_start: int = 0

    #: GPUs in the cluster; the GPU lines are printed only when non-zero.
    total_gpus: int = 0
    #: Whether rack/switch come from a fleet file. Without one there is a
    #: single rack and switch, where Definition C can only say "was anything
    #: allocated anywhere", so the text report omits those levels.
    topology_declared: bool = False
    #: The k8s lab's utilization: delivered GPU-seconds over fleet GPUs ×
    #: horizon, with the horizon measured from trace t=0 to the last release.
    #: (CPU `utilization` above keeps its 0.1.0 denominator, the makespan from
    #: first submit.)
    gpu_utilization: float = 0.0
    #: Definition C (fragmentation.py) over GPUs, at node, rack and switch.
    frag_c_node: float = 0.0
    frag_c_rack: float = 0.0
    frag_c_switch: float = 0.0
    #: The variant: a node counts as carved if any GPU *or CPU* is allocated.
    frag_c_any_node: float = 0.0
    #: Definition C applied to CPUs instead of GPUs.
    frag_c_cpu_node: float = 0.0
    frag_c_cpu_rack: float = 0.0
    frag_c_cpu_switch: float = 0.0

    #: Preemption (all zero when it is off).
    preemption_enabled: bool = False
    preemptions: int = 0
    requeues: int = 0
    cancels: int = 0
    #: Preempted jobs whose own end came inside the grace period; Slurm still
    #: applied PreemptMode to them.
    exited_in_grace: int = 0
    #: Run time preemption threw away, start to selection: all of it for a
    #: cancelled run, the uncheckpointed share for a requeued one (the k8s
    #: lab's `preempted_gpu_hours_lost`).
    work_lost_cpu_hours: float = 0.0
    work_lost_gpu_hours: float = 0.0
    #: Held from selection to release (the grace period). Disjoint from work
    #: lost, as in the k8s lab's `grace_locked_gpu_hours`.
    grace_locked_cpu_hours: float = 0.0
    grace_locked_gpu_hours: float = 0.0
    #: Thrash: jobs preempted more than once.
    thrashed_jobs: int = 0
    max_preemptions_per_job: int = 0

    def format(self) -> str:
        lines = [
            f"  jobs                    {self.job_count}",
            f"  makespan                {self.makespan / 3600:>8.2f} h",
            f"  cpu utilization         {self.utilization * 100:>8.1f} %",
        ]
        if self.total_gpus:
            lines.append(f"  gpu utilization         {self.gpu_utilization * 100:>8.1f} %")
        lines += [
            f"  mean wait               {self.mean_wait / 60:>8.1f} min",
            f"  median wait             {self.median_wait / 60:>8.1f} min",
            f"  p95 wait                {self.p95_wait / 60:>8.1f} min",
            f"  mean bounded slowdown   {self.mean_bounded_slowdown:>8.2f}",
        ]
        backfilled = f"  backfilled jobs         {self.backfilled}"
        if self.backfill_starts != self.backfilled:
            backfilled += f"  ({self.backfill_starts} backfill starts)"
        lines += [
            backfilled,
            f"  time-limit accuracy     {self.time_limit_accuracy * 100:>8.1f} %",
        ]
        if self.topology_declared:
            if self.total_gpus:
                lines.append(
                    f"  frag C gpu node/rack/sw  {self.frag_c_node * 100:5.1f} /"
                    f" {self.frag_c_rack * 100:5.1f} / {self.frag_c_switch * 100:5.1f} %"
                )
            lines.append(
                f"  frag C cpu node/rack/sw  {self.frag_c_cpu_node * 100:5.1f} /"
                f" {self.frag_c_cpu_rack * 100:5.1f} / {self.frag_c_cpu_switch * 100:5.1f} %"
            )
        else:
            if self.total_gpus:
                lines.append(f"  frag C gpu node         {self.frag_c_node * 100:>8.1f} %")
            lines.append(f"  frag C cpu node         {self.frag_c_cpu_node * 100:>8.1f} %")
        if self.bf_cycles:
            lines += [
                f"  backfill cycles         {self.bf_cycles}"
                f"  ({self.bf_cycles_evaluated} evaluated)",
                f"  jobs tested per cycle   mean {self.bf_mean_tested:.1f},"
                f" max {self.bf_max_tested}",
                f"  cycles at bf_max_job_test  {self.bf_cycles_hit_max_job_test}",
            ]
            if self.bf_cycles_hit_max_job_start:
                # A cycle that stops at bf_max_job_start also leaves jobs
                # untested, so the line above alone would read as full
                # coverage. Printed only when non-zero: the default is 0
                # (no limit), and then it cannot happen.
                lines.append(
                    f"  cycles at bf_max_job_start  {self.bf_cycles_hit_max_job_start}"
                )
        if self.preemption_enabled:
            lines += [
                f"  preemptions             {self.preemptions}"
                f"  ({self.requeues} requeued, {self.cancels} cancelled,"
                f" {self.exited_in_grace} exited in grace)",
                f"  work lost               {self.work_lost_cpu_hours:.1f} CPU-h,"
                f" {self.work_lost_gpu_hours:.1f} GPU-h",
                f"  grace-locked            {self.grace_locked_cpu_hours:.1f} CPU-h,"
                f" {self.grace_locked_gpu_hours:.1f} GPU-h",
                f"  jobs preempted twice+   {self.thrashed_jobs}"
                f"  (max {self.max_preemptions_per_job} for one job)",
            ]
        if self.per_account_wait:
            lines.append("  mean wait by account")
            for account, wait in sorted(
                self.per_account_wait.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"    {account:<18} {wait / 60:>8.1f} min")
        return "\n".join(lines)


def percentile(values: Sequence[float], pct: float) -> float:
    """The rounded-rank percentile: `k8slab.metrics.percentile`. NOT nearest-rank.

    The value of rank `round(pct/100 * n)`, clamped to `[1, n]`, where
    `round` is Python's round-half-to-even. It differs from nearest-rank
    (rank `ceil(pct/100 * n)`, `nearest_rank`) for some n: the p95 of eleven
    values is the 10th here and the 11th there, and the two differ for every
    n from 11 to 19. The k8s lab keeps this rule for `p95_wait` only, so its
    Phase 1 numbers stay bit-identical, and so does this lab (the report's
    p95 wait, and `p95_wait` in `--json`). 0.0 for no values.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


def nearest_rank(values: Sequence[float], pct: float) -> float:
    """True nearest-rank: `k8slab.metrics.nearest_rank`.

    The smallest value with at least `pct`% of the values at or below it,
    i.e. the value of rank `ceil(pct/100 * n)` (at least 1). `pct * n / 100`
    is computed in that order, as there, so an exact rank (7% of 100) is not
    pushed past an integer by the float error of `0.07 * 100`. The k8s lab
    uses it for every percentile added after Phase 1, `wait_by_footprint`
    p95 included. Raises `ValueError` on no values: a percentile of nothing
    is undefined, and callers report None instead.
    """
    if not values:
        raise ValueError("nearest_rank of an empty sequence is undefined")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = sorted(values)
    rank = max(1, math.ceil(pct * len(ordered) / 100.0))
    return ordered[rank - 1]


@dataclass(frozen=True)
class GpuUse:
    """GPU-seconds delivered, and what the fleet could have delivered.

    The k8s lab's utilization: delivered GPU-seconds over fleet GPUs ×
    horizon, the horizon running from trace t=0 to the last release (not
    from the first submit: that is the CPU line's makespan).
    """

    delivered: float
    capacity: float

    @property
    def utilization(self) -> float:
        return self.delivered / self.capacity if self.capacity > 0 else 0.0


def gpu_use(result: SimulationResult, cluster: Cluster) -> GpuUse:
    """The one computation of GPU use; the text report and `parity` both read it."""
    return GpuUse(
        delivered=sum(j.total_gpus * delivered_seconds(j) for j in result.jobs),
        capacity=cluster.total_gpus * result.horizon,
    )


@dataclass(frozen=True)
class PreemptionCost:
    """Work lost and grace-locked time over every preemption, in seconds.

    Disjoint, as in the k8s lab: lost is start to selection less any
    checkpointed share, grace-locked is selection to release.
    """

    lost_cpu_seconds: float
    lost_gpu_seconds: float
    grace_cpu_seconds: float
    grace_gpu_seconds: float


def preemption_cost(records: Sequence[PreemptionRecord]) -> PreemptionCost:
    """The one computation of preemption cost; the text report and `parity` both read it."""
    return PreemptionCost(
        lost_cpu_seconds=sum(r.lost_seconds * r.cpus for r in records),
        lost_gpu_seconds=sum(r.lost_seconds * r.gpus for r in records),
        grace_cpu_seconds=sum(r.grace_seconds * r.cpus for r in records),
        grace_gpu_seconds=sum(r.grace_seconds * r.gpus for r in records),
    )


def delivered_seconds(job: Job) -> float:
    """Seconds of useful work `job` delivered: its runtime if it completed.

    Checkpointed progress from earlier runs is part of that runtime, and
    work preemption threw away is not delivered — the k8s lab's rule ("their
    Running time is lost work, not delivered"). So a run without preemption
    reports exactly the trace's demand, and a cancelled job delivers nothing.
    Post-hoc only: the scheduler never sees this.
    """
    return job.duration if job.state is State.COMPLETED else 0.0


def allocated_seconds(job: Job) -> float:
    """Seconds `job` held its allocation, over every run, grace included.

    Not used for utilization (see `delivered_seconds`); it is delivered work
    plus work lost plus grace-locked time. Post-hoc only.
    """
    total = 0.0
    completed = False
    for run in job.runs:
        if run.outcome == "completed":
            completed = True
        elif run.end is not None:
            total += run.end - run.start
    if completed:
        total += job.duration - job.saved_progress
    return total


def definition_c(
    result: SimulationResult, cluster: Cluster
) -> tuple[dict[str, float], float, dict[str, float]]:
    """(GPU rates by level, any-resource node rate, CPU rates by level)."""
    domains = cluster.domains()
    gpu_cap = {n.name: n.gpus for n in cluster.nodes}
    cpu_cap = {n.name: n.cpus for n in cluster.nodes}
    gpu_samples = result.free_gpu_samples()
    cpu_samples = result.free_cpu_samples()
    gpu = structural_fragmentation(gpu_cap, domains, gpu_samples).rates()
    cpu = structural_fragmentation(cpu_cap, domains, cpu_samples).rates()
    anyres = structural_fragmentation_any(gpu_cap, cpu_cap, {}, gpu_samples, cpu_samples)
    return gpu, anyres.rate("node"), cpu


def compute(result: SimulationResult, cluster: Cluster) -> Metrics:
    jobs: list[Job] = result.jobs
    started = [j for j in jobs if j.first_start_time is not None or j.start_time is not None]
    waits = [j.wait_time for j in started]

    # Utilization as delivered CPU-seconds over the CPU-seconds the cluster
    # could have delivered across the makespan. Work that preemption threw
    # away is not delivered; it is reported as work lost.
    delivered = sum(j.total_cpus * delivered_seconds(j) for j in jobs)
    capacity = cluster.total_cpus * result.makespan
    utilization = delivered / capacity if capacity > 0 else 0.0

    requested = sum(j.time_limit for j in jobs)
    used = sum(j.duration for j in jobs)
    accuracy = used / requested if requested > 0 else 0.0

    stats = result.backfill_stats
    per_account: dict[str, list[float]] = {}
    for job in started:
        per_account.setdefault(job.account, []).append(job.wait_time)

    # A cancelled job's turnaround is not a slowdown; it never finished.
    finished = [j for j in jobs if j.state is State.COMPLETED]

    if result.node_samples:
        gpu_c, any_c, cpu_c = definition_c(result, cluster)
    else:
        gpu_c = cpu_c = {"node": 0.0, "rack": 0.0, "switch": 0.0}
        any_c = 0.0

    records = result.preemptions
    cost = preemption_cost(records)
    hours = SECONDS_PER_HOUR
    # One count per job, not per id: ids need not be unique.
    per_job = [j.preempt_count for j in jobs if j.preempt_count]
    # Backfill starts are per run, so a requeued job backfilled twice is one
    # job with two starts (RunRecord.backfilled; ids need not be unique).
    backfill_runs = [sum(1 for r in j.runs if r.backfilled) for j in jobs]

    return Metrics(
        job_count=len(jobs),
        makespan=result.makespan,
        utilization=utilization,
        mean_wait=statistics.fmean(waits) if waits else 0.0,
        median_wait=statistics.median(waits) if waits else 0.0,
        p95_wait=percentile(waits, 95),
        mean_bounded_slowdown=(
            statistics.fmean([j.bounded_slowdown() for j in finished]) if finished else 0.0
        ),
        backfilled=sum(1 for n in backfill_runs if n),
        backfill_starts=sum(backfill_runs),
        time_limit_accuracy=accuracy,
        per_account_wait={a: statistics.fmean(w) for a, w in per_account.items()},
        backfill_mode=result.backfill_mode,
        bf_cycles=stats.count,
        bf_cycles_evaluated=stats.evaluated,
        bf_mean_tested=stats.mean_tested,
        bf_max_tested=stats.max_tested,
        bf_cycles_hit_max_job_test=stats.hit_max_job_test,
        bf_cycles_hit_max_job_start=stats.hit_max_job_start,
        total_gpus=cluster.total_gpus,
        topology_declared=cluster.topology_declared,
        gpu_utilization=gpu_use(result, cluster).utilization,
        frag_c_node=gpu_c["node"],
        frag_c_rack=gpu_c["rack"],
        frag_c_switch=gpu_c["switch"],
        frag_c_any_node=any_c,
        frag_c_cpu_node=cpu_c["node"],
        frag_c_cpu_rack=cpu_c["rack"],
        frag_c_cpu_switch=cpu_c["switch"],
        preemption_enabled=result.preemption is not None and result.preemption.enabled,
        preemptions=len(records),
        requeues=sum(1 for r in records if r.mode == "REQUEUE"),
        cancels=sum(1 for r in records if r.mode == "CANCEL"),
        exited_in_grace=sum(1 for r in records if r.exited_in_grace),
        work_lost_cpu_hours=cost.lost_cpu_seconds / hours,
        work_lost_gpu_hours=cost.lost_gpu_seconds / hours,
        grace_locked_cpu_hours=cost.grace_cpu_seconds / hours,
        grace_locked_gpu_hours=cost.grace_gpu_seconds / hours,
        thrashed_jobs=sum(1 for n in per_job if n > 1),
        max_preemptions_per_job=max(per_job, default=0),
    )
