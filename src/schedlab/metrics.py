"""Queue-health metrics.

Every number here answers a question someone actually asks in a cluster review:
"is the hardware busy", "how long do I wait", "are small jobs starving", and
"is one account eating the machine".
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .model import Cluster, Job
from .simulate import SimulationResult


@dataclass
class Metrics:
    job_count: int
    makespan: float
    utilization: float
    mean_wait: float
    median_wait: float
    p95_wait: float
    mean_bounded_slowdown: float
    backfilled: int
    #: Fraction of requested wall-clock actually used. Low values are the
    #: single biggest cause of bad backfill decisions.
    time_limit_accuracy: float
    per_account_wait: dict[str, float]

    def format(self) -> str:
        lines = [
            f"  jobs                    {self.job_count}",
            f"  makespan                {self.makespan / 3600:>8.2f} h",
            f"  cpu utilization         {self.utilization * 100:>8.1f} %",
            f"  mean wait               {self.mean_wait / 60:>8.1f} min",
            f"  median wait             {self.median_wait / 60:>8.1f} min",
            f"  p95 wait                {self.p95_wait / 60:>8.1f} min",
            f"  mean bounded slowdown   {self.mean_bounded_slowdown:>8.2f}",
            f"  backfilled jobs         {self.backfilled}",
            f"  time-limit accuracy     {self.time_limit_accuracy * 100:>8.1f} %",
        ]
        if self.per_account_wait:
            lines.append("  mean wait by account")
            for account, wait in sorted(
                self.per_account_wait.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"    {account:<18} {wait / 60:>8.1f} min")
        return "\n".join(lines)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Avoids a numpy dependency for one number."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered))) - 1))
    return ordered[index]


def compute(result: SimulationResult, cluster: Cluster) -> Metrics:
    jobs: list[Job] = result.jobs
    waits = [j.wait_time for j in jobs]

    # Utilization as delivered CPU-seconds over the CPU-seconds the cluster
    # could have delivered across the makespan.
    delivered = sum(j.total_cpus * j.duration for j in jobs)
    capacity = cluster.total_cpus * result.makespan
    utilization = delivered / capacity if capacity > 0 else 0.0

    requested = sum(j.time_limit for j in jobs)
    used = sum(j.duration for j in jobs)
    accuracy = used / requested if requested > 0 else 0.0

    per_account: dict[str, list[float]] = {}
    for job in jobs:
        per_account.setdefault(job.account, []).append(job.wait_time)

    return Metrics(
        job_count=len(jobs),
        makespan=result.makespan,
        utilization=utilization,
        mean_wait=statistics.fmean(waits) if waits else 0.0,
        median_wait=statistics.median(waits) if waits else 0.0,
        p95_wait=_percentile(waits, 95),
        mean_bounded_slowdown=(
            statistics.fmean([j.bounded_slowdown() for j in jobs]) if jobs else 0.0
        ),
        backfilled=len(result.backfilled_ids),
        time_limit_accuracy=accuracy,
        per_account_wait={a: statistics.fmean(w) for a, w in per_account.items()},
    )
