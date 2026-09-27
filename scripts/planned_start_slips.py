"""Regenerate the README's planned-start slips and backfill-restart counts.

Neither is in the text report or in `--json` (`planned_starts` lives on
`SimulationResult` only), so the README's figures come from here. Every run
goes through the CLI's own configuration path (`cli._config`,
`cli._load_workload`, `cli._attach_tables`, `cli._simulate`), exactly as
`schedlab` with the same flags would run it.

* **Slips** (README, "Scope and simplifications"): over `--jobs 300`, seeds
  0-9, how many jobs that a backfill cycle gave a reservation ("planned")
  started later than their *first* planned start, by how much, and the
  worst. Two configurations: the CLI defaults (demo weights, Fair Tree,
  default SchedulerParameters), and FIFO priorities (every PriorityWeight 0,
  which the CLI has no flag for) with `--time-limit-model exact`, keeping the
  default window and resolution.
* **Backfill restarts** (README, "Preemption"): for `--jobs 300 --seed 5
  --preempt-type qos --preempt-mode REQUEUE --grace-time 300`, the jobs
  backfill started more than once, the extra starts, and whether every one
  of them was requeued.

    python scripts/planned_start_slips.py
"""

from __future__ import annotations

from dataclasses import replace

from schedlab import cli
from schedlab.model import Job
from schedlab.priority import PriorityWeights
from schedlab.simulate import SimulationResult

SEEDS = range(10)


def _run(argv: list[str], fifo: bool = False) -> tuple[SimulationResult, list[Job]]:
    args = cli._parser().parse_args(argv)
    cfg = cli._config(args)
    work = cli._load_workload(args, args.grace_time or 0.0)
    cli._attach_tables(cfg, work, args)
    make_cluster, _ = cli._cluster_factory(args)
    weights: PriorityWeights = cfg.priority
    if fifo:
        weights = replace(weights, age=0.0, fairshare=0.0, jobsize=0.0, partition=0.0, qos=0.0)
    result, _ = cli._simulate(
        work.jobs, args, cfg, weights, cfg.backfill_enabled, make_cluster, work.calendar_offset
    )
    return result, result.jobs


def slips(extra: list[str], fifo: bool) -> tuple[int, int, int, int, float]:
    """(planned, late, late by > 60 s, late by > 1 h, worst minutes) over SEEDS."""
    planned = late = over_minute = over_hour = 0
    worst = 0.0
    for seed in SEEDS:
        result, jobs = _run(["--jobs", "300", "--seed", str(seed), *extra], fifo=fifo)
        # Synthetic ids are unique, so the id-keyed `planned_starts` is exact.
        for job in jobs:
            first = result.planned_starts.get(job.job_id)
            if first is None or job.start_time is None:
                continue
            planned += 1
            slip = job.start_time - first
            if slip > 0:
                late += 1
                over_minute += slip > 60
                over_hour += slip > 3600
                worst = max(worst, slip)
    return planned, late, over_minute, over_hour, worst / 60


def backfill_restarts() -> tuple[int, int, int, int, bool]:
    """(backfilled jobs, backfill starts, jobs started by backfill 2+ times,
    extra starts, all of those requeued) for the README's preemption run."""
    argv = ["--jobs", "300", "--seed", "5", "--preempt-type", "qos", "--preempt-mode",
            "REQUEUE", "--grace-time", "300"]  # fmt: skip
    _, jobs = _run(argv)
    per_job = [(job, sum(1 for r in job.runs if r.backfilled)) for job in jobs]
    again = [(job, n) for job, n in per_job if n > 1]
    return (
        sum(1 for _, n in per_job if n),
        sum(n for _, n in per_job),
        len(again),
        sum(n - 1 for _, n in again),
        all(job.requeue_count for job, _ in again),
    )


def main() -> None:
    for label, extra, fifo in (
        ("CLI defaults", [], False),
        ("FIFO priorities, --time-limit-model exact", ["--time-limit-model", "exact"], True),
    ):
        planned, late, minute, hour, worst = slips(extra, fifo)
        print(
            f"{label}: {late:,} of {planned:,} planned jobs started after their first "
            f"planned start; {minute:,} by more than 60 s, {hour:,} by more than an hour; "
            f"worst {worst:,.1f} min"
        )
    jobs, starts, again, extra_starts, all_requeued = backfill_restarts()
    print(
        f"preemption run: {jobs} backfilled jobs, {starts} backfill starts; {again} jobs "
        f"started by backfill more than once, {extra_starts} extra starts; all requeued: "
        f"{all_requeued}"
    )


if __name__ == "__main__":
    main()
