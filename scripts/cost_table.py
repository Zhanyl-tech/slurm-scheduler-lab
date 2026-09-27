"""Regenerate the README's cost table: what a run costs as the queue deepens.

Every row goes through the CLI's own configuration path (`cli._config`,
`cli._simulate`): the lab's demo weights, Fair Tree fairshare over the
trace's accounts, Slurm's default SchedulerParameters, and a fresh
`Cluster.homogeneous(nodes, 8, 2)`. Only the synthetic trace differs from
`schedlab --jobs N --nodes M --seed 5`: its mean arrival gap is set per row,
which the CLI has no flag for. Run time is wall-clock seconds for the
simulation and the metrics, one run each, on whatever machine runs this.

    python scripts/cost_table.py            # conservative rows, then EASY
"""

from __future__ import annotations

import time

from schedlab import cli
from schedlab.metrics import compute
from schedlab.trace import WorkloadProfile, generate

#: (jobs, nodes, mean arrival gap in seconds): the README's rows.
ROWS = (
    (300, 16, 90.0),
    (1_200, 48, 30.0),
    (2_000, 64, 22.5),
    (4_000, 128, 11.25),
    (3_000, 128, 90.0),
)
#: The rows the README also times in `--backfill-mode easy`.
EASY_ROWS = ((1_200, 48, 30.0), (4_000, 128, 11.25))


def run(jobs: int, nodes: int, gap: float, mode: str) -> tuple[float, float]:
    """(mean jobs tested per backfill cycle, seconds) for one row."""
    argv = ["--jobs", str(jobs), "--nodes", str(nodes), "--seed", "5", "--backfill-mode", mode]
    args = cli._parser().parse_args(argv)
    cfg = cli._config(args)
    make_cluster, _ = cli._cluster_factory(args)
    work = generate(WorkloadProfile(job_count=jobs, arrival_interval=gap), seed=5)
    began = time.perf_counter()
    result, cluster = cli._simulate(
        work, args, cfg, cfg.priority, cfg.backfill_enabled, make_cluster
    )
    metrics = compute(result, cluster)
    return metrics.bf_mean_tested, time.perf_counter() - began


def main() -> None:
    print("| jobs | nodes | mean arrival gap | jobs tested per cycle (mean) | run time |")
    print("| --- | --- | --- | --- | --- |")
    for jobs, nodes, gap in ROWS:
        tested, seconds = run(jobs, nodes, gap, "conservative")
        print(f"| {jobs:,} | {nodes} | {gap:g} s | {tested:.1f} | {seconds:.1f} s |")
    for jobs, nodes, gap in EASY_ROWS:
        _, seconds = run(jobs, nodes, gap, "easy")
        print(f"easy, {jobs:,} jobs: {seconds:.1f} s")


if __name__ == "__main__":
    main()
