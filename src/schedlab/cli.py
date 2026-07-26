"""Command line entry point."""

from __future__ import annotations

import argparse
import copy
import sys

from .metrics import Metrics, compute
from .model import Cluster
from .priority import FairshareTree, PriorityWeights
from .simulate import simulate
from .trace import WorkloadProfile, from_sacct, generate


def _load_jobs(args: argparse.Namespace):
    if args.sacct:
        return from_sacct(args.sacct)
    return generate(WorkloadProfile(job_count=args.jobs), seed=args.seed)


def _build(args: argparse.Namespace):
    return Cluster.homogeneous(args.nodes, args.cpus, args.gpus)


def _weights(args: argparse.Namespace) -> PriorityWeights:
    if args.slurm_conf:
        return PriorityWeights.from_slurm_conf(args.slurm_conf)
    return PriorityWeights()


def _fairshare(jobs, half_life: float) -> FairshareTree:
    accounts = {job.account for job in jobs}
    return FairshareTree(
        shares={a: 1.0 for a in accounts}, half_life=half_life
    )


def _run(jobs, args, weights, backfill: bool) -> Metrics:
    # Each run needs its own job objects and cluster — simulation mutates both.
    run_jobs = copy.deepcopy(jobs)
    cluster = _build(args)
    result = simulate(
        run_jobs,
        cluster,
        weights=weights,
        fairshare=_fairshare(run_jobs, weights.decay_half_life),
        backfill=backfill,
    )
    return compute(result, cluster)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="schedlab",
        description="Simulate Slurm priority and backfill policy against a job trace.",
    )
    parser.add_argument("--nodes", type=int, default=16, help="cluster node count")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per node")
    parser.add_argument("--gpus", type=int, default=2, help="GPUs per node")
    parser.add_argument("--jobs", type=int, default=400, help="synthetic job count")
    parser.add_argument("--seed", type=int, default=0, help="trace seed")
    parser.add_argument("--sacct", help="replay a real trace from sacct output")
    parser.add_argument("--slurm-conf", help="read PriorityWeight* from a slurm.conf")
    parser.add_argument(
        "--compare-backfill",
        action="store_true",
        help="run with and without backfill and show the delta",
    )
    parser.add_argument(
        "--sweep",
        metavar="FACTOR",
        choices=["age", "fairshare", "jobsize", "qos"],
        help="sweep one PriorityWeight across several magnitudes",
    )
    args = parser.parse_args(argv)

    try:
        jobs = _load_jobs(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    weights = _weights(args)
    source = args.sacct or f"synthetic (seed={args.seed})"
    print(f"\n{len(jobs)} jobs · {args.nodes} nodes × {args.cpus} CPU / {args.gpus} GPU")
    print(f"trace: {source}\n")

    if args.sweep:
        print(f"Sweeping PriorityWeight{args.sweep.capitalize()}\n")
        for value in (0, 1_000, 10_000, 100_000):
            swept = copy.deepcopy(weights)
            setattr(swept, args.sweep, float(value))
            metrics = _run(jobs, args, swept, backfill=True)
            print(f"  weight={value:<7}  "
                  f"util {metrics.utilization * 100:5.1f}%  "
                  f"mean wait {metrics.mean_wait / 60:7.1f} min  "
                  f"p95 {metrics.p95_wait / 60:7.1f} min  "
                  f"slowdown {metrics.mean_bounded_slowdown:5.2f}")
        print()
        return 0

    if args.compare_backfill:
        for label, enabled in (("backfill OFF", False), ("backfill ON", True)):
            print(f"{label}")
            print(_run(jobs, args, weights, backfill=enabled).format())
            print()
        return 0

    print(_run(jobs, args, weights, backfill=True).format())
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
