"""Metric parity with k8s-gpu-scheduler-lab, for S0.

Every field of `ParityMetrics` that shares a name with a field of the k8s
lab's `k8slab.metrics.Metrics` (and so of each run in its results.json) is
computed under that lab's definition, from its docs/metrics.md, restated in
Slurm terms where a translation is needed:

* A Slurm job of N nodes is a k8s job of N pods, one pod per node. `pods`
  counts nodes; a job's total GPUs are `nodes × gpus_per_node` (the k8s
  `gpus × gang_size`).
* **wait** is the time the job was pending (`pending_wait`). Without a
  requeue that is submit to the moment every pod of the job is bound, and
  Slurm allocates all of a job's nodes at one instant, so it is submit to
  the start of the job's only run: `Job.wait_time`. Under eviction the k8s
  lab counts pending time only ("Under eviction: pending time only", its
  docs/metrics.md; `_pending_intervals` / `_pending_seconds`): the union of
  [submit, bind_1), [evict_1, bind_2), ..., [evict_k, final bind). Its
  `evict_time` is when the victim's GPUs are locked for the grace period,
  which is Slurm's selection (`RunRecord.preempt_time`). So a requeued
  job's grace period and requeue delay count as waiting, and the time an
  earlier run was running before selection does not. The text report's
  wait (`Job.wait_time`, submit to the last run's start) counts that time
  too, so under preemption the two differ.
* **utilization** is delivered GPU-seconds over fleet GPUs × horizon, the
  horizon running from trace t=0 to the last release. Work preemption threw
  away is not delivered (the k8s lab's rule). The k8s reference model stops on
  its first 5 s tick after everything drains, so for one schedule its horizon
  can be up to one tick longer than this one's.
* **preemptions** counts evicted pod attempts, "every gang member counts": a
  preempted N-node job is N. **preempted_gpu_hours_lost** is
  `g × (1 - c) × (selection - start)` and **grace_locked_gpu_hours**
  `g × (release - selection)`, disjoint, as there. The Slurm-side job count
  is under `slurm` in the JSON.
* **wait_by_footprint**, **large_job_starvation_ratio**,
  **footprint_wait_spearman**, **fairness_ratio** / **service_ratio**, and
  Definition C are ports of the k8s functions (percentile, bucketing and
  ranking rules included). The k8s lab has two percentile rules, and so does
  this port: `p95_wait` keeps Phase 1's rounded rank (`percentile`), and the
  bucket p95 is true nearest-rank (`nearest_rank`), as in its
  docs/metrics.md, "Wait time".
* **placement_tier_share** uses each job's final run, as the k8s lab uses
  each pod's final attempt: the widest of node / rack / switch / cross-switch
  its nodes span, over multi-node jobs. With no multi-node job every share
  is None (JSON null): 0/0 is undefined, and the k8s lab dropped 0.0 there
  because repeat aggregation averaged it as a real zero.

What is **structurally zero** here, and why (reported, not omitted): a Slurm
job's nodes are allocated atomically, all at once or not at all. No gang is
ever partially placed, so no GPU is ever held by a member waiting for the
rest: `gang_stranded_gpu_hours`, `gang_stranded_share`, the assembly delays,
and the deprecated `gang_stalled` / `gang_deadlocked` / `gang_deadlock_rate`
are all 0. That is a structural advantage of the Slurm substrate over K0, not
a scheduling result, and it must be read as one.

What is **not reported**, with the reason, lives in `NOT_REPORTED`:
fragmentation definitions A and B (A needs the k8s pending *pod* queue; B a
per-pod reference request) and the placement penalty (ASSUMED factors that
only mean something where a replay applies them).
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fragmentation import structural_fragmentation
from .metrics import (
    Metrics,
    delivered_seconds,
    gpu_use,
    nearest_rank,
    percentile,
    preemption_cost,
)
from .model import Cluster, Job
from .simulate import SimulationResult

__all__ = [
    "FOOTPRINT_BUCKETS",
    "NOT_REPORTED",
    "STRUCTURAL_ZEROS",
    "TIERS",
    "ParityMetrics",
    "average_ranks",
    "compute",
    "footprint_bucket",
    "nearest_rank",
    "pending_wait",
    "percentile",
    "spearman",
    "to_json",
    "write_json",
]

SECONDS_PER_HOUR = 3600.0
JSON_SCHEMA_VERSION = 1

#: `k8slab.metrics.FOOTPRINT_BUCKETS`: (label, low, high), high None = open.
FOOTPRINT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17+", 17, None),
)

TIERS: tuple[str, ...] = ("node", "rack", "switch", "cross-switch")

STRUCTURAL_ZEROS: dict[str, str] = {
    "gang_stranded_gpu_hours": (
        "0 by construction: Slurm allocates all of a job's nodes at one instant, so no "
        "member ever holds GPUs while waiting for the others"
    ),
    "gang_stranded_share": "0 for the same reason",
    "gang_assembly_p50": "0 when any multi-node job started: assembly is instantaneous",
    "gang_assembly_p95": "0 when any multi-node job started: assembly is instantaneous",
    "gang_stalled": "deprecated in the k8s lab; 0 here by construction",
    "gang_deadlocked": "deprecated in the k8s lab; 0 here by construction",
    "gang_deadlock_rate": "deprecated in the k8s lab; 0 here by construction",
    "startup_overhead_gpu_hours": (
        "0: the model starts a job the instant its nodes are allocated; prolog and "
        "launch latency are not modelled (the k8s lab's default startup delay is 0 too)"
    ),
    "topology_extension_gpu_hours": (
        "0: this lab never stretches runtimes by assumed topology factors"
    ),
}

NOT_REPORTED: dict[str, str] = {
    "fragmentation_rate": (
        "definition A needs the k8s pending pod queue and per-pod requests; Slurm's "
        "queue differs for reasons unrelated to fragmentation (k8s docs/metrics.md)"
    ),
    "fragmentation_of_fleet": "definition A, as above",
    "stranded_gpu_hours": "definition A, as above",
    "fragmentation_ref": "definition B needs a largest per-pod request; a Slurm job has no pods",
    "reference_request": "definition B, as above",
    "placement_penalty_mean": (
        "ASSUMED per-tier penalty factors; meaningful only where a replay applies them"
    ),
    "penalty_factors": "as above",
    "harness": "k8s execution-layer settings; this lab's run settings are under `model`",
    "harness_seed": "as above",
    "notes": "k8s execution-layer notes; this lab's warnings are printed with the run",
}


def footprint_bucket(total_gpus: int) -> str | None:
    for label, low, high in FOOTPRINT_BUCKETS:
        if total_gpus >= low and (high is None or total_gpus <= high):
            return label
    return None


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks, ties given the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """`k8slab.metrics.spearman`: Pearson correlation of tie-averaged ranks."""
    if len(xs) != len(ys):
        raise ValueError("spearman needs paired samples")
    if len(xs) < 2:
        return None
    rx, ry = average_ranks(xs), average_ranks(ys)
    mx, my = math.fsum(rx) / len(rx), math.fsum(ry) / len(ry)
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = math.fsum((a - mx) ** 2 for a in rx)
    syy = math.fsum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


@dataclass
class ParityMetrics:
    """One run, under the k8s lab's field names and definitions."""

    config: str
    measured_on_cluster: bool
    fleet: str
    #: The k8s lab's trace fingerprint; None for traces that are not k8s CSVs.
    trace_digest: str | None
    jobs: int
    pods: int
    jobs_completed: int
    gpu_hours_used: float
    gpu_hours_idle: float
    utilization: float
    makespan_hours: float
    mean_wait: float
    p95_wait: float
    fragmentation_structural: float
    fragmentation_structural_rack: float
    fragmentation_structural_switch: float
    gang_jobs: int
    gang_stalled: int
    gang_deadlocked: int
    gang_deadlock_rate: float
    gang_assembled: int
    gang_stranded_gpu_hours: float
    gang_stranded_share: float
    gang_assembly_p50: float | None
    gang_assembly_p95: float | None
    footprint_wait_spearman: float | None
    large_job_gpus: int
    large_job_starvation_ratio: float | None
    wait_by_footprint: dict[str, dict[str, float | None]]
    topology_declared: bool
    placement_multi_pod_jobs: int
    #: None for every tier when no multi-node job ran (0/0; the k8s lab's rule).
    placement_tier_share: dict[str, float | None]
    fairness_ratio: float
    service_ratio: dict[str, float]
    gpu_hours_demanded: float
    startup_overhead_gpu_hours: float
    preemptions: int
    preempted_gpu_hours_lost: float
    grace_locked_gpu_hours: float
    topology_extension_gpu_hours: float


def _pending_intervals(
    submit: float, earlier: Sequence[tuple[float, float]], bound: float
) -> list[tuple[float, float]]:
    """`k8slab.metrics._pending_intervals` for one pod that was bound.

    `[submit, b_1), [e_1, b_2), ..., [e_k, bound)` for earlier attempts bound
    at `b_i` and evicted-and-requeued at `e_i`; empty intervals dropped.
    """
    out: list[tuple[float, float]] = []
    opened = submit
    for bind, evict in sorted(earlier):
        if bind > opened:
            out.append((opened, bind))
        opened = max(opened, evict)
    if bound > opened:
        out.append((opened, bound))
    return out


def pending_wait(job: Job) -> float:
    """Seconds `job` was pending before its final run: the k8s lab's wait.

    A port of `k8slab.metrics._pending_seconds` (see the module docstring).
    An earlier attempt is a run that ended `requeued`; it was bound at its
    start and evicted at its selection (`RunRecord.preempt_time`, the k8s
    lab's `evict_time`, where the grace lock starts). The final attempt is
    bound at `start_time`. A cancelled run is its own final attempt, as a
    pod the k8s lab evicts without requeueing is.

    Without a requeued run this is exactly `job.wait_time`, so every run
    without preemption keeps the numbers it had. With one, the union of the
    pending intervals is summed the way the k8s lab sums it (merge, then
    `math.fsum`), so the two agree to the bit on the same intervals.
    """
    if job.start_time is None:
        raise ValueError(f"job {job.job_id} never started")
    earlier = [
        (run.start, run.preempt_time)
        for run in job.runs[:-1]
        if run.outcome == "requeued" and run.preempt_time is not None
    ]
    if not earlier:
        return job.wait_time
    pieces: list[float] = []
    lo: float | None = None
    hi = 0.0
    for start, end in sorted(_pending_intervals(job.submit_time, earlier, job.start_time)):
        if lo is None or start > hi:
            if lo is not None:
                pieces.append(hi - lo)
            lo, hi = start, end
        else:
            hi = max(hi, end)
    if lo is not None:
        pieces.append(hi - lo)
    return math.fsum(pieces)


def _tier(nodes: Sequence[int], cluster: Cluster, domains: dict[str, dict[str, str]]) -> str:
    names = {cluster.nodes[n].name for n in nodes}
    if len(names) == 1:
        return "node"
    if len({domains["rack"][n] for n in names}) == 1:
        return "rack"
    if len({domains["switch"][n] for n in names}) == 1:
        return "switch"
    return "cross-switch"


def compute(
    result: SimulationResult,
    cluster: Cluster,
    *,
    config: str = "S0",
    trace_digest: str | None = None,
) -> ParityMetrics:
    """Score one run under the k8s lab's definitions (module docstring)."""
    everyone: list[Job] = [*result.jobs, *result.unschedulable]
    admitted = [j for j in result.jobs if j.start_time is not None]
    # Keyed by the job itself: ids need not be unique (sacct array tasks).
    # Pending time only, as the k8s lab measures it; every wait-derived key
    # below (buckets, starvation, Spearman) reads this one mapping.
    wait_of: dict[Job, float] = {j: pending_wait(j) for j in admitted}
    waits = list(wait_of.values())

    use = gpu_use(result, cluster)  # the same computation as the text report's
    gpu_seconds, capacity = use.delivered, use.capacity
    horizon = result.horizon
    cost = preemption_cost(result.preemptions)
    domains = cluster.domains()

    frag = structural_fragmentation(
        {n.name: n.gpus for n in cluster.nodes}, domains, result.free_gpu_samples()
    ).rates()

    gangs = [j for j in everyone if j.nodes > 1]
    assembled = [j for j in gangs if j.start_time is not None]

    # ---- size bias in admission (k8s metrics.compute, same rules) ----------
    bucket_jobs = {label: 0 for label, _, _ in FOOTPRINT_BUCKETS}
    bucket_waits: dict[str, list[float]] = {label: [] for label, _, _ in FOOTPRINT_BUCKETS}
    footprints: list[float] = []
    admitted_waits: list[float] = []
    large = cluster.max_node_gpus
    large_waits: list[float] = []
    single_waits: list[float] = []
    for job in everyone:
        bucket = footprint_bucket(job.total_gpus)
        if bucket is None:
            continue  # a zero-GPU job does not compete for GPUs
        bucket_jobs[bucket] += 1
        w = wait_of.get(job)
        if w is None:
            continue
        bucket_waits[bucket].append(w)
        footprints.append(float(job.total_gpus))
        admitted_waits.append(w)
        if job.total_gpus >= large:
            large_waits.append(w)
        if job.total_gpus == 1:
            single_waits.append(w)
    wait_by_footprint: dict[str, dict[str, float | None]] = {
        label: {
            "jobs": float(bucket_jobs[label]),
            "admitted": float(len(bucket_waits[label])),
            "mean": statistics.fmean(bucket_waits[label]) if bucket_waits[label] else None,
            # True nearest-rank, as k8slab.metrics.compute: not `percentile`,
            # which the two labs keep for `p95_wait` alone.
            "p95": nearest_rank(bucket_waits[label], 95) if bucket_waits[label] else None,
        }
        for label, _, _ in FOOTPRINT_BUCKETS
    }
    starvation: float | None = None
    if large_waits and single_waits:
        single_mean = statistics.fmean(single_waits)
        if single_mean > 0:
            starvation = statistics.fmean(large_waits) / single_mean

    # ---- placement ---------------------------------------------------------
    counts = dict.fromkeys(TIERS, 0)
    multi = 0
    for job in admitted:
        if job.nodes > 1 and job.runs:
            counts[_tier(job.runs[-1].nodes, cluster, domains)] += 1
            multi += 1

    # ---- fairness ----------------------------------------------------------
    demanded: dict[str, float] = {}
    delivered: dict[str, float] = {}
    for job in everyone:
        demanded[job.account] = demanded.get(job.account, 0.0) + job.total_gpus * job.duration
    for job in result.jobs:
        delivered[job.account] = delivered.get(job.account, 0.0) + (
            job.total_gpus * delivered_seconds(job)
        )
    ratios = {a: (delivered.get(a, 0.0) / d if d > 0 else 0.0) for a, d in demanded.items()}
    positive = [r for r in ratios.values() if r > 0]
    fairness = (max(positive) / min(positive)) if len(positive) > 1 else 1.0

    return ParityMetrics(
        config=config,
        measured_on_cluster=False,
        fleet=cluster.name,
        trace_digest=trace_digest,
        jobs=len(everyone),
        pods=sum(j.nodes for j in everyone),
        jobs_completed=len(admitted),
        gpu_hours_used=gpu_seconds / SECONDS_PER_HOUR,
        gpu_hours_idle=max(0.0, capacity - gpu_seconds) / SECONDS_PER_HOUR,
        utilization=use.utilization,
        makespan_hours=horizon / SECONDS_PER_HOUR,
        mean_wait=statistics.fmean(waits) if waits else 0.0,
        p95_wait=percentile(waits, 95),
        fragmentation_structural=frag["node"],
        fragmentation_structural_rack=frag["rack"],
        fragmentation_structural_switch=frag["switch"],
        gang_jobs=len(gangs),
        gang_stalled=0,
        gang_deadlocked=0,
        gang_deadlock_rate=0.0,
        gang_assembled=len(assembled),
        gang_stranded_gpu_hours=0.0,
        gang_stranded_share=0.0,
        gang_assembly_p50=0.0 if assembled else None,
        gang_assembly_p95=0.0 if assembled else None,
        footprint_wait_spearman=spearman(footprints, admitted_waits),
        large_job_gpus=large,
        large_job_starvation_ratio=starvation,
        wait_by_footprint=wait_by_footprint,
        topology_declared=cluster.topology_declared,
        placement_multi_pod_jobs=multi,
        placement_tier_share={t: (counts[t] / multi if multi else None) for t in TIERS},
        fairness_ratio=fairness,
        service_ratio=ratios,
        gpu_hours_demanded=math.fsum(j.total_gpus * j.duration for j in everyone)
        / SECONDS_PER_HOUR,
        startup_overhead_gpu_hours=0.0,
        preemptions=sum(r.pods for r in result.preemptions),
        preempted_gpu_hours_lost=cost.lost_gpu_seconds / SECONDS_PER_HOUR,
        grace_locked_gpu_hours=cost.grace_gpu_seconds / SECONDS_PER_HOUR,
        topology_extension_gpu_hours=0.0,
    )


def _finite(value: Any) -> Any:
    """JSON has no inf/nan; record them as null, as the k8s lab does."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def to_json(
    parity: ParityMetrics,
    slurm: Metrics,
    *,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The document `--json` writes.

    Top-level keys that are `ParityMetrics` fields match a run in the k8s
    lab's results.json (`configs[].runs[]`). Everything Slurm-specific sits
    under `slurm`; the run's settings under `model`.
    """
    doc: dict[str, Any] = {
        "schema": JSON_SCHEMA_VERSION,
        "source": "slurm-scheduler-lab",
        "src": "model",
        **asdict(parity),
        "structural_zeros": dict(STRUCTURAL_ZEROS),
        "not_reported": dict(NOT_REPORTED),
        "slurm": {
            k: v for k, v in asdict(slurm).items() if k not in {"per_account_wait"}
        }
        | {"per_account_wait_seconds": dict(slurm.per_account_wait)},
        "model": model or {},
    }
    cleaned: dict[str, Any] = _finite(doc)
    return cleaned


def write_json(doc: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf-8")
