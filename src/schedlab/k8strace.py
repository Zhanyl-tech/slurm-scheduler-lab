"""The S0 adapter: a k8s-gpu-scheduler-lab trace CSV as Slurm jobs.

The k8s lab's trace records (`k8slab.trace`, `TRACE_COLUMNS`) have the header

    job_id,account,submit_time,duration,gpus,gang_size,priority

and nothing else. Each field's mapping, and the choices the CSV forces:

* **Times** are read as they are. The CSV holds uncompressed trace seconds:
  `k8slab trace --out` writes the generator's values, and the k8s runner
  divides by `--speedup` only while replaying (its runner.py docstring). No
  rescaling happens here, and none is needed (see README "No time scaling").
* **gang_size -> nodes, gpus -> gpus_per_node.** A gang of G pods of g GPUs
  becomes one Slurm job asking for G *distinct* nodes with g GPUs each
  (`-N G --gpus-per-node g`). Slurm starts it only when all G nodes are
  free at once — it can never be partially placed. Kubernetes may also put
  two pods of one gang on the same node, which `-N G` forbids, so S0 is
  stricter on placement for gangs whose pods would fit together.
* **cpus_per_node** defaults to 1, the `cpu: "1"` every k8s lab pod requests
  (k8slab runner.py). CPUs then never bind before GPUs do on the shipped
  fleets, as in the k8s lab.
* **time_limit** is not in the trace, and backfill cannot run without it. The
  model must be chosen explicitly (`TIME_LIMIT_MODELS`):
    - `exact`: time_limit = duration, rounded up to a whole minute. The
      closest a Slurm user can get: every reservation ends at most a minute
      after the job does. That is *not* a best case for waiting. On five
      default-profile traces it gave a higher mean wait than ×3 padding and
      a lower large-job starvation ratio (README, "Running S0").
    - `padded`: duration × factor (factor >= 1), rounded up likewise.
    - `synthetic`: duration × max(1.05, N(3.0, 1.0)), rounded up likewise:
      the padding distribution of this repo's `trace.WorkloadProfile`, drawn
      from its own seeded generator in file order (the same distribution as
      `trace.generate`, not the same draws).
  Every model rounds up to whole minutes because Slurm holds nothing finer:
  `--time` is parsed by `time_str2mins()`, which rounds seconds up, and
  backfill plans with `time_limit * 60` (`trace.whole_minutes` has the
  citations). Before this rounding the limits were fractional seconds, which
  no Slurm job can carry.
* **priority** maps by `PRIORITY_MAPPINGS`:
    - `qos` (default): each distinct value p becomes a QOS named `p<value>`
      with QOS Priority p. Slurm normalises a QOS priority "to the highest
      priority of all the QOSs to become the QOS factor"
      (https://slurm.schedmd.com/priority_multifactor.html), so
      `qos_factor = p / max(p)`, weighted by `PriorityWeightQOS`. Additive,
      not strict: other factors can outweigh it. With `preempt/qos`, every
      QOS may preempt every QOS with a lower value — the shape of
      Kubernetes' default `preemptionPolicy: PreemptLowerPriority`, which
      "will allow pods of that PriorityClass to preempt lower-priority pods"
      (https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/,
      read 2026-09-26).
    - `tier`: each distinct value becomes a partition `p<value>` whose
      `PriorityTier` is the value's rank (1 = lowest). Partition PriorityTier
      is evaluated before job priority, so this is strict priority order —
      the closer match to kube-scheduler, where "a pending Pod is placed ahead
      of other pending Pods with lower priority in the scheduling queue"
      (https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/,
      read 2026-09-26). With `preempt/partition_prio`, higher tiers may
      preempt lower ones.
    - `none`: ignored.
  Negative priorities are refused (a Slurm QOS priority is unsigned).
* **account** is kept; there is no user column, so the account stands in for
  the user in per-user backfill limits.

`trace_digest()` is the k8s lab's `metrics.trace_digest` over the records as
`k8slab.trace.read_csv` parses them, so an S0 result can prove it replayed
the same trace file a k8s run read.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, get_args

from .model import Job
from .preempt import PartitionSpec, QOSSpec, partition_ladder, qos_ladder
from .trace import WorkloadProfile, check_row_width, whole_minutes

#: Exactly `k8slab.trace.TRACE_COLUMNS`.
TRACE_COLUMNS = ("job_id", "account", "submit_time", "duration", "gpus", "gang_size", "priority")

TimeLimitModel = Literal["exact", "padded", "synthetic"]
TIME_LIMIT_MODELS: tuple[str, ...] = get_args(TimeLimitModel)
PriorityMapping = Literal["qos", "tier", "none"]
PRIORITY_MAPPINGS: tuple[str, ...] = get_args(PriorityMapping)


class HasDuration(Protocol):
    """Anything with a true runtime: a `K8sRecord`, or a synthetic `Job`."""

    @property
    def duration(self) -> float: ...


@dataclass(frozen=True)
class K8sRecord:
    """One k8s trace row, typed as `k8slab.trace.read_csv` types it."""

    job_id: int
    account: str
    submit_time: float
    duration: float
    gpus: int
    gang_size: int
    priority: int


@dataclass
class S0Trace:
    """Slurm jobs plus the QOS / partition tables the priority mapping built."""

    jobs: list[Job]
    digest: str
    time_limit_model: TimeLimitModel
    time_limit_factor: float | None
    priority_mapping: PriorityMapping
    cpus_per_pod: int
    qos: dict[str, QOSSpec] = field(default_factory=dict)
    partitions: dict[str, PartitionSpec] = field(default_factory=dict)

    def describe(self) -> str:
        limit: str = self.time_limit_model
        if self.time_limit_model == "padded":
            limit += f" ×{self.time_limit_factor:g}"
        return (
            f"k8s trace {self.digest} · time limits {limit} · priority→{self.priority_mapping}"
            f" · {self.cpus_per_pod} CPU per pod"
        )


def read_csv(path: str | Path) -> list[K8sRecord]:
    """Parse a k8s lab trace CSV; the header must match exactly, as there."""
    records: list[K8sRecord] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != TRACE_COLUMNS:
            raise ValueError(
                f"{path}: header must be exactly {','.join(TRACE_COLUMNS)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            check_row_width(row, path, reader.line_num)
            records.append(
                K8sRecord(
                    job_id=int(row["job_id"]),
                    account=row["account"],
                    submit_time=float(row["submit_time"]),
                    duration=float(row["duration"]),
                    gpus=int(row["gpus"]),
                    gang_size=int(row["gang_size"]),
                    priority=int(row["priority"]),
                )
            )
    if not records:
        raise ValueError(f"{path}: trace is empty")
    return records


def trace_digest(records: Iterable[K8sRecord]) -> str:
    """`k8slab.metrics.trace_digest`, byte for byte."""
    h = hashlib.sha256()
    for r in records:
        h.update(
            f"{r.job_id},{r.account},{r.submit_time!r},{r.duration!r},"
            f"{r.gpus},{r.gang_size},{r.priority}\n".encode()
        )
    return h.hexdigest()[:12]


def time_limits(
    records: Sequence[HasDuration],
    model: TimeLimitModel,
    factor: float | None = None,
    seed: int = 0,
) -> list[float]:
    """The requested limit for each record under `model` (see module docstring).

    Also used for the CLI's synthetic traces (`exact` and `padded`), so both
    trace sources share one definition and one factor check. Every limit is
    a whole number of minutes, rounded up, as Slurm stores it.
    """
    if model == "exact":
        return [whole_minutes(r.duration) for r in records]
    if model == "padded":
        if factor is None or factor < 1.0:
            raise ValueError("the padded time-limit model needs a factor >= 1")
        return [whole_minutes(r.duration * factor) for r in records]
    if model == "synthetic":
        profile = WorkloadProfile()
        rng = random.Random(seed)
        return [
            whole_minutes(
                r.duration * max(1.05, rng.gauss(profile.request_padding, profile.padding_sigma))
            )
            for r in records
        ]
    raise ValueError(f"time-limit model must be one of {TIME_LIMIT_MODELS}, got {model!r}")


def to_jobs(
    records: Sequence[K8sRecord],
    *,
    time_limit_model: TimeLimitModel,
    time_limit_factor: float | None = None,
    seed: int = 0,
    priority_mapping: PriorityMapping = "qos",
    cpus_per_pod: int = 1,
    grace_time: float = 0.0,
) -> S0Trace:
    """Build Slurm jobs from k8s records. `time_limit_model` has no default."""
    if priority_mapping not in PRIORITY_MAPPINGS:
        raise ValueError(f"priority mapping must be one of {PRIORITY_MAPPINGS}")
    if cpus_per_pod < 1:
        raise ValueError("cpus_per_pod must be >= 1")
    for r in records:
        if r.priority < 0:
            raise ValueError(f"job {r.job_id}: negative priority {r.priority} has no QOS mapping")
        if r.gang_size < 1:
            raise ValueError(f"job {r.job_id}: gang_size must be >= 1")
    limits = time_limits(records, time_limit_model, time_limit_factor, seed)
    levels = sorted({r.priority for r in records})
    top = max(levels) if levels else 0

    qos: dict[str, QOSSpec] = {}
    partitions: dict[str, PartitionSpec] = {}
    if priority_mapping == "qos":
        qos = qos_ladder({f"p{p}": p for p in levels}, grace_time=grace_time)
    elif priority_mapping == "tier":
        partitions = partition_ladder(
            {f"p{p}": rank for rank, p in enumerate(levels, start=1)}, grace_time=grace_time
        )

    jobs: list[Job] = []
    for r, limit in zip(records, limits, strict=True):
        job = Job(
            job_id=r.job_id,
            account=r.account,
            submit_time=r.submit_time,
            duration=r.duration,
            time_limit=limit,
            nodes=r.gang_size,
            cpus_per_node=cpus_per_pod,
            gpus_per_node=r.gpus,
        )
        if priority_mapping == "qos":
            job.qos = f"p{r.priority}"
            job.qos_factor = r.priority / top if top > 0 else 0.0
        elif priority_mapping == "tier":
            job.partition = f"p{r.priority}"
        jobs.append(job)

    return S0Trace(
        jobs=jobs,
        digest=trace_digest(records),
        time_limit_model=time_limit_model,
        time_limit_factor=time_limit_factor if time_limit_model == "padded" else None,
        priority_mapping=priority_mapping,
        cpus_per_pod=cpus_per_pod,
        qos=qos,
        partitions=partitions,
    )


def load(
    path: str | Path,
    *,
    time_limit_model: TimeLimitModel,
    time_limit_factor: float | None = None,
    seed: int = 0,
    priority_mapping: PriorityMapping = "qos",
    cpus_per_pod: int = 1,
    grace_time: float = 0.0,
) -> S0Trace:
    return to_jobs(
        read_csv(path),
        time_limit_model=time_limit_model,
        time_limit_factor=time_limit_factor,
        seed=seed,
        priority_mapping=priority_mapping,
        cpus_per_pod=cpus_per_pod,
        grace_time=grace_time,
    )
