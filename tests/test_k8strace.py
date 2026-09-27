"""B3: the S0 adapter from a k8s lab trace CSV."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from schedlab import k8strace
from schedlab.model import Cluster
from schedlab.preempt import PreemptionConfig, PreemptMode
from schedlab.simulate import simulate
from schedlab.trace import whole_minutes

HEADER = "job_id,account,submit_time,duration,gpus,gang_size,priority\n"
ROWS = (
    "1,research,83.727,2742.097,8,1,0\n"
    "2,research,238.338,73.899,1,1,0\n"
    "3,trading,281.686,3641.422,2,4,100\n"
    "4,infra,301.335,382.871,4,2,500\n"
)
#: A row whose floats `{x:g}` would shorten ("1.23457e+06"), so a digest that
#: formatted them that way instead of with repr() cannot match.
LONG_FLOAT_ROW = "5,infra,1234567.891011,98765.4321098765,1,1,0\n"
#: `k8slab.metrics.trace_digest(k8slab.trace.read_csv(...))` of HEADER + ROWS
#: + LONG_FLOAT_ROW, computed once with k8s-gpu-scheduler-lab's own code
#: (its working tree on 2026-09-26). Pinned so the rule is checked in CI,
#: where the k8s lab is not installed.
K8S_DIGEST = "d1d0fff4892e"


def _csv(tmp_path: Path, body: str = ROWS, header: str = HEADER) -> Path:
    path = tmp_path / "trace.csv"
    path.write_text(header + body)
    return path


def test_rows_map_onto_slurm_jobs(tmp_path):
    s0 = k8strace.load(_csv(tmp_path), time_limit_model="exact")
    j = {x.job_id: x for x in s0.jobs}
    # Times are uncompressed trace seconds, read as they are.
    assert j[1].submit_time == 83.727 and j[1].duration == 2742.097
    # gang_size -> nodes, gpus -> per node, one CPU per pod.
    assert (j[3].nodes, j[3].gpus_per_node, j[3].cpus_per_node, j[3].total_gpus) == (4, 2, 1, 8)
    assert j[4].account == "infra" and j[4].user is None
    # exact: the true runtime, rounded up to the whole minute Slurm stores.
    assert [x.time_limit for x in s0.jobs] == [2760.0, 120.0, 3660.0, 420.0]
    assert all(x.duration <= x.time_limit < x.duration + 60 for x in s0.jobs)


def test_header_must_match_exactly(tmp_path):
    with pytest.raises(ValueError, match="header"):
        k8strace.read_csv(_csv(tmp_path, header="job_id,account,submit,duration,gpus\n"))
    with pytest.raises(ValueError, match="empty"):
        k8strace.read_csv(_csv(tmp_path, body=""))


def test_time_limit_models(tmp_path):
    records = k8strace.read_csv(_csv(tmp_path))
    # 2742.097 × 2.5 = 6855.2 s -> 115 min; 73.899 × 2.5 = 184.7 s -> 4 min.
    assert k8strace.time_limits(records, "padded", 2.5) == [6900.0, 240.0, 9120.0, 960.0]
    with pytest.raises(ValueError, match="factor"):
        k8strace.time_limits(records, "padded")
    with pytest.raises(ValueError, match="factor"):
        k8strace.time_limits(records, "padded", 0.5)
    synthetic = k8strace.time_limits(records, "synthetic", seed=3)
    rng = random.Random(3)
    expected = [whole_minutes(r.duration * max(1.05, rng.gauss(3.0, 1.0))) for r in records]
    assert synthetic == expected
    assert all(limit >= 1.05 * r.duration for limit, r in zip(synthetic, records, strict=True))
    assert k8strace.time_limits(records, "synthetic", seed=3) == synthetic  # seeded
    with pytest.raises(ValueError, match="time-limit model"):
        k8strace.time_limits(records, "guess")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("model", "factor"), [("exact", None), ("padded", 1.7), ("synthetic", None)]
)
def test_every_time_limit_model_gives_whole_minutes_slurm_can_hold(tmp_path, model, factor):
    """`--time` goes through `time_str2mins()`, which rounds seconds up
    (parse_time.c L841-L847): a Slurm job never has a sub-minute limit.
    These used to be fractional seconds."""
    records = k8strace.read_csv(_csv(tmp_path))
    for limit, r in zip(k8strace.time_limits(records, model, factor, seed=1), records, strict=True):
        assert limit % 60 == 0 and limit >= r.duration


def test_whole_minutes_rounds_up_and_never_below():
    assert [whole_minutes(s) for s in (0.5, 60.0, 60.000001, 119.9, 3600.0)] == [
        60.0, 60.0, 120.0, 120.0, 3600.0,
    ]  # fmt: skip
    rng = random.Random(0)
    for _ in range(10_000):
        s = rng.uniform(0, 1e6)
        assert s <= whole_minutes(s) < s + 60 and whole_minutes(s) % 60 == 0


def test_qos_mapping_normalises_like_slurm_and_builds_a_preemption_ladder(tmp_path):
    s0 = k8strace.load(_csv(tmp_path), time_limit_model="exact", grace_time=30.0)
    j = {x.job_id: x for x in s0.jobs}
    assert [j[i].qos for i in (1, 3, 4)] == ["p0", "p100", "p500"]
    # "normalized to the highest priority of all the QOSs".
    assert [j[i].qos_factor for i in (1, 3, 4)] == [0.0, 0.2, 1.0]
    assert s0.qos["p500"].preempt == frozenset({"p0", "p100"})
    assert s0.qos["p0"].preempt == frozenset()
    assert s0.qos["p100"].grace_time == 30.0 and s0.partitions == {}


def test_tier_mapping_uses_partition_priority_tier(tmp_path):
    s0 = k8strace.load(_csv(tmp_path), time_limit_model="exact", priority_mapping="tier")
    assert {x.partition for x in s0.jobs} == {"p0", "p100", "p500"}
    assert {n: p.priority_tier for n, p in s0.partitions.items()} == {
        "p0": 1,
        "p100": 2,
        "p500": 3,
    }
    assert all(x.qos is None and x.qos_factor == 0.0 for x in s0.jobs)
    none = k8strace.load(_csv(tmp_path), time_limit_model="exact", priority_mapping="none")
    assert none.qos == {} and none.partitions == {}


def test_negative_priorities_and_bad_gangs_are_refused(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        k8strace.load(_csv(tmp_path, body="1,a,0,10,1,1,-5\n"), time_limit_model="exact")
    with pytest.raises(ValueError, match="gang_size"):
        k8strace.load(_csv(tmp_path, body="1,a,0,10,1,0,0\n"), time_limit_model="exact")


def test_digest_matches_the_k8s_lab_rule(tmp_path):
    """Pinned to the value the k8s lab's own code gives, so a change to the
    rule (field order, repr vs a shorter float format, the slice) fails
    here without the k8s lab installed. It only checked the length before."""
    # The four-row fixture alone must differ (checked first: `_csv` rewrites
    # the same file).
    assert k8strace.trace_digest(k8strace.read_csv(_csv(tmp_path))) != K8S_DIGEST
    path = _csv(tmp_path, body=ROWS + LONG_FLOAT_ROW)
    assert k8strace.trace_digest(k8strace.read_csv(path)) == K8S_DIGEST
    km = pytest.importorskip("k8slab.metrics")
    kt = pytest.importorskip("k8slab.trace")
    assert km.trace_digest(kt.read_csv(path)) == K8S_DIGEST


def test_truncated_or_overlong_rows_are_refused_with_the_line(tmp_path):
    """csv fills a short row with None; `int(None)` was a raw TypeError."""
    with pytest.raises(ValueError, match=r"line 3: fewer fields than the header"):
        k8strace.read_csv(_csv(tmp_path, body="1,a,0,10,1,1,0\n2,b,6.0,300.0,2,2\n"))
    with pytest.raises(ValueError, match=r"line 2: more fields than the header"):
        k8strace.read_csv(_csv(tmp_path, body="1,a,0,10,1,1,0,9\n"))


def test_a_gang_starts_all_at_once_on_distinct_nodes(tmp_path):
    """Slurm allocates a job atomically: never part of a gang."""
    body = "1,a,0,100,2,1,0\n2,a,1,100,2,3,0\n"
    s0 = k8strace.load(_csv(tmp_path, body=body), time_limit_model="exact")
    cluster = Cluster.homogeneous(3, 4, 4)
    simulate(s0.jobs, cluster)
    gang = s0.jobs[1]
    # Three nodes with 2 free GPUs exist at t=1 (node 0 has 2 left), so it starts.
    assert gang.start_time == 1.0 and len(set(gang.runs[0].nodes)) == 3


def test_preemption_on_s0_follows_the_qos_ladder(tmp_path):
    body = "1,a,0,1000,4,2,0\n2,b,10,100,4,1,500\n"
    s0 = k8strace.load(_csv(tmp_path, body=body), time_limit_model="exact")
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=s0.qos)
    result = simulate(s0.jobs, Cluster.homogeneous(2, 4, 4), preemption=cfg)
    assert [r.job_id for r in result.preemptions] == [1]
    assert s0.jobs[1].start_time == 10.0
