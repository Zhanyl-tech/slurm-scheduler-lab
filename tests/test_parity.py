"""B5: metrics under the k8s lab's definitions, and the --json document."""

from __future__ import annotations

import json
import math
import random
from dataclasses import fields
from typing import Any

import pytest

from schedlab import parity
from schedlab.metrics import compute
from schedlab.model import Cluster, Job, Node
from schedlab.simulate import SimulationResult, simulate
from tests._factory import job, weights_only

#: Field names of `k8slab.metrics.Metrics` (k8s-gpu-scheduler-lab, read
#: 2026-09-26) that this lab reports under the same definition.
K8S_SHARED_FIELDS = {
    "config", "measured_on_cluster", "fleet", "trace_digest", "jobs", "pods",
    "jobs_completed", "gpu_hours_used", "gpu_hours_idle", "utilization",
    "makespan_hours", "mean_wait", "p95_wait", "fragmentation_structural",
    "fragmentation_structural_rack", "fragmentation_structural_switch", "gang_jobs",
    "gang_stalled", "gang_deadlocked", "gang_deadlock_rate", "gang_assembled",
    "gang_stranded_gpu_hours", "gang_stranded_share", "gang_assembly_p50",
    "gang_assembly_p95", "footprint_wait_spearman", "large_job_gpus",
    "large_job_starvation_ratio", "wait_by_footprint", "topology_declared",
    "placement_multi_pod_jobs", "placement_tier_share", "fairness_ratio", "service_ratio",
    # Added to the k8s lab during this work (its execution layer):
    "gpu_hours_demanded", "startup_overhead_gpu_hours", "preemptions",
    "preempted_gpu_hours_lost", "grace_locked_gpu_hours", "topology_extension_gpu_hours",
}  # fmt: skip


def _cluster() -> Cluster:
    """Two 8-GPU nodes in rack-0 and two 4-GPU nodes in rack-1, one switch."""
    nodes = [
        Node(0, 16, 8, name="a-0", rack="rack-0", switch="switch-0"),
        Node(1, 16, 8, name="a-1", rack="rack-0", switch="switch-0"),
        Node(2, 16, 4, name="b-0", rack="rack-1", switch="switch-0"),
        Node(3, 16, 4, name="b-1", rack="rack-1", switch="switch-0"),
    ]
    return Cluster(nodes, name="mini", topology_declared=True)


def _hand_run() -> tuple[SimulationResult, Cluster, list[Job]]:
    """Four jobs whose every number can be worked out by hand (all weights 0,
    so the queue is submit order).

    t=0   j1: 1 node x 8 GPU, 100 s -> a-0.
    t=0   j2: 2 nodes x 8 GPU, 50 s -> needs a-0 and a-1: blocked until 100.
          The main pass stops at it, so nothing behind it starts on an event.
    t=10  j3: 1 GPU, 20 s; t=20 j4: CPU only, 10 s.
    t=30  backfill: j2 reserved on a-0, a-1 from floor(100, 60 s) = 60, so j3
          (ends 50) starts on a-1 and j4 (ends 40) on a-0's spare CPUs.
    t=100 j2 starts; it ends at 150, the horizon.
    """
    jobs = [
        job(1, 0, 100, gpus_per_node=8, account="x"),
        job(2, 0, 50, nodes=2, gpus_per_node=8, account="y"),
        job(3, 10, 20, gpus_per_node=1, account="x"),
        job(4, 20, 10, account="y"),
    ]
    cluster = _cluster()
    return simulate(jobs, cluster, weights=weights_only()), cluster, jobs


def test_field_names_match_the_k8s_lab():
    assert {f.name for f in fields(parity.ParityMetrics)} == K8S_SHARED_FIELDS


def test_field_names_match_the_installed_k8s_lab_when_importable():
    km = pytest.importorskip("k8slab.metrics")
    theirs = {f.name for f in fields(km.Metrics)}
    assert theirs >= K8S_SHARED_FIELDS
    assert set(parity.NOT_REPORTED) <= theirs


def test_hand_worked_run():
    result, cluster, jobs = _hand_run()
    by = {j.job_id: j for j in jobs}
    assert [by[i].start_time for i in (1, 2, 3, 4)] == [0.0, 100.0, 30.0, 30.0]
    p = parity.compute(result, cluster, config="S0", trace_digest="abc")
    assert (p.jobs, p.pods, p.jobs_completed) == (4, 5, 4)
    # GPU-seconds: 8*100 + 16*50 + 1*20 = 1620 over 24 GPUs x 150 s.
    assert p.gpu_hours_used == pytest.approx(1620 / 3600)
    assert p.makespan_hours == pytest.approx(150 / 3600)
    assert p.utilization == pytest.approx(1620 / (24 * 150))
    assert p.gpu_hours_idle == pytest.approx((24 * 150 - 1620) / 3600)
    # Waits 0, 100, 20, 10.
    assert p.mean_wait == 32.5 and p.p95_wait == 100.0
    assert p.wait_by_footprint["1"] == {"jobs": 1.0, "admitted": 1.0, "mean": 20.0, "p95": 20.0}
    assert p.wait_by_footprint["5-8"]["mean"] == 0.0
    assert p.wait_by_footprint["9-16"]["mean"] == 100.0
    # Large = the largest node (8 GPUs): j1 and j2 (mean 50) over the 1-GPU job (20).
    assert p.large_job_gpus == 8 and p.large_job_starvation_ratio == pytest.approx(2.5)
    # The atomic-allocation zeros, and the gang counted as assembled.
    assert (p.gang_jobs, p.gang_assembled, p.gang_assembly_p50) == (1, 1, 0.0)
    assert p.gang_stranded_gpu_hours == p.gang_stranded_share == 0.0
    # j2 spans a-0 and a-1: one rack.
    assert p.placement_multi_pod_jobs == 1 and p.placement_tier_share["rack"] == 1.0
    # Every job completed: delivered equals demanded for every account.
    assert p.service_ratio == {"x": 1.0, "y": 1.0} and p.fairness_ratio == 1.0
    assert p.topology_declared and p.fleet == "mini" and p.trace_digest == "abc"


def test_definition_c_by_hand():
    """Free GPUs per node (a-0, a-1, b-0, b-1), interval by interval.

    [0,30)    (0,8,4,4)  a-0 full: carved, 0 free. rack-0 carved: 0+8.
    [30,50)   (0,7,4,4)  a-1 carved too: node 7; rack-0 7.
    [50,100)  (0,8,4,4)  node 0; rack-0 8.
    [100,150) (0,0,4,4)  node 0; rack-0 0.
    Free GPU-time 16*30 + 15*20 + 16*50 + 8*50 = 1980; node 7*20 = 140;
    rack 8*30 + 7*20 + 8*50 = 780; the one switch is carved throughout, so
    switch = 1980. rack-1 (the 4-GPU nodes) is never carved.
    """
    result, cluster, _ = _hand_run()
    p = parity.compute(result, cluster)
    assert p.fragmentation_structural == pytest.approx(140 / 1980)
    assert p.fragmentation_structural_rack == pytest.approx(780 / 1980)
    assert p.fragmentation_structural_switch == pytest.approx(1.0)
    m = compute(result, cluster)
    assert m.frag_c_node == p.fragmentation_structural


def test_spearman_and_percentile_rules():
    assert parity.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert parity.spearman([1, 1, 1], [1, 2, 3]) is None
    assert parity.spearman([1], [1]) is None
    assert parity.average_ranks([5, 1, 5, 3]) == [3.5, 1.0, 3.5, 2.0]
    assert parity.percentile([], 95) == 0.0
    assert parity.percentile([4, 1, 3, 2], 50) == 2
    assert [parity.footprint_bucket(g) for g in (0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 64)] == [
        None, "1", "2", "3-4", "3-4", "5-8", "5-8", "9-16", "9-16", "17+", "17+",
    ]  # fmt: skip


def test_the_two_percentile_rules_differ_where_the_k8s_lab_says():
    """Rounded rank (`p95_wait`) against nearest-rank (bucket p95).

    Eleven values: 0.95 × 11 = 10.45, rank 10 rounded and rank 11 by
    ceiling. The k8s docs/metrics.md: the rules differ for every n from 11
    to 19. The median of 1..5: 2.5 rounds half-to-even to 2, ceils to 3.
    """
    eleven = [float(v) for v in range(1, 12)]
    assert parity.percentile(eleven, 95) == 10.0
    assert parity.nearest_rank(eleven, 95) == 11.0
    assert parity.percentile([1, 2, 3, 4, 5], 50) == 2
    assert parity.nearest_rank([1, 2, 3, 4, 5], 50) == 3
    differ = [
        n for n in range(1, 40)
        if parity.percentile(list(range(n)), 95) != parity.nearest_rank(list(range(n)), 95)
    ]  # fmt: skip
    assert differ[:9] == list(range(11, 20))
    # pct × n is computed first, so an exact rank is not pushed up by float error.
    assert parity.nearest_rank(list(range(1, 101)), 7) == 7
    assert parity.nearest_rank([3.0], 0) == 3.0
    with pytest.raises(ValueError, match="empty"):
        parity.nearest_rank([], 95)
    with pytest.raises(ValueError, match="pct"):
        parity.nearest_rank([1.0], 101)


def test_bucket_p95_is_nearest_rank_and_p95_wait_the_rounded_rank():
    """Eleven 1-GPU jobs of 10 s on one 1-GPU node, all submitted at 0, FIFO:
    each starts when the previous ends, so the waits are 0, 10, ..., 100.
    The top-level p95 is rank round(10.45) = 10 (90 s); the bucket p95 is
    rank ceil(10.45) = 11 (100 s), as in `k8slab.metrics.compute`."""
    cluster = Cluster.homogeneous(1, 1, 1)
    work = [job(i, 0, 10, gpus_per_node=1) for i in range(1, 12)]
    p = parity.compute(simulate(work, cluster, weights=weights_only()), cluster)
    assert sorted(j.wait_time for j in work) == [10.0 * k for k in range(11)]
    assert p.p95_wait == 90.0
    assert p.wait_by_footprint["1"] == {"jobs": 11.0, "admitted": 11.0, "mean": 50.0, "p95": 100.0}


def test_helpers_agree_with_the_k8s_lab_when_importable():
    km = pytest.importorskip("k8slab.metrics")
    rng = random.Random(0)
    for _ in range(100):
        xs = [float(rng.choice([1, 2, 4, 8])) for _ in range(rng.randint(0, 20))]
        ys = [rng.random() for _ in xs]
        assert km.spearman(xs, ys) == parity.spearman(xs, ys)
        assert km.percentile(ys, 95) == parity.percentile(ys, 95)
    for n in range(1, 60):
        values = [rng.random() for _ in range(n)]
        for pct in (0, 7, 50, 95, 99, 100):
            assert km.nearest_rank(values, pct) == parity.nearest_rank(values, pct)
    assert km.FOOTPRINT_BUCKETS == parity.FOOTPRINT_BUCKETS


def test_json_document(tmp_path):
    result, cluster, _ = _hand_run()
    p = parity.compute(result, cluster, config="S0")
    doc = parity.to_json(p, compute(result, cluster), model={"trace": "k8s"})
    assert set(doc) >= K8S_SHARED_FIELDS
    assert doc["src"] == "model" and doc["measured_on_cluster"] is False
    assert set(doc["structural_zeros"]) >= {"gang_stranded_gpu_hours", "gang_deadlock_rate"}
    assert "fragmentation_rate" in doc["not_reported"]
    assert doc["slurm"]["frag_c_node"] == doc["fragmentation_structural"]
    path = tmp_path / "out.json"
    parity.write_json(doc, path)
    assert json.loads(path.read_text())["model"] == {"trace": "k8s"}


def test_json_has_no_infinities():
    assert parity._finite({"a": [math.inf, 1.0], "b": math.nan}) == {"a": [None, 1.0], "b": None}


def test_unschedulable_jobs_count_as_never_admitted():
    """A job larger than every node is in the trace's demand (k8s
    docs/metrics.md: demanded is "the trace's work") and delivers nothing.

    acct: 1 GPU × 10 s delivered of 1 × 10 + 16 × 10 = 170 demanded;
    other: 10 of 10. Fairness is the ratio of the two: 1 / (10/170) = 17.
    """
    cluster = _cluster()
    jobs = [
        job(1, 0, 10, gpus_per_node=1),
        job(2, 0, 10, gpus_per_node=16),
        job(3, 0, 10, gpus_per_node=1, account="other"),
    ]
    result = simulate(jobs, cluster)
    assert [j.job_id for j in result.unschedulable] == [2]
    p = parity.compute(result, cluster)
    assert (p.jobs, p.jobs_completed) == (3, 2)
    assert p.wait_by_footprint["9-16"] == {"jobs": 1.0, "admitted": 0.0, "mean": None, "p95": None}
    assert p.gpu_hours_demanded == pytest.approx(180 / 3600)
    assert p.service_ratio == pytest.approx({"acct": 10 / 170, "other": 1.0})
    assert p.fairness_ratio == pytest.approx(17.0)


def test_placement_tier_share_is_null_without_multi_node_jobs(tmp_path):
    """0/0 is undefined: the k8s lab reports every share as null there, since
    a 0.0 read as "never placed at that tier" and averaged as a zero."""
    cluster = _cluster()
    result = simulate(
        [job(1, 0, 10, gpus_per_node=1), job(2, 5, 10, gpus_per_node=8)], cluster
    )
    p = parity.compute(result, cluster)
    assert p.placement_multi_pod_jobs == 0
    assert p.placement_tier_share == dict.fromkeys(parity.TIERS)
    path = tmp_path / "single.json"
    parity.write_json(parity.to_json(p, compute(result, cluster)), path)
    assert json.loads(path.read_text())["placement_tier_share"] == dict.fromkeys(parity.TIERS)


def test_text_report_and_json_share_the_gpu_and_preemption_numbers():
    """One GPU-bearing preemption, worked by hand, read from both reports.

    Two nodes of 2 CPUs and 1 GPU. Job 1 (low, 2 nodes × 2 CPU × 1 GPU,
    1000 s) starts at its submit, 100. Job 2 (high, 1 node) arrives at 110
    and selects it: grace 50 s, released at 160, checkpoint keeps half of
    the 10 s run. Job 2 runs 160-260. Job 1 is eligible at 160 + 121 = 281
    and runs its remaining 995 s to 1276.

    * GPU utilization: delivered 2 × 1000 + 1 × 100 = 2100 GPU-s over
      2 GPUs × 1276 s. The horizon runs from trace t=0, not from the first
      submit (makespan 1176), so the two denominators differ here.
    * Work lost 5 s on 2 nodes: 20 CPU-s, 10 GPU-s. Grace-locked 50 s: 200
      CPU-s, 100 GPU-s. CPUs and GPUs differ per node, so a mix-up shows.
    """
    from schedlab.preempt import PartitionSpec, PreemptionConfig, PreemptMode

    cfg = PreemptionConfig(
        "preempt/partition_prio",
        PreemptMode("REQUEUE"),
        partitions={
            "low": PartitionSpec("low", 1, grace_time=50.0),
            "high": PartitionSpec("high", 2),
        },
        checkpoint_fraction=0.5,
    )
    cluster = Cluster.homogeneous(2, 2, 1)
    jobs = [
        job(1, 100, 1000, nodes=2, cpus_per_node=2, gpus_per_node=1, partition="low"),
        job(2, 110, 100, cpus_per_node=2, gpus_per_node=1, partition="high"),
    ]
    result = simulate(jobs, cluster, weights=weights_only(), preemption=cfg)
    assert [(r.start, r.end) for r in jobs[0].runs] == [(100.0, 160.0), (281.0, 1276.0)]
    assert (result.horizon, result.makespan) == (1276.0, 1176.0)
    m, p = compute(result, cluster), parity.compute(result, cluster)
    assert m.gpu_utilization == p.utilization == pytest.approx(2100 / (2 * 1276))
    assert m.work_lost_cpu_hours == pytest.approx(20 / 3600)
    assert m.work_lost_gpu_hours == p.preempted_gpu_hours_lost == pytest.approx(10 / 3600)
    assert m.grace_locked_cpu_hours == pytest.approx(200 / 3600)
    assert m.grace_locked_gpu_hours == p.grace_locked_gpu_hours == pytest.approx(100 / 3600)
    doc = parity.to_json(p, m)
    assert doc["slurm"]["gpu_utilization"] == doc["utilization"]
    assert doc["slurm"]["work_lost_gpu_hours"] == doc["preempted_gpu_hours_lost"]
    assert doc["slurm"]["grace_locked_gpu_hours"] == doc["grace_locked_gpu_hours"]


def test_preemption_keys_follow_the_k8s_accounting():
    """Pods counted per node, lost and grace-locked disjoint, lost work not
    delivered — k8s docs/metrics.md, "Execution layer"."""
    from schedlab.preempt import PartitionSpec, PreemptionConfig, PreemptMode

    cfg = PreemptionConfig(
        "preempt/partition_prio",
        PreemptMode("REQUEUE"),
        partitions={
            "low": PartitionSpec("low", 1, grace_time=50.0),
            "high": PartitionSpec("high", 2),
        },
        checkpoint_fraction=0.5,
    )
    cluster = Cluster.homogeneous(2, 1, 1)
    jobs = [
        job(1, 0, 1000, nodes=2, gpus_per_node=1, partition="low"),
        job(2, 10, 100, gpus_per_node=1, partition="high"),
    ]
    result = simulate(jobs, cluster, weights=weights_only(), preemption=cfg)
    p = parity.compute(result, cluster)
    # One preemption of a 2-node job is two evicted pod attempts.
    assert p.preemptions == 2
    # Selected at 10 after running 10 s on 2 GPUs; half kept by the checkpoint.
    assert p.preempted_gpu_hours_lost == pytest.approx(2 * 5.0 / 3600)
    assert p.grace_locked_gpu_hours == pytest.approx(2 * 50.0 / 3600)
    # Everything completed, so delivered equals demanded exactly.
    assert p.gpu_hours_used == pytest.approx(p.gpu_hours_demanded)
    assert p.gpu_hours_demanded == pytest.approx((2 * 1000 + 100) / 3600)
    assert p.startup_overhead_gpu_hours == p.topology_extension_gpu_hours == 0.0
    # The text report's wait runs to the victim's final start: requeued at
    # 60, eligible 181.
    assert jobs[0].wait_time == 181.0
    # The k8s lab counts pending time only (its docs/metrics.md, "Under
    # eviction: pending time only"): [0, 0) is empty, then [10, 181) from its
    # selection (the k8s evict_time) to its rebind. The 10 s it ran is not
    # wait. Job 2 waits from 10 to the grace release at 60.
    assert parity.pending_wait(jobs[0]) == 171.0 and parity.pending_wait(jobs[1]) == 50.0
    assert p.mean_wait == (171.0 + 50.0) / 2
    assert compute(result, cluster).mean_wait == (181.0 + 50.0) / 2  # the text report's rule


def _tiers_config(**kw: Any) -> Any:
    from schedlab.preempt import PartitionSpec, PreemptionConfig, PreemptMode

    return PreemptionConfig(
        "preempt/partition_prio",
        PreemptMode.parse(kw.pop("mode", "REQUEUE")),
        partitions={"low": PartitionSpec("low", 1), "high": PartitionSpec("high", 2)},
        **kw,
    )


def test_pending_wait_by_hand_for_a_job_preempted_twice_and_a_cancelled_one():
    """Two nodes of one CPU; grace 0, requeue_delay 120.

    Job 1 (low, 2 nodes) runs from 0, is selected and released at 10,
    eligible 131, runs again from 131; job 3 (high) selects it at 200;
    eligible 321, and it runs from 321. Pending: [10, 131) + [200, 321) =
    242 s. The text report's wait is 321 s.
    """
    jobs = [
        job(1, 0, 1000, nodes=2, partition="low"),
        job(2, 10, 100, partition="high"),
        job(3, 200, 50, partition="high"),
    ]
    cluster = Cluster.homogeneous(2, 1)
    simulate(jobs, cluster, weights=weights_only(), preemption=_tiers_config())
    victim = jobs[0]
    assert [(r.start, r.preempt_time, r.outcome) for r in victim.runs] == [
        (0.0, 10.0, "requeued"), (131.0, 200.0, "requeued"), (321.0, None, "completed"),
    ]  # fmt: skip
    assert parity.pending_wait(victim) == 242.0 and victim.wait_time == 321.0
    # A cancelled run is its own final attempt: nothing is re-pending.
    jobs = [job(1, 0, 1000, nodes=2, partition="low"), job(2, 10, 100, partition="high")]
    simulate(jobs, Cluster.homogeneous(2, 1), weights=weights_only(),
             preemption=_tiers_config(mode="CANCEL"))  # fmt: skip
    assert parity.pending_wait(jobs[0]) == jobs[0].wait_time == 0.0


def test_pending_wait_equals_wait_time_without_preemption():
    _, _, jobs = _hand_run()
    assert [parity.pending_wait(j) for j in jobs] == [j.wait_time for j in jobs]


def test_pending_wait_agrees_with_the_k8s_lab_when_importable():
    """Every job of a preemption-heavy synthetic run, against
    `k8slab.metrics._pending_seconds` fed the same attempts: one pod per
    node, each earlier attempt (bind = run start, evict = selection)."""
    km = pytest.importorskip("k8slab.metrics")
    kmodel = pytest.importorskip("k8slab.model")
    from schedlab.cli import _synthetic_ladder
    from schedlab.preempt import PreemptionConfig, PreemptMode
    from schedlab.trace import WorkloadProfile, generate

    jobs = generate(WorkloadProfile(job_count=150), seed=2)
    ladder = _synthetic_ladder(jobs, "preempt/qos", 300.0)
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder.qos)
    result = simulate(jobs, Cluster.homogeneous(16, 8, 2), preemption=cfg)
    requeued = [j for j in result.jobs if j.requeue_count]
    assert requeued, "the scenario should requeue jobs"
    for j in result.jobs:
        kjob = kmodel.Job(j.job_id, j.account, j.submit_time, j.duration, j.gpus_per_node, j.nodes)
        pods = [kmodel.PodEvent(j.job_id, i, scheduled_time=j.start_time) for i in range(j.nodes)]
        earlier = [(r.start, r.preempt_time) for r in j.runs[:-1] if r.outcome == "requeued"]
        attempts = {(j.job_id, i): list(earlier) for i in range(j.nodes)} if earlier else {}
        theirs = (
            km._pending_seconds(kjob, pods, attempts)
            if attempts
            else max(0.0, (j.start_time or 0.0) - j.submit_time)
        )
        assert parity.pending_wait(j) == theirs


def test_placement_tiers_above_rack():
    """Four nodes: a-0, a-1 in rack-0 and b-0 in rack-1 under switch-0; c-0
    in rack-2 under switch-1. FIFO, first-fit by node id, no overlap:
    j1 (2 nodes) on a-0, a-1: rack. j2 (3 nodes) adds b-0: switch.
    j3 (4 nodes) spans both switches: cross-switch. j4 (2 nodes): rack.
    Only a one-rack case was tested, so reading the switch where the rack
    belongs went unnoticed."""
    nodes = [
        Node(0, 1, 1, name="a-0", rack="rack-0", switch="switch-0"),
        Node(1, 1, 1, name="a-1", rack="rack-0", switch="switch-0"),
        Node(2, 1, 1, name="b-0", rack="rack-1", switch="switch-0"),
        Node(3, 1, 1, name="c-0", rack="rack-2", switch="switch-1"),
    ]
    cluster = Cluster(nodes, name="two-switch", topology_declared=True)
    jobs = [
        job(1, 0, 10, nodes=2, gpus_per_node=1),
        job(2, 20, 10, nodes=3, gpus_per_node=1),
        job(3, 40, 10, nodes=4, gpus_per_node=1),
        job(4, 60, 10, nodes=2, gpus_per_node=1),
    ]
    result = simulate(jobs, cluster, weights=weights_only())
    assert [j.start_time for j in jobs] == [0.0, 20.0, 40.0, 60.0]
    p = parity.compute(result, cluster)
    assert p.placement_multi_pod_jobs == 4
    assert p.placement_tier_share == {
        "node": 0.0, "rack": 0.5, "switch": 0.25, "cross-switch": 0.25,
    }  # fmt: skip
