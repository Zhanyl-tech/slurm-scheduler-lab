"""Tests pinning the behaviours that make the simulator worth trusting."""

from __future__ import annotations

import pytest

from schedlab.metrics import compute
from schedlab.model import Cluster, Job, Node
from schedlab.priority import (
    MAX_PRIORITY,
    FairshareTree,
    PriorityWeights,
    _parse_duration,
    age_factor,
    compute_priority,
    controller_priority,
    jobsize_factor,
)
from schedlab.simulate import Scheduler, simulate
from schedlab.trace import WorkloadProfile, _slurm_time, _tres_gpus, generate


def job(job_id: int, submit: float, duration: float, nodes: int = 1, **kw) -> Job:
    kw.setdefault("time_limit", duration)
    return Job(
        job_id=job_id,
        account=kw.pop("account", "acct"),
        submit_time=submit,
        duration=duration,
        nodes=nodes,
        cpus_per_node=kw.pop("cpus_per_node", 1),
        **kw,
    )


# ── Resource model ──────────────────────────────────────────────────────────


def test_node_rejects_overallocation():
    node = Node(0, cpus=4, gpus=1)
    node.allocate(4, 1)
    assert not node.fits(1, 0)
    with pytest.raises(ValueError):
        node.allocate(1, 0)


def test_release_never_exceeds_capacity():
    node = Node(0, cpus=4)
    node.allocate(2, 0)
    node.release(2, 0)
    node.release(2, 0)  # double release must not inflate capacity
    assert node.free_cpus == 4


def test_find_nodes_needs_every_node_available():
    cluster = Cluster.homogeneous(count=2, cpus=4)
    cluster.nodes[0].allocate(4, 0)
    assert cluster.find_nodes(job(1, 0, 10, nodes=2)) is None
    assert cluster.find_nodes(job(2, 0, 10, nodes=1)) == [1]


# ── Priority factors ────────────────────────────────────────────────────────


def test_age_factor_saturates_at_max_age():
    j = job(1, submit=0, duration=10)
    assert age_factor(j, now=0, max_age=100) == 0.0
    assert age_factor(j, now=50, max_age=100) == 0.5
    assert age_factor(j, now=500, max_age=100) == 1.0


def test_favor_small_inverts_jobsize_factor():
    cluster = Cluster.homogeneous(count=10, cpus=1)
    big = job(1, 0, 10, nodes=8)
    assert jobsize_factor(big, cluster, favor_small=False) == pytest.approx(0.8)
    assert jobsize_factor(big, cluster, favor_small=True) == pytest.approx(0.2)


def test_jobsize_factor_averages_node_and_cpu_fractions_like_slurm():
    """`set_priority_factors()` (priority_multifactor.c L2175-L2244):
    `(min_nodes/node_count + cpu_cnt/cluster_cpus) / 2`, or for FavorSmall
    `((node_count - min_nodes)/node_count + (cluster_cpus - cpu_cnt)/cluster_cpus) / 2`.
    """
    cluster = Cluster.homogeneous(count=26, cpus=64)  # 26 nodes, 1664 CPUs
    gang = job(1, 0, 10, nodes=8, cpus_per_node=1)  # an S0 gang: 8 one-CPU pods
    large = (8 / 26 + 8 / 1664) / 2
    assert jobsize_factor(gang, cluster, favor_small=False) == pytest.approx(large)
    assert large > 30 * (8 / 1664)  # the CPU fraction alone under-weighted it ~30x
    small = ((26 - 8) / 26 + (1664 - 8) / 1664) / 2
    assert jobsize_factor(gang, cluster, favor_small=True) == pytest.approx(small)
    # A job asking for every node: the favor-small node term is 0, not negative.
    every = job(2, 0, 10, nodes=26, cpus_per_node=1)
    assert jobsize_factor(every, cluster, favor_small=True) == pytest.approx(
        (1664 - 26) / 1664 / 2
    )
    # Whole-node jobs on a homogeneous cluster: both fractions are equal, so
    # the factor is the CPU fraction, as before (the synthetic workload's case).
    whole = job(3, 0, 10, nodes=4, cpus_per_node=64)
    assert jobsize_factor(whole, cluster, favor_small=False) == pytest.approx(4 / 26)


def test_priority_is_stored_as_slurms_integer():
    """`_get_priority_internal()`: below 1 becomes 1, then `(uint32_t)` truncation."""
    assert controller_priority(500.7) == 500
    assert controller_priority(1.0) == 1
    assert controller_priority(0.4) == 1
    assert controller_priority(-10.0) == 1
    assert controller_priority(float("nan")) == 1
    assert controller_priority(1e12) == MAX_PRIORITY == 2**32 - 1


@pytest.mark.parametrize("mode", ["easy", "conservative"])
def test_near_equal_priorities_tie_and_fall_to_submit_time(mode):
    """Sums of 500.2 (submitted first) and 500.7 are both stored as 500 in
    Slurm, and `sort_job_queue2()` then orders by submit time: the earlier
    job runs first. On float priorities the later one used to win."""
    weights = PriorityWeights(age=0, fairshare=0, jobsize=0, partition=0, qos=1000)
    blocker = job(1, 0, 100)
    first = job(2, 1, 10, qos_factor=0.5002)
    second = job(3, 2, 10, qos_factor=0.5007)
    simulate(
        [blocker, first, second], Cluster.homogeneous(1, 1), weights=weights, backfill_mode=mode
    )
    assert first.priority == second.priority == 500
    assert first.start_time == 100.0
    assert second.start_time == 110.0


def test_fairshare_halves_at_exact_share_usage():
    # The classic formula. Fair Tree, now the default as in Slurm, ranks instead.
    tree = FairshareTree(shares={"a": 1.0, "b": 1.0}, algorithm="classic")
    tree.charge("a", 100)
    tree.charge("b", 100)
    # Each account uses exactly its half -> F = 2^-1 = 0.5
    assert tree.factor("a") == pytest.approx(0.5)

    tree.charge("a", 300)  # now a is heavily over its share
    assert tree.factor("a") < tree.factor("b")


def test_fairshare_decay_reduces_recorded_usage():
    tree = FairshareTree(shares={"a": 1.0}, half_life=100.0)
    tree.charge("a", 80.0)
    tree.decay(100.0)
    assert tree.usage["a"] == pytest.approx(40.0)


def test_priority_weights_parse_from_slurm_conf(tmp_path):
    conf = tmp_path / "slurm.conf"
    conf.write_text(
        "PriorityType=priority/multifactor\n"
        "PriorityWeightFairshare=10000   # inline comment\n"
        "PriorityWeightAge=1000\n"
        "PriorityMaxAge=7-0\n"
        "PriorityFavorSmall=YES\n"
    )
    weights = PriorityWeights.from_slurm_conf(str(conf))
    assert weights.fairshare == 10_000
    assert weights.age == 1_000
    assert weights.max_age == 7 * 86_400
    assert weights.favor_small is True


def test_parse_duration_handles_slurm_syntax():
    assert _parse_duration("5-0") == 5 * 86_400
    assert _parse_duration("1-12:00:00") == 86_400 + 12 * 3600
    assert _parse_duration("00:30:00") == 1800


def test_weights_alone_decide_ordering():
    """Same jobs, different weights, different winner — the core claim."""
    cluster = Cluster.homogeneous(count=10, cpus=1)
    big_new = job(1, submit=100, duration=10, nodes=8)
    small_old = job(2, submit=0, duration=10, nodes=1)

    size_first = PriorityWeights(age=0, fairshare=0, jobsize=10_000, qos=0)
    age_first = PriorityWeights(age=10_000, fairshare=0, jobsize=0, qos=0, max_age=200)

    p_big = compute_priority(big_new, 200, cluster, size_first)
    p_small = compute_priority(small_old, 200, cluster, size_first)
    assert p_big > p_small

    p_big = compute_priority(big_new, 200, cluster, age_first)
    p_small = compute_priority(small_old, 200, cluster, age_first)
    assert p_small > p_big


# ── Backfill ────────────────────────────────────────────────────────────────


def test_shadow_time_is_when_enough_resources_free_up():
    cluster = Cluster.homogeneous(count=4, cpus=1)
    scheduler = Scheduler(cluster)

    running = [job(1, 0, 100, nodes=2), job(2, 0, 50, nodes=2)]
    for r in running:
        r.start_time = 0.0
        nodes = cluster.find_nodes(r)
        assert nodes is not None
        cluster.allocate(r, nodes)

    blocked = job(3, 0, 10, nodes=3)
    # Freeing job 2 (ends at 50) leaves 2 nodes — not enough. Job 1 also has to
    # go, so the shadow time is 100.
    assert scheduler._shadow_time(blocked, running, now=0.0) == 100.0


def test_backfill_runs_a_short_job_that_fits_in_the_hole():
    """The canonical EASY case: a small short job jumps a blocked big job."""
    cluster = Cluster.homogeneous(count=4, cpus=1)

    jobs = [
        job(1, submit=0, duration=100, nodes=3),   # occupies 3 of 4 nodes
        job(2, submit=1, duration=100, nodes=4),   # blocked: needs all 4
        job(3, submit=2, duration=10, nodes=1),    # fits the 1 free node, ends early
    ]

    # EASY decides on every event; the conservative default waits for the next
    # bf_interval tick (see test_backfill_conservative.py).
    result = simulate(jobs, cluster, backfill=True, backfill_mode="easy")
    filler = next(j for j in jobs if j.job_id == 3)

    assert 3 in result.backfilled_ids
    # It starts as soon as it arrives rather than waiting behind job 2.
    assert filler.start_time == pytest.approx(2.0)
    # The report counts backfill from each run's own flag (RunRecord.backfilled),
    # which EASY sets when a job jumps the reservation. Without it every EASY
    # report read "backfilled jobs 0" and no test noticed.
    assert filler.runs[0].backfilled
    m = compute(result, cluster)
    assert m.backfilled == m.backfill_starts == len(result.backfilled_ids) == 1


def test_backfill_refuses_a_job_that_would_delay_the_reservation():
    cluster = Cluster.homogeneous(count=4, cpus=1)

    jobs = [
        job(1, submit=0, duration=100, nodes=3),
        job(2, submit=1, duration=100, nodes=4),   # reserved for t=100
        # Runs long past the shadow time and needs the reserved node.
        job(3, submit=2, duration=500, nodes=1, time_limit=500),
    ]

    simulate(jobs, cluster, backfill=True)
    blocker, filler = jobs[1], jobs[2]

    assert filler.start_time is not None
    assert filler.start_time >= 100.0, "long job must not steal the reservation"
    assert blocker.start_time == pytest.approx(100.0)


def test_backfill_plans_against_time_limit_not_true_runtime():
    """Over-requesting must be visible: it pushes the shadow time out."""
    honest = [
        job(1, 0, duration=100, nodes=3, time_limit=100),
        job(2, 1, duration=100, nodes=4, time_limit=100),
        job(3, 2, duration=50, nodes=1, time_limit=50),
    ]
    padded = [
        job(1, 0, duration=100, nodes=3, time_limit=100),
        job(2, 1, duration=100, nodes=4, time_limit=100),
        # Same real runtime, but claims 500s — no longer provably safe.
        job(3, 2, duration=50, nodes=1, time_limit=500),
    ]

    simulate(honest, Cluster.homogeneous(4, 1), backfill=True, backfill_mode="easy")
    simulate(padded, Cluster.homogeneous(4, 1), backfill=True, backfill_mode="easy")

    assert honest[2].start_time == pytest.approx(2.0)
    assert padded[2].start_time is not None and honest[2].start_time is not None
    assert padded[2].start_time > honest[2].start_time


def test_backfill_beats_no_backfill_on_utilization():
    jobs = generate(WorkloadProfile(job_count=150), seed=7)

    off_cluster = Cluster.homogeneous(16, 8, 2)
    off = compute(simulate([*map(_clone, jobs)], off_cluster, backfill=False), off_cluster)

    on_cluster = Cluster.homogeneous(16, 8, 2)
    on = compute(simulate([*map(_clone, jobs)], on_cluster, backfill=True), on_cluster)

    assert on.mean_wait < off.mean_wait
    assert on.makespan <= off.makespan


def test_oversized_jobs_are_reported_not_fatal():
    """A job bigger than the cluster must not take the whole run down."""
    cluster = Cluster.homogeneous(count=2, cpus=4)
    jobs = [
        job(1, submit=0, duration=10, nodes=1),
        job(2, submit=0, duration=10, nodes=8),          # more nodes than exist
        job(3, submit=0, duration=10, nodes=1, cpus_per_node=99),  # too wide
    ]

    result = simulate(jobs, cluster, backfill=True)

    assert [j.job_id for j in result.unschedulable] == [2, 3]
    assert [j.job_id for j in result.jobs] == [1]
    assert result.jobs[0].end_time is not None


def test_all_jobs_oversized_returns_empty_not_crash():
    cluster = Cluster.homogeneous(count=1, cpus=1)
    result = simulate([job(1, 0, 10, nodes=64)], cluster, backfill=True)
    assert result.jobs == []
    assert len(result.unschedulable) == 1
    assert result.makespan == 0.0


def _clone(j: Job) -> Job:
    import copy

    return copy.deepcopy(j)


# ── Simulation invariants ───────────────────────────────────────────────────


def test_every_job_completes_and_never_starts_before_submit():
    jobs = generate(WorkloadProfile(job_count=120), seed=3)
    cluster = Cluster.homogeneous(16, 8, 2)
    simulate(jobs, cluster, backfill=True)

    for j in jobs:
        assert j.start_time is not None and j.end_time is not None
        assert j.start_time >= j.submit_time
        assert j.end_time == pytest.approx(j.start_time + j.duration)

    # Everything is handed back at the end.
    assert cluster.busy_cpus == 0


def test_cluster_capacity_is_never_exceeded():
    jobs = generate(WorkloadProfile(job_count=200), seed=11)
    cluster = Cluster.homogeneous(16, 8, 2)
    simulate(jobs, cluster, backfill=True)

    for node in cluster.nodes:
        assert 0 <= node.free_cpus <= node.cpus
        assert 0 <= node.free_gpus <= node.gpus


def test_simulation_is_deterministic():
    def run() -> list[float | None]:
        jobs = generate(WorkloadProfile(job_count=100), seed=42)
        simulate(jobs, Cluster.homogeneous(16, 8, 2), backfill=True)
        return [j.start_time for j in jobs]

    assert run() == run()


# ── Trace parsing ───────────────────────────────────────────────────────────


def test_slurm_time_parsing():
    assert _slurm_time("00:01:40") == 100
    assert _slurm_time("2-03:00:00") == 2 * 86_400 + 3 * 3600
    assert _slurm_time("UNLIMITED") == 0.0
    assert _slurm_time("10:00") == 600  # MM:SS


def test_tres_gpu_extraction():
    assert _tres_gpus("cpu=16,mem=64G,node=1,billing=16,gres/gpu=4") == 4
    assert _tres_gpus("cpu=16,mem=64G") == 0


def test_synthetic_limits_are_whole_minutes_padded_from_the_runtime():
    """Slurm stores a limit in minutes, rounding seconds up (`time_str2mins()`,
    parse_time.c L841-L847); the generator used to hand the scheduler
    fractional seconds no Slurm job can have. The padding is still
    max(1.05, N(3, 1)) of the runtime before the rounding."""
    jobs = generate(WorkloadProfile(job_count=500), seed=9)
    for j in jobs:
        assert j.time_limit % 60 == 0 and j.time_limit >= 1.05 * j.duration


def test_sacct_round_trip(tmp_path):
    from schedlab.trace import from_sacct

    path = tmp_path / "sacct.txt"
    path.write_text(
        "JobID|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
        "101|research|2026-07-26T09:00:00|00:10:00|01:00:00|2|16|cpu=16,gres/gpu=2\n"
        "102|trading|2026-07-26T09:05:00|00:30:00|00:45:00|1|8|cpu=8\n"
        "103|infra|2026-07-26T09:06:00|00:00:00|01:00:00|1|8|cpu=8\n"  # never ran
    )

    jobs = from_sacct(str(path))
    assert len(jobs) == 2, "jobs that never ran are skipped"

    first = jobs[0]
    assert first.job_id == 101
    assert first.account == "research"
    assert first.submit_time == 0.0  # earliest submit becomes the origin
    assert first.duration == 600
    assert first.time_limit == 3600
    assert first.cpus_per_node == 8
    assert first.gpus_per_node == 1

    assert jobs[1].submit_time == 300.0
