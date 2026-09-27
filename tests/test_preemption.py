"""B1: Slurm preemption (preempt.py) — CANCEL, REQUEUE, GraceTime, and who is chosen.

Every expected time below is worked out by hand from the rules in preempt.py's
docstring (each cited to slurm.conf(5), sacctmgr(1), preempt.html or the
Slurm source), not read back from a run. Common timeline: the main scheduler
runs on every submit and completion and every 60 s from t=0; backfill every
30 s; a requeued job is eligible again `requeue_delay + 1` = 121 s after its
release.
"""

from __future__ import annotations

import copy
import random
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

import schedlab.simulate as sim
from schedlab.metrics import allocated_seconds, compute
from schedlab.model import Cluster, Job, Node, RunRecord, State
from schedlab.preempt import (
    PartitionSpec,
    PreemptionConfig,
    PreemptMode,
    QOSSpec,
    partition_ladder,
    qos_ladder,
    queue_order,
    select_preemptees,
)
from schedlab.priority import PriorityWeights
from schedlab.simulate import SlurmScheduler, simulate
from schedlab.trace import WorkloadProfile, generate
from tests._factory import by_id, clone, job, weights_only

FLAT = weights_only()  # every weight 0: order is submit time, then job id


def tiers(grace: float = 0.0, mode: str = "REQUEUE", **kw: Any) -> PreemptionConfig:
    return PreemptionConfig(
        preempt_type="preempt/partition_prio",
        mode=PreemptMode.parse(mode),
        partitions={
            "low": PartitionSpec("low", priority_tier=1, grace_time=grace),
            "high": PartitionSpec("high", priority_tier=2),
        },
        **kw,
    )


def victim_and_preemptor(duration: float = 1000.0, limit: float | None = None) -> list[Job]:
    return [
        job(1, 0, duration, nodes=2, partition="low", time_limit=limit or duration),
        job(2, 10, 100, nodes=1, partition="high"),
    ]


def run(jobs: list[Job], cfg: PreemptionConfig | None, nodes: int = 2, **kw: Any):
    cluster = Cluster.homogeneous(nodes, 1)
    return simulate(jobs, cluster, weights=FLAT, preemption=cfg, **kw), cluster


# ── Configuration and parsing ───────────────────────────────────────────────


def test_default_config_is_off_and_consistent_with_slurm():
    assert not PreemptionConfig().enabled
    # read_config.c refuses both mismatches.
    with pytest.raises(ValueError, match="incompatible"):
        PreemptionConfig(mode=PreemptMode("REQUEUE"))
    with pytest.raises(ValueError, match="PreemptMode=OFF"):
        PreemptionConfig(preempt_type="preempt/qos")
    with pytest.raises(ValueError, match="checkpoint"):
        tiers(checkpoint_fraction=1.0)


def test_preempt_mode_parsing():
    m = PreemptMode.parse("requeue,within")
    assert (m.base, m.within, m.priority) == ("REQUEUE", True, False)
    assert str(PreemptMode.parse("CANCEL,PRIORITY")) == "CANCEL,PRIORITY"
    with pytest.raises(ValueError, match="SUSPEND"):
        PreemptMode.parse("SUSPEND,GANG")
    with pytest.raises(ValueError, match="unknown"):
        PreemptMode.parse("KILL")


def test_preemption_is_refused_in_easy_mode():
    with pytest.raises(ValueError, match="conservative"):
        run(victim_and_preemptor(), tiers(), backfill_mode="easy")


# ── REQUEUE, CANCEL and GraceTime ───────────────────────────────────────────


def test_requeue_without_grace():
    jobs = victim_and_preemptor()
    result, cluster = run(jobs, tiers())
    victim, preemptor = jobs
    # t=10: job 2 cannot fit, selects job 1; zero grace releases it at t=10
    # and job 2 starts in the pass that release triggers.
    assert preemptor.start_time == 10.0
    # batch_requeue_fini(): new submit time 10, begin time 10 + 120 + 1.
    assert victim.requeue_submit_time == 10.0
    assert victim.eligible_time == 131.0
    assert victim.state is State.COMPLETED
    assert victim.start_time == 131.0 and victim.end_time == 1131.0
    # Wait runs to the last run's start, as the k8s lab measures to the final
    # attempt's bind.
    assert victim.first_start_time == 0.0 and victim.wait_time == 131.0
    assert [(r.start, r.end, r.outcome) for r in victim.runs] == [
        (0.0, 10.0, "requeued"),
        (131.0, 1131.0, "completed"),
    ]
    (rec,) = result.preemptions
    assert (rec.job_id, rec.preemptor_id, rec.mode) == (1, 2, "REQUEUE")
    assert (rec.run_seconds, rec.grace_seconds, rec.lost_seconds) == (10.0, 0.0, 10.0)
    assert cluster.busy_cpus == 0


def test_grace_time_keeps_the_nodes_allocated_until_the_deadline():
    jobs = victim_and_preemptor()
    result, cluster = run(jobs, tiers(grace=50.0))
    victim, preemptor = jobs
    # Selected at 10, end time reset to 10 + 50; job 2 waits for the release.
    assert preemptor.start_time == 60.0
    assert victim.eligible_time == 60.0 + 121.0
    assert victim.runs[0].preempt_time == 10.0 and victim.runs[0].end == 60.0
    (rec,) = result.preemptions
    # Lost is start to selection; the grace period is grace-locked, not both.
    assert (rec.grace_seconds, rec.lost_seconds, rec.exited_in_grace) == (50.0, 10.0, False)
    # Both nodes stay busy through the grace period: no state change at t=10.
    states = {t: cpus for t, _, cpus in result.node_samples}
    assert states[0.0] == (0, 0) and 10.0 not in states and states[60.0] == (0, 1)
    # The backfill plan saw the reset end time and reserved job 2 at 60.
    assert result.planned_starts[2] == 60.0
    m = compute(result, cluster)
    assert m.preemptions == m.requeues == 1
    assert m.grace_locked_cpu_hours == pytest.approx(100.0 / 3600)
    assert m.work_lost_cpu_hours == pytest.approx(20.0 / 3600)


def test_cancel_ends_the_job():
    jobs = victim_and_preemptor()
    result, cluster = run(jobs, tiers(mode="CANCEL"))
    victim, preemptor = jobs
    assert victim.state is State.PREEMPTED
    assert victim.end_time == 10.0 and victim.runs[0].outcome == "cancelled"
    assert preemptor.start_time == 10.0
    assert result.preemptions[0].mode == "CANCEL"
    m = compute(result, cluster)
    assert (m.cancels, m.requeues) == (1, 0)
    # A cancelled job never finished; it is left out of slowdown, not of wait.
    assert m.mean_bounded_slowdown == pytest.approx(preemptor.bounded_slowdown())


def test_requeue_falls_back_to_cancel_without_job_requeue():
    jobs = victim_and_preemptor()
    run(jobs, tiers(job_requeue=False))
    assert jobs[0].state is State.PREEMPTED


def test_a_job_that_exits_inside_its_grace_period_is_still_requeued():
    """slurm.conf(5), GraceTime: "handled according to PreemptMode regardless
    of why it exited (most visible if PreemptMode=REQUEUE)"."""
    jobs = victim_and_preemptor(duration=30.0, limit=100.0)
    result, _ = run(jobs, tiers(grace=50.0))
    victim, preemptor = jobs
    (rec,) = result.preemptions
    assert rec.exited_in_grace and rec.release_time == 30.0
    assert preemptor.start_time == 30.0
    # It finished its work, and runs it again from scratch.
    assert victim.requeue_count == 1
    assert [(r.start, r.end) for r in victim.runs] == [(0.0, 30.0), (151.0, 181.0)]


def test_checkpoint_fraction_shortens_the_rerun():
    jobs = victim_and_preemptor()
    result, _ = run(jobs, tiers(checkpoint_fraction=0.5))
    victim = jobs[0]
    assert victim.saved_progress == 5.0
    assert victim.end_time == 131.0 + 995.0
    assert result.preemptions[0].lost_seconds == 5.0
    assert allocated_seconds(victim) == 10.0 + 995.0


def test_preempt_exempt_time_protects_young_jobs():
    jobs = victim_and_preemptor()
    result, _ = run(jobs, tiers(exempt_time=100.0))
    # Exempt until 100; the next main pass is the periodic one at 120.
    assert result.preemptions[0].preempt_time == 120.0
    assert jobs[1].start_time == 120.0


def test_requeue_delay_sets_the_new_begin_time():
    jobs = victim_and_preemptor()
    run(jobs, tiers(requeue_delay=0.0))
    # Eligible at 11; job 2 holds one node until 110, job 1 needs both.
    assert jobs[0].eligible_time == 11.0
    assert jobs[0].runs[1].start == 110.0


def test_a_job_preempted_twice_counts_as_thrash():
    jobs = [*victim_and_preemptor(), job(3, 200, 50, nodes=1, partition="high")]
    result, cluster = run(jobs, tiers())
    victim = jobs[0]
    assert victim.preempt_count == 2
    assert [r.outcome for r in victim.runs] == ["requeued", "requeued", "completed"]
    m = compute(result, cluster)
    assert (m.thrashed_jobs, m.max_preemptions_per_job) == (1, 2)


def test_backfilled_jobs_count_jobs_and_backfill_starts_count_runs():
    """sdiag's "Total backfilled jobs" counts starts (backfill.c increments
    it per start), so a requeued job backfilled again counts twice there.
    `backfill_starts` keeps that count; `backfilled` counts each job once."""
    from schedlab.cli import _synthetic_ladder

    jobs = generate(WorkloadProfile(job_count=100), seed=5)
    ladder = _synthetic_ladder(jobs, "preempt/qos", 300.0)
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder.qos)
    cluster = Cluster.homogeneous(16, 8, 2)
    result = simulate(jobs, cluster, preemption=cfg)
    m = compute(result, cluster)
    per_job = [sum(r.backfilled for r in j.runs) for j in jobs]
    assert m.backfill_starts == len(result.backfilled_ids) == sum(per_job)
    assert m.backfilled == sum(1 for n in per_job if n) < m.backfill_starts
    # Only a requeued job can be started by backfill more than once.
    assert all(j.requeue_count for j, n in zip(jobs, per_job, strict=True) if n > 1)


def test_thrash_counts_only_jobs_preempted_more_than_once():
    """Two victims, one preempted twice and one once: thrash is 1, not 2.

    Three nodes. At 0, job 1 (low, 2 nodes) and job 4 (low, 1 node) start.
    Job 2 (high) at 10 takes job 4, the smaller low job (`preempt_p_get_prio`);
    job 5 (high) at 30 takes job 1; job 3 (high) at 200 takes job 4 again.
    """
    jobs = [
        job(1, 0, 1000, nodes=2, partition="low"),
        job(2, 10, 100, nodes=1, partition="high"),
        job(3, 200, 50, nodes=1, partition="high"),
        job(4, 0, 1000, nodes=1, partition="low"),
        job(5, 30, 50, nodes=1, partition="high"),
    ]
    result, cluster = run(jobs, tiers(), nodes=3)
    assert [(r.job_id, r.preempt_time) for r in result.preemptions] == [
        (4, 10.0), (1, 30.0), (4, 200.0),
    ]  # fmt: skip
    assert [j.preempt_count for j in jobs] == [1, 0, 0, 2, 0]
    m = compute(result, cluster)
    assert (m.preemptions, m.thrashed_jobs, m.max_preemptions_per_job) == (3, 1, 2)


# ── Who may preempt whom ────────────────────────────────────────────────────


def test_equal_tiers_never_preempt():
    jobs = victim_and_preemptor()
    jobs[1].partition = "low"
    result, _ = run(jobs, tiers())
    assert result.preemptions == [] and jobs[1].start_time == 1000.0


def test_partition_preempt_mode_off_protects_its_jobs():
    cfg = tiers()
    cfg = replace(
        cfg,
        partitions={
            **cfg.partitions,
            "low": PartitionSpec("low", 1, preempt_mode=PreemptMode("OFF")),
        },
    )
    result, _ = run(victim_and_preemptor(), cfg)
    assert result.preemptions == []


def _qos_jobs(lo: str = "lo", hi: str = "hi") -> list[Job]:
    jobs = victim_and_preemptor()
    jobs[0].qos, jobs[1].qos = lo, hi
    return jobs


def test_qos_preempt_lists_decide_not_qos_priority():
    ladder = qos_ladder({"lo": 0, "hi": 10})
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder)
    result, _ = run(_qos_jobs(), cfg)
    assert [r.job_id for r in result.preemptions] == [1]
    # Same QOS: no preemption without WITHIN.
    result, _ = run(_qos_jobs("lo", "lo"), cfg)
    assert result.preemptions == []
    # "The Priority of a QOS is NOT related to QOS preemption" (sacctmgr(1)).
    reversed_list = {
        "lo": QOSSpec("lo", priority=0, preempt=frozenset({"hi"})),
        "hi": QOSSpec("hi", priority=10),
    }
    result, _ = run(_qos_jobs(), replace(cfg, qos=reversed_list))
    assert result.preemptions == []


def test_within_lets_a_qos_preempt_itself_by_job_priority():
    cfg = PreemptionConfig(
        "preempt/qos", PreemptMode.parse("REQUEUE,WITHIN"), qos=qos_ladder({"lo": 0})
    )
    jobs = _qos_jobs("lo", "lo")
    jobs[1].nice = -10  # priority 10 > 1 (a sum below 1 is stored as 1, as in Slurm)
    result, _ = run(jobs, cfg)
    assert [r.job_id for r in result.preemptions] == [1]
    # Sums of 0 and -10 are both stored as 1: equal, so WITHIN does not apply.
    jobs = _qos_jobs("lo", "lo")
    jobs[0].nice = 10
    result, _ = run(jobs, cfg)
    assert result.preemptions == []


def test_qos_grace_time_comes_from_the_preemptees_qos():
    ladder = qos_ladder({"lo": 0, "hi": 10}, grace_time=40.0)
    cfg = PreemptionConfig("preempt/qos", PreemptMode("CANCEL"), qos=ladder)
    jobs = _qos_jobs()
    run(jobs, cfg)
    assert jobs[1].start_time == 50.0


def test_candidates_prefer_smaller_jobs_then_youngest_first_when_asked():
    def scenario(**kw: Any) -> list[int]:
        jobs = [
            job(1, 0, 1000, nodes=1, partition="low"),
            job(2, 5, 1000, nodes=2, partition="low"),
            job(3, 20, 100, nodes=1, partition="high"),
        ]
        result, _ = run(jobs, tiers(**kw), nodes=3)
        return [r.job_id for r in result.preemptions]

    # preempt_p_get_prio(): tier << 16 + node count, ascending.
    assert scenario() == [1]
    # PreemptParameters=youngest_first: latest start first.
    assert scenario(youngest_first=True) == [2]


def test_the_lowest_tier_goes_first_even_when_it_is_larger():
    """`preempt_p_get_prio()` puts the tier in the upper 16 bits and the node
    count in the lower 16, so tier decides before size. A 3-tier ladder: the
    2-node low job is chosen over the 1-node mid job, although fewer nodes
    would be preempted the other way round."""
    ladder = PreemptionConfig(
        "preempt/partition_prio",
        PreemptMode("REQUEUE"),
        partitions={
            "low": PartitionSpec("low", 1),
            "mid": PartitionSpec("mid", 2),
            "high": PartitionSpec("high", 3),
        },
    )
    jobs = [
        job(1, 0, 1000, nodes=2, partition="low"),
        job(2, 5, 1000, nodes=1, partition="mid"),
        job(3, 20, 100, nodes=1, partition="high"),
    ]
    assert ladder.preempt_prio(jobs[0]) < ladder.preempt_prio(jobs[1])
    result, _ = run(jobs, ladder, nodes=3)
    assert [r.job_id for r in result.preemptions] == [1]
    assert jobs[1].preempt_count == 0 and jobs[2].start_time == 20.0


def test_select_preemptees_reorders_to_preempt_fewer_jobs():
    """cons_tres `_run_now()`: the first pass removes A then X; the re-sort
    tries X alone, which suffices, so A is spared."""
    cluster = Cluster([Node(i, 1) for i in range(3)])
    a = job(1, 0, 100, nodes=1, partition="low")
    x = job(2, 0, 100, nodes=2, partition="low")
    for j, nodes in ((a, [0]), (x, [1, 2])):
        cluster.allocate(j, nodes)
        j.start_time, j.state = 0.0, State.RUNNING
    p = job(3, 0, 100, nodes=2, partition="high")
    cfg = tiers()
    cands = cfg.candidates(p, [a, x], 0.0)
    assert cands == [a, x]
    victims, nodes = select_preemptees(p, cands, cluster, cfg) or ([], [])
    assert victims == [x] and nodes == [1, 2]
    victims, _ = select_preemptees(p, cands, cluster, replace(cfg, strict_order=True)) or ([], [])
    assert victims == [x]


def test_one_preemptor_waits_kill_wait_plus_message_timeout_before_preempting_again():
    cluster = Cluster.homogeneous(2, 1)
    low1 = job(1, 0, 1000, partition="low")
    low2 = job(2, 0, 1000, partition="low")
    for j, n in ((low1, [0]), (low2, [1])):
        cluster.allocate(j, n)
        j.start_time, j.state = 0.0, State.RUNNING
        j.runs.append(RunRecord(0.0, tuple(n)))
    hi = job(3, 0, 10, partition="high")
    sched = SlurmScheduler(cluster, preemption=tiers(grace=500.0))
    sched._preempt_for(hi, [low1, low2], 0.0)
    assert sched.drain_signalled() == [low1]
    sched._preempt_for(hi, [low2], 39.0)  # within 30 + 10 s
    assert sched.drain_signalled() == []
    sched._preempt_for(hi, [low2], 41.0)
    assert sched.drain_signalled() == [low2]


# ── Queue order ─────────────────────────────────────────────────────────────


def test_queue_order_puts_preemptors_and_higher_tiers_first():
    low = job(1, 0, 10, partition="low")
    high = job(2, 5, 10, partition="high")
    low.priority, high.priority = 100.0, 0.0
    assert queue_order([low, high], tiers()) == [high, low]
    # PriorityTier orders the queue even with preemption off.
    no_preempt = PreemptionConfig(partitions=partition_ladder({"low": 1, "high": 2}))
    assert queue_order([low, high], no_preempt) == [high, low]
    assert sim.queue_order([low, high]) == [low, high]


def test_requeued_job_sorts_by_its_new_submit_time():
    a, b = job(1, 0, 10), job(2, 5, 10)
    a.requeue_submit_time = 50.0
    assert sim.queue_order([a, b]) == [b, a]


# ── Off by default, and invariants when on ──────────────────────────────────


def test_disabled_or_inert_preemption_changes_nothing():
    base = generate(WorkloadProfile(job_count=120), seed=2)
    plain = clone(base)
    simulate(plain, Cluster.homogeneous(16, 8, 2))
    for cfg in (
        PreemptionConfig(),
        # Enabled, but one partition: nobody outranks anybody.
        PreemptionConfig("preempt/partition_prio", PreemptMode("REQUEUE")),
    ):
        jobs = clone(base)
        result = simulate(jobs, Cluster.homogeneous(16, 8, 2), preemption=cfg)
        assert result.preemptions == []
        assert [j.start_time for j in jobs] == [j.start_time for j in plain]
        assert [j.end_time for j in jobs] == [j.end_time for j in plain]


def _ladder_workload(seed: int) -> tuple[list[Job], PreemptionConfig]:
    jobs = generate(WorkloadProfile(job_count=150), seed=seed)
    names = {lv: f"q{lv:g}" for lv in sorted({j.qos_factor for j in jobs})}
    for j in jobs:
        j.qos = names[j.qos_factor]
    ladder = qos_ladder({n: rank for rank, n in enumerate(names.values())}, grace_time=45.0)
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder,
                           checkpoint_fraction=0.25)
    return jobs, cfg


@pytest.mark.parametrize("seed", [1, 7])
def test_invariants_hold_with_preemption_on(seed):
    jobs, cfg = _ladder_workload(seed)
    cluster = Cluster.homogeneous(16, 8, 2)
    result = simulate(jobs, cluster, weights=PriorityWeights(), preemption=cfg)
    assert result.preemptions, "the scenario should exercise preemption"
    assert cluster.busy_cpus == 0
    for j in jobs:
        assert j.state is State.COMPLETED
        final = j.runs[-1]
        assert final.outcome == "completed" and final.end is not None
        assert final.end - final.start == pytest.approx(j.duration - j.saved_progress)
        assert all(r.outcome == "requeued" for r in j.runs[:-1])
        assert j.requeue_count == len(j.runs) - 1
    # No node is ever overcommitted, run by run, grace periods included.
    for node in cluster.nodes:
        events = []
        for j in jobs:
            for r in j.runs:
                if node.node_id in r.nodes:
                    assert r.end is not None
                    events += [(r.start, 1, j), (r.end, -1, j)]
        cpus = gpus = 0
        for _, sign, j in sorted(events, key=lambda e: (e[0], e[1])):
            cpus += sign * j.cpus_per_node
            gpus += sign * j.gpus_per_node
            assert 0 <= cpus <= node.cpus and 0 <= gpus <= node.gpus


def test_preemption_runs_are_deterministic():
    def once() -> list[tuple[float, float]]:
        jobs, cfg = _ladder_workload(3)
        simulate(jobs, Cluster.homogeneous(16, 8, 2), preemption=cfg)
        return [(r.start, r.end or -1.0) for j in jobs for r in j.runs]

    assert once() == once()


class _GuardedJob(Job):
    """`duration` readable only by `_world_end_time()` (see test_accounting)."""

    reads: ClassVar[dict[int, int]] = {}
    _WORLD = str(Path(sim.__file__).resolve())

    def __getattribute__(self, name: str) -> Any:
        if name == "duration":
            code = sys._getframe(1).f_code
            if code.co_name != "_world_end_time" or str(
                Path(code.co_filename).resolve()
            ) != (_GuardedJob._WORLD):
                raise AssertionError(f"job.duration read by {code.co_filename}:{code.co_name}")
            job_id = object.__getattribute__(self, "job_id")
            _GuardedJob.reads[job_id] = _GuardedJob.reads.get(job_id, 0) + 1
        return object.__getattribute__(self, name)


def test_preemption_never_reads_true_runtime():
    plain, cfg = _ladder_workload(1)
    guarded: list[Job] = [
        _GuardedJob(**{f.name: copy.copy(getattr(j, f.name)) for f in fields(Job) if f.init})
        for j in plain
    ]
    _GuardedJob.reads = {}
    result = simulate(guarded, Cluster.homogeneous(16, 8, 2), preemption=cfg)
    assert result.preemptions
    # Exactly one read per dispatch: the world scheduling each run's end.
    assert _GuardedJob.reads == {j.job_id: len(j.runs) for j in guarded}


def test_tier_mapping_without_preemption_orders_but_never_preempts():
    jobs = [
        job(1, 0, 100, nodes=2, partition="low"),
        job(2, 1, 100, nodes=2, partition="low"),
        job(3, 2, 100, nodes=2, partition="high"),
    ]
    run(jobs, PreemptionConfig(partitions=partition_ladder({"low": 1, "high": 2})))
    ids = by_id(jobs)
    assert ids[1].start_time == 0.0
    assert ids[3].start_time == 100.0  # overtakes job 2 by tier
    assert ids[2].start_time == 200.0


def test_easy_mode_honours_priority_tier_too():
    jobs = [
        job(1, 0, 100, nodes=2, partition="low"),
        job(2, 1, 100, nodes=2, partition="low"),
        job(3, 2, 100, nodes=2, partition="high"),
    ]
    run(jobs, PreemptionConfig(partitions=partition_ladder({"low": 1, "high": 2})),
        backfill_mode="easy")  # fmt: skip
    assert [j.start_time for j in jobs] == [0.0, 200.0, 100.0]


# ── QOS preempt lists that are not a ladder ─────────────────────────────────


def _non_ladder() -> dict[str, QOSSpec]:
    """c preempts b, b preempts a, c does not preempt a. Acyclic: slurmdbd
    accepts it. "Can preempt" is then not transitive."""
    return {
        "a": QOSSpec("a"),
        "b": QOSSpec("b", preempt=frozenset({"a"})),
        "c": QOSSpec("c", preempt=frozenset({"b"})),
    }


def test_a_job_started_earlier_in_the_same_pass_can_be_preempted():
    """Pairwise queue order on a non-ladder graph: A (qos a, priority 100)
    before C (qos c, 50) by priority; C before B (qos b) because c preempts
    b; B before A because b preempts a. Sorting three jobs submitted in the
    order A, C, B gives [A, C, B]. The pass starts A and C on the two nodes;
    B cannot start and selects A — which started moments ago in this very
    pass. That used to raise KeyError (its run had no epoch yet)."""
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=_non_ladder())
    a = job(1, 0, 1000, qos="a", nice=-100)
    c = job(2, 0, 1000, qos="c", nice=-50)
    b = job(3, 0, 1000, qos="b")
    result = simulate([a, c, b], Cluster.homogeneous(2, 1), weights=FLAT, preemption=cfg)
    records = [(r.job_id, r.preemptor_id, r.start_time, r.release_time) for r in result.preemptions]
    assert records == [(1, 3, 0.0, 0.0)]
    assert [(run.start, run.end, run.outcome) for run in a.runs] == [
        (0.0, 0.0, "requeued"),
        (1000.0, 2000.0, "completed"),
    ]
    assert b.start_time == 0.0 and c.start_time == 0.0


@pytest.mark.parametrize(
    ("lists", "loop"),
    [
        ({"a": {"b"}, "b": {"a"}}, "a -> b -> a"),
        ({"a": {"b"}, "b": {"c"}, "c": {"a"}}, "a -> b -> c -> a"),
        ({"a": {"a"}}, "a -> a"),  # slurmdbd: "has an internal loop"
    ],
)
def test_qos_preempt_loops_are_refused_like_slurmdbd(lists, loop):
    """`_preemption_loop()`, as_mysql_qos.c: slurmdbd will not store a loop."""
    qos = {n: QOSSpec(n, preempt=frozenset(p)) for n, p in lists.items()}
    with pytest.raises(ValueError, match=loop):
        PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=qos)
    # Acyclic and non-ladder is fine, and so is a name the table lacks.
    PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=_non_ladder())
    PreemptionConfig(
        "preempt/qos",
        PreemptMode("REQUEUE"),
        qos={"a": QOSSpec("a", preempt=frozenset({"missing"}))},
    )


@pytest.mark.parametrize("seed", range(40))
def test_random_acyclic_qos_graphs_run_to_completion(seed):
    """Random DAGs of preempt lists, modes and priorities: every run ends,
    every job finishes or is cancelled, and no node is overcommitted
    (`Node.allocate` raises if one would be)."""
    rng = random.Random(seed)
    names = ["q0", "q1", "q2", "q3"]
    qos = {
        n: QOSSpec(n, preempt=frozenset(m for m in names[:i] if rng.random() < 0.5))
        for i, n in enumerate(names)
    }  # edges only point to earlier names: acyclic by construction
    mode = rng.choice(["REQUEUE", "CANCEL", "REQUEUE,WITHIN", "CANCEL,WITHIN"])
    cfg = PreemptionConfig(
        "preempt/qos",
        PreemptMode.parse(mode),
        qos=qos,
        exempt_time=rng.choice([0.0, 30.0]),
        youngest_first=rng.random() < 0.5,
    )
    work = [
        job(
            i,
            rng.uniform(0, 600),
            (d := rng.uniform(20, 400)),
            nodes=rng.choice([1, 1, 2, 3]),
            cpus_per_node=rng.choice([1, 2]),
            time_limit=d * rng.uniform(1, 3),
            qos=rng.choice(names),
            nice=rng.randint(-50, 50),
        )
        for i in range(1, 31)
    ]
    result = simulate(work, Cluster.homogeneous(3, 2), weights=FLAT, preemption=cfg)
    assert all(j.state in (State.COMPLETED, State.PREEMPTED) for j in result.jobs)


def test_preempt_exempt_time_of_minus_one_or_infinity_means_none():
    """slurm.conf(5): "A time of -1 disables the option, equivalent to 0"."""
    for value in (-1.0, float("inf")):
        cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), exempt_time=value)
        assert cfg.exempt_time == 0.0
    # A QOS value of -1 or INFINITE defers to the global one.
    ladder = qos_ladder({"lo": 0, "hi": 10})
    ladder["lo"] = replace(ladder["lo"], preempt_exempt_time=float("inf"))
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder, exempt_time=5.0)
    victim = job(1, 0, 100, qos="lo")
    victim.start_time = 10.0
    assert cfg.exempt_until(victim) == 15.0


# ── PreemptMode values slurmctld refuses or reads differently ───────────────


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("CANCEL,REQUEUE", "only one mode"),
        ("REQUEUE,CANCEL", "only one mode"),
        ("OFF,REQUEUE", "only one mode"),
        ("CLUSTER,CANCEL", "only one mode"),
        ("GANG,WITHIN", "GANG cannot be combined with WITHIN"),
        ("REQUEUE,GANG,PRIORITY", "GANG cannot be combined with PRIORITY"),
        ("ON", "gang scheduler"),
        ("on,GANG", "gang scheduler"),
    ],
)
def test_preempt_mode_values_slurmctld_refuses_are_refused(text, message):
    """`preempt_mode_num()` (slurm_protocol_defs.c L2746-L2803, SchedMD/slurm
    @9f9da53) returns NO_VAL16 for more than one base mode and for GANG with
    WITHIN or PRIORITY; read_config.c then refuses to start. `CANCEL,REQUEUE`
    used to run as REQUEUE, whichever came last. ON is SUSPEND's alias."""
    with pytest.raises(ValueError, match=message):
        PreemptMode.parse(text)


def test_cluster_is_off_and_flags_alone_are_accepted():
    assert PreemptMode.parse("CLUSTER") == PreemptMode("OFF")
    assert PreemptMode.parse("cluster").is_unset and PreemptMode.parse("OFF").is_unset
    within = PreemptMode.parse("WITHIN")
    assert (within.base, within.within, within.is_unset) == ("OFF", True, False)
    assert PreemptMode.parse("REQUEUE,GANG").gang  # parsed, then reported as not modelled


def test_preempt_type_none_refuses_within_and_priority_but_not_gang():
    """read_config.c L4711-L4719 masks only GANG for preempt/none, so a
    PreemptMode of WITHIN or PRIORITY there is "PreemptType and PreemptMode
    values incompatible". It used to load as preemption off, silently."""
    from schedlab.config import preemption_from_conf

    for mode in ("WITHIN", "PRIORITY", "OFF,WITHIN"):
        with pytest.raises(ValueError, match="incompatible"):
            preemption_from_conf({"preempttype": "preempt/none", "preemptmode": mode})
    cfg, warnings = preemption_from_conf({"preemptmode": "OFF,GANG"})
    assert not cfg.enabled and any("GANG" in w for w in warnings)


# ── Per-QOS and per-partition settings ──────────────────────────────────────


def test_a_qos_preempt_mode_of_flags_only_resolves_to_off_and_protects_its_jobs():
    """`preempt_p_get_mode()` (preempt_qos.c L66-L80) takes any non-zero QOS
    value before stripping GANG, PRIORITY and WITHIN. A bare WITHIN is
    non-zero, so the job's mode is OFF and it cannot be preempted
    (`_is_job_preempt_exempt()`, preempt.c L164). It used to fall back to the
    cluster's REQUEUE."""
    ladder = qos_ladder({"lo": 0, "hi": 10})
    ladder["lo"] = replace(ladder["lo"], preempt_mode=PreemptMode.parse("WITHIN"))
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder)
    jobs = _qos_jobs()
    assert cfg.mode_of(jobs[0]) == "OFF"
    result, _ = run(jobs, cfg)
    assert result.preemptions == [] and jobs[1].start_time == 1000.0


def test_a_qos_preempt_mode_of_off_means_cluster():
    """sacctmgr(1): OFF is the same stored 0 as CLUSTER, so it defers."""
    ladder = qos_ladder({"lo": 0, "hi": 10})
    ladder["lo"] = replace(ladder["lo"], preempt_mode=PreemptMode("OFF"))
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder)
    jobs = _qos_jobs()
    result, _ = run(jobs, cfg)
    assert [(r.job_id, r.mode) for r in result.preemptions] == [(1, "REQUEUE")]
    assert jobs[0].state is State.COMPLETED and jobs[0].requeue_count == 1


def test_a_qos_preempt_mode_overrides_the_cluster_mode():
    ladder = qos_ladder({"lo": 0, "hi": 10})
    ladder["lo"] = replace(ladder["lo"], preempt_mode=PreemptMode("CANCEL"))
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder)
    jobs = _qos_jobs()
    result, _ = run(jobs, cfg)
    assert [(r.job_id, r.mode) for r in result.preemptions] == [(1, "CANCEL")]
    assert jobs[0].state is State.PREEMPTED


@pytest.mark.parametrize(("cluster", "partition"), [("REQUEUE", "CANCEL"), ("CANCEL", "REQUEUE")])
def test_a_partition_preempt_mode_overrides_the_cluster_mode(cluster, partition):
    """`preempt_p_get_mode()` in preempt_partition_prio.c: the preemptee's
    partition value when set, else the cluster's."""
    cfg = tiers(mode=cluster)
    cfg = replace(
        cfg,
        partitions={
            **cfg.partitions,
            "low": PartitionSpec("low", 1, preempt_mode=PreemptMode.parse(partition)),
        },
    )
    jobs = victim_and_preemptor()
    result, _ = run(jobs, cfg)
    assert [r.mode for r in result.preemptions] == [partition]
    assert jobs[0].state is (State.PREEMPTED if partition == "CANCEL" else State.COMPLETED)


def test_a_finite_qos_exempt_time_overrides_the_global_one_either_way():
    """acct_policy_get_preemptable_time(): the QOS value first, then the
    global one. Exempt until 100: the next main pass is the periodic one at
    120 (as in test_preempt_exempt_time_protects_young_jobs)."""
    ladder = qos_ladder({"lo": 0, "hi": 10})
    longer = dict(ladder, lo=replace(ladder["lo"], preempt_exempt_time=100.0))
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=longer)
    result, _ = run(_qos_jobs(), cfg)
    assert result.preemptions[0].preempt_time == 120.0
    # A QOS value of 0 overrides a global 100: preempted on arrival.
    shorter = dict(ladder, lo=replace(ladder["lo"], preempt_exempt_time=0.0))
    cfg = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=shorter, exempt_time=100.0)
    result, _ = run(_qos_jobs(), cfg)
    assert result.preemptions[0].preempt_time == 10.0


def test_a_qos_exempt_time_applies_under_partition_prio_too():
    """acct_policy_get_preemptable_time() (acct_policy.c L5203-L5226) never
    reads PreemptType; `_is_job_preempt_exempt_internal()` calls it for
    every plugin. The model read the QOS value under preempt/qos only."""
    cfg = replace(
        tiers(), qos={"normal": QOSSpec("normal", preempt_exempt_time=100.0)}
    )
    jobs = victim_and_preemptor()
    jobs[0].qos = "normal"
    result, _ = run(jobs, cfg)
    assert result.preemptions[0].preempt_time == 120.0 and jobs[1].start_time == 120.0


# ── The PRIORITY flag ───────────────────────────────────────────────────────


def test_priority_flag_under_partition_prio_refuses_a_lower_priority_preemptor():
    """preempt_partition_prio.c: with PRIORITY the preemptor's job priority
    must not be lower than the preemptee's. The victim has priority 100
    (nice -100), the higher-tier job 1."""

    def scenario(mode: str) -> tuple[list[int], float | None]:
        jobs = victim_and_preemptor()
        jobs[0].nice = -100
        result, _ = run(jobs, tiers(mode=mode))
        return [r.job_id for r in result.preemptions], jobs[1].start_time

    assert scenario("REQUEUE") == ([1], 10.0)
    assert scenario("REQUEUE,PRIORITY") == ([], 1000.0)


def test_priority_flag_under_partition_prio_in_the_queue_order_test():
    """`preempt_p_job_preempt_check()`: with PRIORITY a higher-tier job of
    lower priority no longer "can preempt". (The queue order is unchanged
    here: the PriorityTier rule that follows gives the same answer.)"""
    low, high = job(1, 0, 10, partition="low"), job(2, 5, 10, partition="high")
    low.priority, high.priority = 100.0, 1.0
    assert tiers().sorts_before(high, low)
    assert not tiers(mode="REQUEUE,PRIORITY").sorts_before(high, low)
    assert queue_order([low, high], tiers(mode="REQUEUE,PRIORITY")) == [high, low]


def test_priority_flag_under_qos_refuses_preemption_and_changes_the_queue_order():
    """preempt_qos.c: with PRIORITY (cluster-wide or on the preemptor's QOS)
    the preemptor's job priority must be higher. Queue order's "can
    preempt" rule is the same test, so the order flips back to priority."""
    ladder = qos_ladder({"lo": 0, "hi": 10})

    def scenario(mode: str, qos: dict[str, QOSSpec] = ladder) -> list[int]:
        jobs = _qos_jobs()
        jobs[0].nice = -100
        result, _ = run(jobs, PreemptionConfig("preempt/qos", PreemptMode.parse(mode), qos=qos))
        return [r.job_id for r in result.preemptions]

    assert scenario("REQUEUE") == [1]
    assert scenario("REQUEUE,PRIORITY") == []
    flag = PreemptMode.parse("REQUEUE,PRIORITY")
    on_qos = dict(ladder, hi=replace(ladder["hi"], preempt_mode=flag))
    assert scenario("REQUEUE", on_qos) == []

    lo, hi = job(1, 0, 10, qos="lo"), job(2, 5, 10, qos="hi")
    lo.priority, hi.priority = 100.0, 1.0
    plain = PreemptionConfig("preempt/qos", PreemptMode("REQUEUE"), qos=ladder)
    flagged = PreemptionConfig("preempt/qos", PreemptMode.parse("REQUEUE,PRIORITY"), qos=ladder)
    assert queue_order([lo, hi], plain) == [hi, lo]
    assert queue_order([lo, hi], flagged) == [lo, hi]


# ── Requeue recomputes priority at once ─────────────────────────────────────


def test_a_requeued_job_gets_its_new_priority_at_once_not_at_the_next_tick():
    """batch_requeue_fini() resets accrual, and the priority is recomputed
    then. One 1-CPU node; age weight 1000 with PriorityMaxAge 1000 s; ticks
    every 300 s.

    J0 runs 0-600. V (low) waits from 0 and starts at 600 with priority 600
    (the 600 tick). P (high) selects V at 700: grace 0, eligible 821. P runs
    700-850. C (low, nice -100) arrives at 710 with priority 100. At 850,
    before the 900 tick, V's recomputed priority is 1 (age 0 from its new
    begin time), so C starts first; with the stale 600, V would.
    """
    weights = weights_only(age=1000.0, max_age=1000.0)
    jobs = [
        job(1, 0, 600, partition="low"),
        job(2, 0, 1000, partition="low"),
        job(3, 700, 150, partition="high"),
        job(4, 710, 100, partition="low", nice=-100),
    ]
    simulate(jobs, Cluster.homogeneous(1, 1), weights=weights, preemption=tiers())
    v, c = jobs[1], jobs[3]
    assert v.runs[0].start == 600.0
    assert v.runs[0].preempt_time == 700.0 and v.eligible_time == 821.0
    assert c.start_time == 850.0 and v.start_time == 950.0
    # Its final dispatch came after the 900 tick: age 900 - 821 = 79 s.
    assert v.dispatch_priority == 79.0


# ── Jobs that share an id ───────────────────────────────────────────────────


def test_victims_that_share_an_id_are_still_preempted():
    """`select_preemptees` kept its bookkeeping by job id, so two candidates
    with one id overwrote each other's entry and nothing was preempted: the
    high job waited 990 s. Ids need not be unique (sacct array tasks, a k8s
    CSV), so it is keyed by the job now."""

    def scenario(first: int, second: int) -> list[tuple[float | None, State]]:
        jobs = [
            job(first, 0, 1000, partition="low"),
            job(second, 0, 1000, partition="low"),
            job(3, 10, 100, partition="high"),
        ]
        result, _ = run(jobs, tiers(mode="CANCEL"))
        assert len(result.preemptions) == 1
        return [(j.start_time, j.state) for j in jobs]

    assert scenario(1, 2) == scenario(7, 7)
    assert scenario(7, 7)[2] == (10.0, State.COMPLETED)
