"""A1 and A3: usage accrues incrementally on PriorityCalcPeriod ticks.

Pins the three properties the 0.1.0 dispatch-time charge violated:

(a) nothing on the scheduling or priority path reads `job.duration`;
(b) usage grows with elapsed time, never ahead of it;
(c) the calc period is a real knob — it changes which job goes first;

plus decay's event-count independence and the reset periods.
"""

from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

import schedlab.simulate as sim
from schedlab.accounting import PriorityEngine
from schedlab.model import Cluster, Job
from schedlab.params import SchedulerParameters
from schedlab.priority import FairshareTree, PriorityWeights
from schedlab.simulate import simulate
from schedlab.trace import WorkloadProfile, generate
from tests._factory import job, weights_only

# ── (a) The scheduler never reads the true runtime ──────────────────────────


class _GuardedJob(Job):
    """A job whose `duration` may be read by exactly one function.

    `_world_end_time()` in simulate.py plays the job finishing. Any other
    reader — priority, fairshare accrual, EASY, the main pass, the backfill
    planner — raises. Reads are counted so the test can also prove the world
    looks exactly once per job.
    """

    reads: ClassVar[dict[int, int]] = {}
    _WORLD = str(Path(sim.__file__).resolve())

    def __getattribute__(self, name: str) -> Any:
        if name == "duration":
            code = sys._getframe(1).f_code
            if code.co_name != "_world_end_time" or str(Path(code.co_filename).resolve()) != (
                _GuardedJob._WORLD
            ):
                raise AssertionError(
                    f"job.duration read by {code.co_filename}:{code.co_name}"
                )
            job_id = object.__getattribute__(self, "job_id")
            _GuardedJob.reads[job_id] = _GuardedJob.reads.get(job_id, 0) + 1
        return object.__getattribute__(self, name)


def _guarded(jobs: list[Job]) -> list[Job]:
    return [
        _GuardedJob(**{f.name: getattr(j, f.name) for f in fields(Job) if f.init}) for j in jobs
    ]


@pytest.mark.parametrize("mode", ["easy", "conservative"])
@pytest.mark.parametrize("algorithm", ["fair_tree", "classic"])
@pytest.mark.parametrize("calc_period", [300.0, 0.0])
@pytest.mark.parametrize("backfill", [True, False])
def test_scheduling_and_priority_never_read_true_runtime(mode, algorithm, calc_period, backfill):
    plain = generate(WorkloadProfile(job_count=80), seed=5)
    jobs = _guarded(plain)
    weights = replace(PriorityWeights(), fairshare_algorithm=algorithm, calc_period=calc_period)
    _GuardedJob.reads = {}

    result = simulate(
        jobs,
        Cluster.homogeneous(16, 8, 2),
        weights=weights,
        fairshare=weights.make_fairshare(j.account for j in plain),
        backfill=backfill,
        backfill_mode=mode,
    )

    assert len(result.jobs) == len(plain)
    # The world read each runtime once, to schedule the completion event.
    assert _GuardedJob.reads == {j.job_id: 1 for j in plain}


def test_end_time_is_unknown_until_the_job_finishes(monkeypatch):
    """0.1.0 stamped `end_time` at dispatch, parking the true end on the job."""
    seen: list[float | None] = []
    real = sim._world_end_time

    def spy(j: Job, start: float) -> float:
        seen.append(j.end_time)
        return real(j, start)

    monkeypatch.setattr(sim, "_world_end_time", spy)
    jobs = generate(WorkloadProfile(job_count=40), seed=2)
    simulate(jobs, Cluster.homogeneous(16, 8, 2))
    assert len(seen) == len(jobs)
    assert all(e is None for e in seen)
    assert all(j.end_time is not None for j in jobs)


# ── (b) Usage accrues with elapsed time ─────────────────────────────────────


def _snapshots(result: sim.SimulationResult, account: str) -> dict[float, float]:
    return {s.time: s.usage.get(account, 0.0) for s in result.fairshare_snapshots}


def test_usage_accrues_per_calc_period_not_at_dispatch():
    # 2 CPUs for 1000 s, no decay, 300 s calc period.
    weights = replace(PriorityWeights(), decay_half_life=0.0, calc_period=300.0)
    tree = weights.make_fairshare(["a"])
    work = [job(1, 0, 1000, cpus_per_node=2, account="a")]
    result = simulate(work, Cluster.homogeneous(1, 4), weights=weights, fairshare=tree)

    usage = _snapshots(result, "a")
    # 0.1.0 would have shown the full 2000 CPU-s at t=0.
    assert usage[0.0] == 0.0
    assert usage[300.0] == pytest.approx(600.0)
    assert usage[600.0] == pytest.approx(1200.0)
    assert usage[900.0] == pytest.approx(1800.0)
    # The last 100 s were charged at job end (priority_p_job_end), not rounded
    # up to a whole period.
    assert tree.usage["a"] == pytest.approx(2000.0)


def test_accrued_usage_is_decayed_over_its_own_span():
    """`_apply_new_usage()`: run_decay = run_delta * pow(decay_factor, run_delta)."""
    half_life = 3600.0
    weights = replace(PriorityWeights(), decay_half_life=half_life, calc_period=300.0)
    tree = weights.make_fairshare(["a"])
    result = simulate(
        [job(1, 0, 1000, account="a")], Cluster.homogeneous(1, 1), weights=weights, fairshare=tree
    )
    m = 0.5 ** (300.0 / half_life)
    usage = _snapshots(result, "a")
    assert usage[300.0] == pytest.approx(300.0 * m)
    assert usage[600.0] == pytest.approx(300.0 * m * m + 300.0 * m)


@pytest.mark.parametrize("calc_period", [60.0, 300.0, 3600.0, 0.0])
@pytest.mark.parametrize("mode", ["easy", "conservative"])
def test_total_usage_equals_cpu_seconds_consumed_for_any_calc_period(calc_period, mode):
    """With decay off, accrual telescopes: the total is exactly what ran."""
    jobs = generate(WorkloadProfile(job_count=60), seed=9)
    consumed: dict[str, float] = {}
    for j in jobs:
        consumed[j.account] = consumed.get(j.account, 0.0) + j.total_cpus * j.duration

    weights = replace(PriorityWeights(), decay_half_life=0.0, calc_period=calc_period)
    tree = weights.make_fairshare(j.account for j in jobs)
    cluster = Cluster.homogeneous(16, 8, 2)
    simulate(jobs, cluster, weights=weights, fairshare=tree, backfill_mode=mode)

    assert tree.usage == pytest.approx(consumed)


def test_usage_is_proportional_to_elapsed_time():
    weights = replace(PriorityWeights(), decay_half_life=0.0, calc_period=300.0)
    tree = weights.make_fairshare(["a", "b"])
    work = [job(1, 0, 600, account="a"), job(2, 0, 1800, account="b")]
    result = simulate(work, Cluster.homogeneous(2, 1), weights=weights, fairshare=tree)
    usage_a, usage_b = _snapshots(result, "a"), _snapshots(result, "b")
    # While both run, they accrue at the same rate...
    assert usage_a[300.0] == pytest.approx(usage_b[300.0])
    # ...and once a stops, only b grows.
    assert usage_a[1200.0] == pytest.approx(600.0)
    assert usage_b[1200.0] == pytest.approx(1200.0)
    assert tree.usage["b"] == pytest.approx(3 * tree.usage["a"])


# ── (c) The calc period changes who goes first ──────────────────────────────


def _race(calc_period: float, algorithm: str, mode: str, first_ends: float = 600.0) -> list[int]:
    """One CPU. Account a runs job 1; a's job 2 and b's job 3 queue behind it.

    Only fairshare carries weight. Whether job 3 overtakes job 2 when job 1
    finishes depends purely on whether a's usage has become visible yet.
    """
    weights = replace(
        weights_only(fairshare=1_000.0),
        calc_period=calc_period,
        fairshare_algorithm=algorithm,  # type: ignore[arg-type]
    )
    work = [
        job(1, 0, first_ends, account="a"),
        job(2, 10, 100, account="a"),
        job(3, 20, 100, account="b"),
    ]
    simulate(
        work,
        Cluster.homogeneous(1, 1),
        weights=weights,
        fairshare=weights.make_fairshare(["a", "b"]),
        backfill_mode=mode,  # type: ignore[arg-type]
    )
    return [j.job_id for j in sorted(work, key=lambda j: j.start_time or 0.0)]


@pytest.mark.parametrize("mode", ["easy", "conservative"])
@pytest.mark.parametrize("algorithm", ["fair_tree", "classic"])
def test_calc_period_granularity_changes_priority_order(algorithm, mode):
    # 1-minute ticks: a's usage is visible long before t=600, so b goes first.
    assert _race(60.0, algorithm, mode) == [1, 3, 2]
    # 1-hour ticks: no tick since t=0, both accounts still look unused, and
    # the tie falls to submit order.
    assert _race(3600.0, algorithm, mode) == [1, 2, 3]


@pytest.mark.parametrize("mode", ["easy", "conservative"])
def test_classic_sees_usage_one_period_later_than_fair_tree(mode):
    """Ticks every 300 s; job 1 (account a) runs from t=0.

    Fair Tree accrues usage, then recomputes factors, then priorities, all
    in one tick, so the 300 s tick already sees a's first 300 CPU-seconds.
    Classic's step 2 marks each user's effective usage NO_VAL, and the same
    tick's priority loop recomputes it from the account's value at the
    *start* of the tick (`_decay_thread()`, `_get_fairshare_priority()`,
    `_set_usage_efctv()`): before that tick's accrual. So:

    * job 1 ends at 400: Fair Tree's 300 s tick saw usage; classic's saw the
      usage at the start of that tick, none, and job 2 keeps submit order;
    * job 1 ends at 700: classic's 600 s tick starts from the 300 CPU-s
      accrued at 300 (a: 2^(-1/0.5) = 0.25, b: 1), so b's job goes first,
      as under Fair Tree. Classic is one period behind, not two.
    """
    assert _race(300.0, "fair_tree", mode, first_ends=400.0) == [1, 3, 2]
    assert _race(300.0, "classic", mode, first_ends=400.0) == [1, 2, 3]
    assert _race(300.0, "fair_tree", mode, first_ends=700.0) == [1, 3, 2]
    assert _race(300.0, "classic", mode, first_ends=700.0) == [1, 3, 2]
    # The idealised no-lag mode sees it under either algorithm.
    assert _race(0.0, "classic", mode, first_ends=400.0) == [1, 3, 2]


def test_idealised_mode_refreshes_before_a_timer_only_backfill_cycle():
    """`--calc-period 0` promises no lag, so a backfill cycle on a timer, with
    no submit or completion at that instant, must sort by priorities as of
    that instant. It used to tick on submits and completions only.

    Age (weight 1000, PriorityMaxAge 1000 s) and QOS (590) only; the main
    scheduler is off (`sched_interval=-1`), so only backfill starts jobs, on
    the 30 s grid. Job 1 holds the one CPU until 1000. Job 2 (submit 0) is
    at age 1 by then: 1000. Job 3 (submit 600, QOS) has 1000 × 0.4 + 590 =
    990 at the completion at t=1000, but 420 + 590 = 1010 at the backfill
    cycle at t=1020, the first decision after the CPU frees. So job 3 starts
    at 1020. With the refresh at t=1000 only, job 2 went first.
    """
    weights = replace(weights_only(age=1000.0, qos=590.0), max_age=1000.0, calc_period=0.0)
    work = [job(1, 0, 1000), job(2, 0, 100), job(3, 600, 100, qos_factor=1.0)]
    result = simulate(
        work,
        Cluster.homogeneous(1, 1),
        weights=weights,
        sched_params=SchedulerParameters(sched_interval=-1),
    )
    assert [j.start_time for j in work] == [0.0, 1140.0, 1020.0]
    assert work[2].dispatch_priority == 1010
    assert result.backfilled_ids == [1, 3, 2]


def test_classic_tick_uses_the_factor_from_the_start_of_the_same_tick():
    """Pinned at the engine: a tick's priorities and its snapshot use one factor."""
    weights = replace(
        weights_only(fairshare=1_000.0), calc_period=300.0, fairshare_algorithm="classic"
    )
    tree = weights.make_fairshare(["a", "b"])
    engine = PriorityEngine(Cluster.homogeneous(1, 1), weights, tree)
    runner, waiter = job(1, 0, 1000, account="a"), job(2, 0, 10, account="a")
    runner.start_time = 0.0
    engine.job_started(runner, 0.0)
    engine.tick(0.0, [waiter], [runner])
    engine.tick(300.0, [waiter], [runner])
    # Start of the 300 s tick: nothing accrued yet, so a still looks unused.
    assert waiter.priority == 1_000
    engine.tick(600.0, [waiter], [runner])
    # Start of the 600 s tick: a has all the usage so far, U=1, S=0.5.
    assert waiter.priority == int(1_000 * 2 ** (-1 / 0.5))
    assert engine.snapshots[-1].factors["a"] == pytest.approx(0.25)


def test_priority_is_a_snapshot_between_ticks():
    """Age grows continuously, but the priority the scheduler reads does not."""
    weights = replace(weights_only(age=1_000.0), max_age=2_000.0, calc_period=250.0)
    blocker, waiting = job(1, 0, 1000), job(2, 0, 10)
    simulate([blocker, waiting], Cluster.homogeneous(1, 1), weights=weights)
    # Dispatched at t=1000, which is a tick: age factor 1000/2000.
    assert waiting.dispatch_priority == pytest.approx(500.0)

    blocker, waiting = job(1, 0, 1200), job(2, 0, 10)
    simulate([blocker, waiting], Cluster.homogeneous(1, 1), weights=weights)
    # Dispatched at t=1200, but the last tick was t=1000: still 500, not 600.
    assert waiting.dispatch_priority == pytest.approx(500.0)


# ── A3: decay depends on simulated time only ────────────────────────────────


def _half_life_run(flood: int) -> dict[float, float]:
    """Account a runs 600 s then stops; b floods the other node with events."""
    weights = replace(PriorityWeights(), decay_half_life=3600.0, calc_period=300.0)
    tree = weights.make_fairshare(["a", "b"])
    work = [job(1, 0, 600, account="a")]
    work += [job(100 + k, 600 + 17.0 * k, 5, account="b") for k in range(flood)]
    # A late arrival keeps the simulation (and its ticks) going past 4500 s.
    work.append(job(99, 5000, 5, account="b"))
    result = simulate(work, Cluster.homogeneous(2, 1), weights=weights, fairshare=tree)
    return _snapshots(result, "a")


@pytest.mark.parametrize("flood", [0, 1, 200])
def test_one_half_life_halves_usage_regardless_of_event_count(flood):
    usage = _half_life_run(flood)
    assert usage[900.0] > 0
    assert usage[900.0 + 3600.0] / usage[900.0] == pytest.approx(0.5, rel=1e-12)


def test_decay_is_identical_with_and_without_scheduling_events():
    quiet, busy = _half_life_run(0), _half_life_run(200)
    for t in (900.0, 2700.0, 4500.0):
        assert busy[t] == pytest.approx(quiet[t], rel=1e-12)


def test_half_life_zero_disables_decay():
    tree = FairshareTree(shares={"a": 1.0}, half_life=0.0)
    tree.charge("a", 100.0)
    tree.decay_by(1e9)
    assert tree.usage["a"] == 100.0


# ── PriorityUsageResetPeriod ────────────────────────────────────────────────


def _reset_run(period: str, offset: float, duration: float, seed_usage: float = 0.0):
    weights = replace(
        PriorityWeights(),
        decay_half_life=0.0,
        calc_period=300.0,
        usage_reset_period=period,  # type: ignore[arg-type]
    )
    tree = weights.make_fairshare(["a"])
    if seed_usage:
        tree.charge("a", seed_usage)
    result = simulate(
        [job(1, 0, duration, account="a")],
        Cluster.homogeneous(1, 1),
        weights=weights,
        fairshare=tree,
        calendar_offset=offset,
    )
    return _snapshots(result, "a"), tree


def test_daily_reset_fires_at_the_first_tick_after_midnight():
    # Midnight falls at simulated t=1000; ticks at 900 and 1200.
    usage, _ = _reset_run("DAILY", offset=86_400.0 - 1000.0, duration=2000.0)
    assert usage[900.0] == pytest.approx(900.0)
    # Reset, then the whole 900-1200 period accrues: Slurm resets before it
    # applies the period's new usage (`_decay_thread()` order).
    assert usage[1200.0] == pytest.approx(300.0)
    assert usage[1500.0] == pytest.approx(600.0)


def test_weekly_reset_uses_a_seven_day_cycle():
    usage, _ = _reset_run("WEEKLY", offset=7 * 86_400.0 - 1000.0, duration=2000.0)
    assert usage[1200.0] == pytest.approx(300.0)
    usage, _ = _reset_run("WEEKLY", offset=86_400.0 - 1000.0, duration=2000.0)
    assert usage[1200.0] == pytest.approx(1200.0)  # a day boundary is not a week


def test_reset_now_clears_preexisting_usage_once():
    usage, _ = _reset_run("NOW", offset=0.0, duration=600.0, seed_usage=1e6)
    assert usage[0.0] == 0.0
    assert usage[300.0] == pytest.approx(300.0)


@pytest.mark.parametrize("period", ["MONTHLY", "QUARTERLY", "YEARLY"])
def test_calendar_month_resets_are_refused_not_faked(period):
    weights = replace(PriorityWeights(), usage_reset_period=period)
    with pytest.raises(ValueError, match="calendar"):
        PriorityEngine(Cluster.homogeneous(1, 1), weights, weights.make_fairshare(["a"]))
