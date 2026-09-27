"""A2: the Slurm-like conservative mode and its SchedulerParameters bounds.

Each scenario is small enough to work out by hand; the expected times in the
assertions were derived from the plan semantics in backfill.py, not read back
from a run.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest

import schedlab.simulate as sim
from schedlab import k8strace
from schedlab.backfill import BackfillCycle, NodeSpace
from schedlab.metrics import compute
from schedlab.model import Cluster, Job
from schedlab.params import SchedulerParameters
from schedlab.simulate import simulate
from schedlab.trace import WorkloadProfile, generate
from tests._factory import by_id, clone, job, started, weights_only

SIZE = weights_only(jobsize=10_000.0)  # bigger job = higher priority, nothing else


def params(**kw: float) -> SchedulerParameters:
    return replace(SchedulerParameters(), **kw)  # type: ignore[arg-type]


# ── Cadence: backfill runs on bf_interval, not on every event ──────────────


def _hole(long_first: float = 1000.0) -> list[Job]:
    return [
        job(1, submit=0, duration=long_first, nodes=3),
        job(2, submit=1, duration=100, nodes=4),  # blocked: needs all 4
        job(3, submit=2, duration=10, nodes=1),  # fits the free node
    ]


def test_backfill_waits_for_the_next_bf_interval_tick():
    jobs = _hole()
    result = simulate(jobs, Cluster.homogeneous(4, 1))
    assert jobs[2].start_time == 30.0  # first bf tick after it arrived
    assert 3 in result.backfilled_ids

    jobs = _hole()
    simulate(jobs, Cluster.homogeneous(4, 1), sched_params=params(bf_interval=120))
    assert jobs[2].start_time == 120.0

    jobs = _hole()
    simulate(jobs, Cluster.homogeneous(4, 1), backfill_mode="easy")
    assert jobs[2].start_time == 2.0  # EASY decides on the arrival event


def test_sched_builtin_never_backfills():
    jobs = _hole(long_first=100.0)
    result = simulate(jobs, Cluster.homogeneous(4, 1), backfill=False)
    assert jobs[2].start_time == 200.0  # strictly behind job 2
    assert result.backfilled_ids == []
    assert result.backfill_stats.count == 0


def test_bf_interval_minus_one_disables_the_backfill_loop():
    jobs = _hole(long_first=100.0)
    result = simulate(jobs, Cluster.homogeneous(4, 1), sched_params=params(bf_interval=-1))
    assert jobs[2].start_time == 200.0
    assert result.backfill_stats.count == 0


# ── Conservative: every planned job is protected, not just the first ───────


def _two_blocked() -> list[Job]:
    return [
        job(1, 0, 120, nodes=2),  # R1
        job(2, 0, 60, nodes=1),  # R2
        job(3, 1, 120, nodes=3),  # A: first blocked
        job(4, 1, 60, nodes=2),  # B: second blocked
        job(5, 1, 100, nodes=1),  # C: fits now, ends before A's shadow time
    ]


def test_conservative_protects_the_second_blocked_job():
    """EASY protects only A, so C may take the node B needs at t=60."""
    easy, cons = _two_blocked(), _two_blocked()
    simulate(easy, Cluster.homogeneous(4, 1), weights=SIZE, backfill_mode="easy")
    simulate(cons, Cluster.homogeneous(4, 1), weights=SIZE)
    easy_ids, cons_ids = by_id(easy), by_id(cons)

    # EASY: C jumps in at t=1, B is stuck behind it and then behind A.
    assert easy_ids[5].start_time == 1.0
    assert easy_ids[4].start_time == 240.0
    # Conservative: B is planned for t=60 (R2's end) and C may not touch its
    # nodes, so B starts on time and C waits.
    assert cons_ids[4].start_time == 60.0
    assert cons_ids[5].start_time == 120.0
    # A is protected in both.
    assert easy_ids[3].start_time == 120.0
    assert cons_ids[3].start_time == 120.0


def test_per_node_plan_is_not_fooled_by_scattered_free_cpus():
    """2 nodes x 4 CPUs, each half-used until 100 and half until 300.

    At t=100 four CPUs are free in aggregate but no single node has four, so
    A (one 4-CPU node) cannot start before 300. EASY's aggregate shadow time
    says 100 — so it forbids B from running past 100 and B waits until 300.
    The per-node plan knows A starts at 300 and lets B run 120-270 on node 0.
    """

    def work() -> list[Job]:
        return [
            job(1, 0, 100, cpus_per_node=2),  # node0
            job(2, 0, 300, cpus_per_node=2),  # node0
            job(3, 0, 100, cpus_per_node=2),  # node1
            job(4, 0, 300, cpus_per_node=2),  # node1
            job(5, 1, 100, cpus_per_node=4),  # A
            job(6, 2, 150, cpus_per_node=2),  # B
        ]

    easy, cons = work(), work()
    easy_sched = sim.EasyScheduler(Cluster.homogeneous(2, 4))
    simulate(easy, easy_sched.cluster, weights=SIZE, backfill_mode="easy")
    result = simulate(cons, Cluster.homogeneous(2, 4), weights=SIZE)
    e, c = by_id(easy), by_id(cons)

    assert e[6].start_time == 300.0
    assert c[6].start_time == 120.0  # first bf tick after the t=100 completions
    assert e[5].start_time == c[5].start_time == 300.0  # A not delayed by B
    assert result.planned_starts[5] == 300.0


def test_aggregate_shadow_time_is_optimistic_where_the_plan_is_not():
    cluster = Cluster.homogeneous(2, 4)
    running = [job(1, 0, 300, cpus_per_node=2), job(2, 0, 300, cpus_per_node=2)]
    for r in running:
        r.start_time = 0.0
    cluster.allocate(running[0], [0])
    cluster.allocate(running[1], [1])
    wide = job(3, 0, 100, cpus_per_node=4)

    # 4 CPUs are free in total, so EASY thinks it can start now.
    assert sim.EasyScheduler(cluster)._shadow_time(wide, running, now=0.0) == 0.0
    # Per node, it cannot start before either node empties.
    found = NodeSpace(cluster, running, resolution=1.0).earliest_start(wide, 0.0, 1e9)
    assert found == (300.0, [0])


# ── Bounds: bf_max_job_test, bf_window, bf_resolution ──────────────────────


def _deep_queue() -> list[Job]:
    return [
        job(1, 0, 1000, nodes=3),  # R
        job(2, 1, 100, nodes=4),  # A (blocked)
        job(3, 2, 100, nodes=2),  # X1..X3: blocked, only one node free
        job(4, 3, 100, nodes=2),
        job(5, 4, 100, nodes=2),
        job(6, 5, 50, nodes=1),  # D: could backfill, if anyone looked
    ]


def test_bf_max_job_test_leaves_the_tail_of_the_queue_unexamined():
    unlimited = _deep_queue()
    simulate(unlimited, Cluster.homogeneous(4, 1), weights=SIZE)
    assert by_id(unlimited)[6].start_time == 30.0

    limited = _deep_queue()
    result = simulate(
        limited, Cluster.homogeneous(4, 1), weights=SIZE, sched_params=params(bf_max_job_test=3)
    )
    assert started(by_id(limited)[6]) >= 1000.0
    stats = result.backfill_stats
    assert stats.max_tested == 3
    assert stats.hit_max_job_test > 0
    assert compute(result, Cluster.homogeneous(4, 1)).bf_cycles_hit_max_job_test > 0


def test_a_job_planned_beyond_bf_window_gets_no_reservation():
    def work() -> list[Job]:
        return [
            job(1, 0, 1000, nodes=1),  # R
            job(2, 1, 100, nodes=2),  # A: earliest start 1000
            job(3, 2, 5000, nodes=1),  # B: long, fits the free node now
        ]

    wide = work()
    simulate(wide, Cluster.homogeneous(2, 1), weights=SIZE)
    assert by_id(wide)[2].start_time == 1000.0
    assert by_id(wide)[3].start_time == 1100.0

    narrow = work()
    result = simulate(
        narrow, Cluster.homogeneous(2, 1), weights=SIZE, sched_params=params(bf_window=5 * 60)
    )
    # A is 970 s out at the first cycle, beyond a 5-minute window: unprotected.
    assert by_id(narrow)[3].start_time == 30.0
    assert by_id(narrow)[2].start_time == 5030.0
    assert any(c.beyond_window for c in result.backfill_stats.cycles)


def test_bf_resolution_widens_reservations_and_costs_backfill():
    """A is planned at t=100. At resolution 60 its reservation is quantised
    outward to [60, 240), so B's 30-70 run now collides with it."""

    def work() -> list[Job]:
        return [job(1, 0, 100, nodes=1), job(2, 1, 100, nodes=2), job(3, 2, 40, nodes=1)]

    fine = work()
    simulate(fine, Cluster.homogeneous(2, 1), weights=SIZE, sched_params=params(bf_resolution=1))
    coarse = work()
    simulate(coarse, Cluster.homogeneous(2, 1), weights=SIZE, sched_params=params(bf_resolution=60))

    assert by_id(fine)[3].start_time == 30.0
    assert by_id(coarse)[3].start_time == 200.0
    assert by_id(fine)[2].start_time == by_id(coarse)[2].start_time == 100.0


def test_bf_max_job_start_caps_starts_per_cycle():
    def work() -> list[Job]:
        return [
            job(1, 0, 1000, nodes=2),
            job(2, 1, 100, nodes=4),
            job(3, 2, 20, nodes=1),
            job(4, 3, 20, nodes=1),
            job(5, 4, 20, nodes=1),
        ]

    free = work()
    simulate(free, Cluster.homogeneous(4, 1), weights=SIZE)
    assert [by_id(free)[i].start_time for i in (3, 4, 5)] == [30.0, 30.0, 60.0]

    capped = work()
    result = simulate(
        capped, Cluster.homogeneous(4, 1), weights=SIZE, sched_params=params(bf_max_job_start=1)
    )
    assert [by_id(capped)[i].start_time for i in (3, 4, 5)] == [30.0, 60.0, 90.0]
    assert result.backfill_stats.hit_max_job_start > 0


def test_bf_max_job_user_stops_one_user_flooding_the_test_budget():
    def work() -> list[Job]:
        return [
            job(1, 0, 1000, nodes=3, user="r"),
            job(2, 1, 100, nodes=4, user="u1"),
            job(3, 2, 100, nodes=2, user="u1"),
            job(4, 3, 100, nodes=2, user="u1"),
            job(5, 4, 50, nodes=1, user="u2"),
        ]

    flooded = work()
    cluster = Cluster.homogeneous(4, 1)
    simulate(flooded, cluster, weights=SIZE, sched_params=params(bf_max_job_test=2))
    assert started(by_id(flooded)[5]) >= 1000.0

    capped = work()
    result = simulate(
        capped,
        Cluster.homogeneous(4, 1),
        weights=SIZE,
        sched_params=params(bf_max_job_test=2, bf_max_job_user=1),
    )
    # u1's extra jobs are skipped without spending the test budget.
    assert by_id(capped)[5].start_time == 30.0
    assert result.backfill_stats.cycles[0].skipped_by_limits == 2


# ── Main scheduler: default_queue_depth, sched_interval, partitions ─────────


def test_event_passes_test_default_queue_depth_plus_one_jobs():
    """`_schedule()` breaks on `job_depth++ > def_job_limit` (job_scheduler.c
    L1578): the post-increment lets depth + 1 jobs through."""

    def work() -> list[Job]:
        return [job(0, 0, 1000)] + [job(i, 10, 1000) for i in range(1, 6)]

    def starts(depth: int) -> list[float | None]:
        jobs = work()
        simulate(
            jobs,
            Cluster.homogeneous(8, 1),
            weights=weights_only(),
            backfill=False,
            sched_params=params(default_queue_depth=depth),
        )
        return [j.start_time for j in jobs[1:]]

    # Three start on the arrival event; the rest wait for the sched_interval pass.
    assert starts(2) == [10.0, 10.0, 10.0, 60.0, 60.0]
    # 0 is a valid depth in Slurm, and it still tests one job.
    assert starts(0) == [10.0, 60.0, 60.0, 60.0, 60.0]

    deep = work()
    simulate(deep, Cluster.homogeneous(8, 1), weights=weights_only(), backfill=False)
    assert [j.start_time for j in deep[1:]] == [10.0] * 5


def test_sched_interval_minus_one_disables_every_main_pass():
    """job_scheduler.c L1360-L1372: `_schedule()` returns before doing
    anything, for event-triggered calls too. Only backfill starts jobs."""
    jobs = _hole(long_first=100.0)
    result = simulate(jobs, Cluster.homogeneous(4, 1), sched_params=params(sched_interval=-1))
    assert result.main_passes == 0
    # Job 1 arrives on an idle cluster at t=0, where the first bf tick is.
    assert jobs[0].start_time == 0.0
    assert jobs[2].start_time == 30.0  # the next bf tick, not its arrival at t=2
    assert set(result.backfilled_ids) == {1, 2, 3}

    with pytest.raises(ValueError, match="no job could ever start"):
        simulate(
            _hole(),
            Cluster.homogeneous(4, 1),
            backfill=False,
            sched_params=params(sched_interval=-1),
        )


def test_sched_interval_zero_makes_every_event_pass_a_full_one():
    """controller.c L2939: a full pass whenever `now - last >= sched_interval`,
    so with 0 no event pass is ever depth-limited."""

    def run(interval: float) -> list[float | None]:
        jobs = [job(0, 0, 1000)] + [job(i, 10, 1000) for i in range(1, 6)]
        simulate(
            jobs,
            Cluster.homogeneous(8, 1),
            weights=weights_only(),
            backfill=False,
            sched_params=params(default_queue_depth=0, sched_interval=interval),
        )
        return [j.start_time for j in jobs[1:]]

    assert run(60.0) == [10.0, 60.0, 60.0, 60.0, 60.0]
    assert run(0.0) == [10.0] * 5


def test_a_blocked_job_removes_its_partitions_nodes_from_the_main_pass():
    """job_scheduler.c: "Do not schedule more jobs in this partition or on nodes
    in this partition" — `bit_and_not(avail_node_bitmap, part_ptr->node_bitmap)`.

    Every partition spans every node here, so p2's job cannot take the free
    node either. PART A pinned the opposite (job 3 starting at t=2), which let
    a lower-priority job in an overlapping partition start on the nodes the
    blocked job was waiting for.
    """
    work = [
        job(1, 0, 100, nodes=1, partition="p1"),
        job(2, 1, 100, nodes=2, partition="p1"),  # blocked until t=100
        job(3, 2, 50, nodes=1, partition="p2"),  # overlapping partition: also held
        job(4, 3, 10, nodes=1, partition="p1"),  # behind job 2
    ]
    simulate(work, Cluster.homogeneous(2, 1), weights=SIZE, backfill=False)
    ids = by_id(work)
    assert ids[2].start_time == 100.0
    assert ids[3].start_time == 200.0
    assert ids[4].start_time == 200.0


# ── Time limits still matter in conservative mode ───────────────────────────


def test_padded_time_limit_forfeits_backfill_in_conservative_mode():
    honest = [job(1, 0, 1000, nodes=3), job(2, 1, 100, nodes=4), job(3, 2, 50, nodes=1)]
    padded = [
        job(1, 0, 1000, nodes=3),
        job(2, 1, 100, nodes=4),
        job(3, 2, 50, nodes=1, time_limit=5000),
    ]
    simulate(honest, Cluster.homogeneous(4, 1))
    simulate(padded, Cluster.homogeneous(4, 1))
    assert honest[2].start_time == 30.0
    assert started(padded[2]) > 1000.0


# ── The planner itself ──────────────────────────────────────────────────────


def _brute_force(space: NodeSpace, j: Job, now: float, horizon: float) -> float | None:
    """Earliest feasible start by checking every candidate instant."""
    candidates = {now}
    for node in space.nodes:
        candidates.update(end for end, _, _ in node.allocs)
        candidates.update(node.res_ends)
    for t in sorted(c for c in candidates if now <= c <= horizon):
        ok = 0
        for node in space.nodes:
            busy_c = sum(c for end, c, _ in node.allocs if end > t)
            busy_g = sum(g for end, _, g in node.allocs if end > t)
            fits = node.cpus - busy_c >= j.cpus_per_node and node.gpus - busy_g >= j.gpus_per_node
            clear = all(
                not (t + j.time_limit > s and t < e)
                for s, e in zip(node.res_starts, node.res_ends, strict=True)
            )
            ok += fits and clear
        if ok >= j.nodes:
            return t
    return None


@pytest.mark.parametrize("seed", range(40))
def test_planner_matches_brute_force(seed):
    rng = random.Random(seed)
    cluster = Cluster.homogeneous(rng.randint(2, 6), rng.choice([1, 4, 8]), rng.choice([0, 2]))
    space = NodeSpace(cluster, [], resolution=rng.choice([1.0, 60.0]))
    res = space.resolution
    for node in space.nodes:
        used_c, used_g = 0, 0
        for _ in range(rng.randint(0, 3)):
            c = rng.randint(0, node.cpus - used_c)
            g = rng.randint(0, node.gpus - used_g)
            used_c, used_g = used_c + c, used_g + g
            node.add_alloc(rng.uniform(0, 2000), c, g)
        t = 0.0
        for _ in range(rng.randint(0, 3)):
            s = math.ceil((t + rng.uniform(0, 800)) / res) * res
            e = s + math.ceil(rng.uniform(1, 600) / res) * res
            node.add_reservation(s, e)
            t = e
    probe = job(
        99,
        0,
        rng.uniform(10, 900),
        nodes=rng.randint(1, len(space.nodes)),
        cpus_per_node=rng.randint(1, cluster.nodes[0].cpus),
        gpus_per_node=rng.randint(0, cluster.nodes[0].gpus),
    )
    now, horizon = rng.uniform(0, 100), rng.uniform(500, 5000)

    found = space.earliest_start(probe, now, horizon)
    expected = _brute_force(space, probe, now, horizon)
    assert (found[0] if found else None) == expected
    if found:
        assert len(found[1]) == probe.nodes == len(set(found[1]))


def _sweep_only(
    space: NodeSpace, j: Job, now: float, horizon: float
) -> tuple[float, list[int]] | None:
    """`earliest_start` without its shortcuts: the plain interval sweep."""
    events: list[tuple[float, int, int]] = []
    per_node: dict[int, list[tuple[float, float]]] = {}
    for nid, node in enumerate(space.nodes):
        spans = node.feasible_starts(j.cpus_per_node, j.gpus_per_node, j.time_limit, now, horizon)
        if spans:
            per_node[nid] = spans
            for a, b in spans:
                events += [(a, 0, nid), (b, 1, nid)]
    open_count = 0
    for t, kind, _ in sorted(events):
        if kind == 1:
            open_count -= 1
            continue
        open_count += 1
        if open_count >= j.nodes:
            chosen = [n for n, spans in per_node.items() if any(a <= t <= b for a, b in spans)]
            return t, sorted(chosen)[: j.nodes]
    return None


@pytest.mark.parametrize("seed", range(40))
def test_planner_shortcuts_pick_the_same_time_and_nodes_as_the_sweep(seed):
    """The start-now fast path and the capacity filter change no answer."""
    rng = random.Random(seed)
    cluster = Cluster.homogeneous(rng.randint(2, 6), 4, rng.choice([0, 2]))
    running = []
    for i in range(rng.randint(0, 5)):
        r = job(100 + i, 0, rng.uniform(10, 200), cpus_per_node=rng.randint(1, 2))
        nodes = cluster.find_nodes(r)
        if nodes is not None:
            r.start_time = 0.0
            cluster.allocate(r, nodes)
            running.append(r)
    space = NodeSpace(cluster, running, resolution=rng.choice([1, 60]))
    for k in range(12):
        probe = job(
            k,
            0,
            rng.uniform(5, 300),
            nodes=rng.randint(1, len(cluster.nodes)),
            cpus_per_node=rng.randint(1, 4),
            gpus_per_node=rng.randint(0, cluster.nodes[0].gpus),
        )
        now = rng.uniform(0, 50)
        horizon = now + rng.choice([0.0, 100.0, 5000.0])
        got = space.earliest_start(probe, now, horizon)
        assert got == _sweep_only(space, probe, now, horizon)
        if got is not None and rng.random() < 0.6:
            space.reserve(probe, got[1], got[0])


def _fifo_exact_run(seed: int, whole_node: bool) -> tuple[list[Job], sim.SimulationResult]:
    """FIFO priorities (no later arrival outranks a planned job), exact runtime
    estimates, 1 s resolution and a 1 s cycle."""
    rng = random.Random(seed)
    work, t = [], 0
    for i in range(1, 61):
        t += rng.randint(0, 40)
        d = rng.randint(10, 400)
        nodes = rng.randint(1, 3)
        cpus = 4 if whole_node else rng.randint(1, 4)
        work.append(job(i, t, d, nodes=nodes, cpus_per_node=cpus))
    result = simulate(
        work,
        Cluster.homogeneous(6, 4),
        weights=weights_only(),
        sched_params=params(bf_resolution=1, bf_interval=1, bf_window=43_200),
    )
    return work, result


def _slipped(work: list[Job], result: sim.SimulationResult) -> list[int]:
    return [
        j.job_id
        for j in work
        if j.job_id in result.planned_starts
        and (j.start_time or 0.0) > result.planned_starts[j.job_id] + 1e-9
    ]


@pytest.mark.parametrize("seed", range(8))
def test_whole_node_jobs_never_start_later_than_first_planned(seed):
    """The conservative guarantee, end to end, where it genuinely holds: FIFO
    priorities, exact limits, a window covering every plan, and 1 s
    resolution and cycle (`_fifo_exact_run`). Outside those conditions a
    planned start is a prediction, not a bound (next tests, and README)."""
    work, result = _fifo_exact_run(seed, whole_node=True)
    assert result.planned_starts, "the scenario should exercise reservations"
    assert _slipped(work, result) == []


def test_sub_node_jobs_can_slip_behind_their_first_plan():
    """A measured limit of the model, pinned so it cannot change silently.

    Reservations are whole-node (sched_config.html), allocations are not. When
    a reserved sub-node job starts, the CPUs its reservation blocked become
    free, the next cycle rebuilds the plan from scratch (as backfill.c does),
    higher-priority jobs move earlier into that capacity, and a lower-priority
    job's slot can go with them. There is no "never later than last cycle"
    rule, so the conservative guarantee is exact only for whole-node jobs.
    Seed 11: job 51 was first planned for t=1556 and started at 1734.
    """
    work, result = _fifo_exact_run(11, whole_node=False)
    assert 51 in _slipped(work, result)
    assert result.planned_starts[51] == 1556
    assert by_id(work)[51].start_time == 1734


def test_planned_start_is_not_a_bound_once_a_higher_priority_job_arrives():
    """The whole-node guarantee above needs FIFO priorities: with any other
    weights, a later, higher-priority job takes a planned slot. One node:
    job 1 holds it to 100; job 2 (submitted 10) is planned for 100; job 3
    (QOS, submitted 40) outranks it, starts at 100, and job 2 slips to 150.
    Whole-node jobs, exact limits, 1 s resolution: only the order moved it.
    """
    work = [job(1, 0, 100), job(2, 10, 50), job(3, 40, 50, qos_factor=1.0)]
    result = simulate(
        work,
        Cluster.homogeneous(1, 1),
        weights=weights_only(qos=1000.0),
        sched_params=params(bf_resolution=1, bf_interval=1, bf_window=43_200 * 60),
    )
    assert result.planned_starts[2] == 100
    assert [j.start_time for j in work] == [0.0, 150.0, 100.0]
    assert _slipped(work, result) == [2]


def _k8s_like(seed: int, count: int = 150) -> list[Job]:
    """An S0-shaped workload: GPU gangs of one-CPU pods, ×3 padded limits.

    Sub-node jobs, where whole-node reservations and exact node timing
    matter most. Built through the S0 adapter from random k8s-style records.
    """
    rng = random.Random(seed)
    records, t = [], 0.0
    for i in range(1, count + 1):
        t += rng.expovariate(1 / 60)
        records.append(
            k8strace.K8sRecord(
                job_id=i,
                account=rng.choice("abc"),
                submit_time=t,
                duration=min(rng.lognormvariate(6.5, 1.2), 20_000.0),
                gpus=rng.choice([1, 1, 2, 4, 8]),
                gang_size=rng.choice([1, 1, 1, 2, 4]),
                priority=rng.choice([0, 0, 100, 500]),
            )
        )
    return k8strace.to_jobs(records, time_limit_model="padded", time_limit_factor=3.0).jobs


def _everything_a_cycle_decides(result: sim.SimulationResult) -> tuple[object, ...]:
    """Start times, first planned starts, and every cycle's statistics."""
    return (
        [(j.job_id, j.start_time) for j in result.jobs],
        sorted(result.planned_starts.items()),
        [
            (
                c.time,
                c.queue_length,
                c.jobs_tested,
                c.jobs_started,
                c.reservations,
                c.beyond_window,
                c.skipped_by_limits,
                c.hit_max_job_test,
            )
            for c in result.backfill_stats.cycles
        ],
    )


@pytest.mark.parametrize(
    ("workload", "seed", "window_minutes"),
    [
        ("synthetic", 3, 1440),
        ("synthetic", 5, 120),
        ("synthetic", 7, 30),
        ("k8s", 8, 120),  # the old replay key changed start times here
        ("k8s", 2, 30),
        ("k8s", 4, 1440),
    ],
)
def test_replayed_cycles_match_full_evaluation(monkeypatch, workload, seed, window_minutes):
    """Skipping unchanged cycles, and skipping the planner for jobs that are
    beyond the window by dominance, are optimisations, not behaviour changes.

    Whole-node and sub-node workloads, default and short windows: identical
    start times, first planned starts and per-cycle statistics against a run
    that plans every tested job of every cycle. A replay key that ignored
    jobs entering `bf_window` failed the ("k8s", 8, 120) case on start
    times, and most cases on planned starts.
    """
    if workload == "synthetic":
        base, shape = generate(WorkloadProfile(job_count=150), seed=seed), (16, 8, 2)
    else:
        base, shape = _k8s_like(seed), (8, 32, 8)
    p = SchedulerParameters.parse(f"bf_window={window_minutes}")

    def run(optimised: bool) -> sim.SimulationResult:
        monkeypatch.setattr(sim, "REPLAY_UNCHANGED_BACKFILL_CYCLES", optimised)
        monkeypatch.setattr(sim, "PRUNE_DOMINATED_BEYOND_WINDOW", optimised)
        return simulate(clone(base), Cluster.homogeneous(*shape), sched_params=p)

    fast, slow = run(True), run(False)
    assert _everything_a_cycle_decides(fast) == _everything_a_cycle_decides(slow)
    assert fast.backfill_stats.evaluated < slow.backfill_stats.evaluated
    assert slow.backfill_stats.evaluated == slow.backfill_stats.count


def test_replay_reevaluates_once_a_job_comes_inside_bf_window(monkeypatch):
    """The plan depends on `now` through `now + bf_window`.

    Two 2-CPU nodes, a 10-minute window. A (1 CPU) and B (2 CPU) run from
    t=0 to 1000 and 1300. G (both nodes, top QOS) can start at 1300, beyond
    the window until t=700, so until then it has no reservation and H (one
    whole node) is planned for 1000 on node 0. X (1 CPU, 500 s) arrives at
    650: node 0's free CPU is blocked by H's reservation at 1000. From t=700
    G is inside the window, reserves both nodes at 1300 and pushes H to
    1400, and X fits on node 0 before 1300 — so the 720 s cycle starts it.
    A replay keyed only on queue and state skipped that cycle and every one
    after it until A ended, and X started at 1400.
    """

    def five() -> list[Job]:
        return [
            job(1, 0, 1000, cpus_per_node=1),
            job(2, 0, 1300, cpus_per_node=2),
            job(3, 1, 100, nodes=2, cpus_per_node=2, qos_factor=1.0),  # G
            job(4, 2, 4500, cpus_per_node=2, qos_factor=0.5),  # H
            job(5, 650, 500, cpus_per_node=1, qos_factor=0.1),  # X
        ]

    for replay in (True, False):
        monkeypatch.setattr(sim, "REPLAY_UNCHANGED_BACKFILL_CYCLES", replay)
        jobs = five()
        result = simulate(
            jobs,
            Cluster.homogeneous(2, 2, 0),
            weights=weights_only(qos=1000.0),
            sched_params=SchedulerParameters.parse("bf_window=10"),
        )
        x = by_id(jobs)[5]
        assert x.start_time == 720.0
        assert 5 in result.backfilled_ids
        if replay:
            assert result.backfill_stats.evaluated < result.backfill_stats.count


def test_cycle_statistics_reach_the_metrics_report():
    jobs = generate(WorkloadProfile(job_count=100), seed=4)
    cluster = Cluster.homogeneous(16, 8, 2)
    metrics = compute(simulate(jobs, cluster), cluster)
    assert metrics.backfill_mode == "conservative"
    assert metrics.bf_cycles >= metrics.bf_cycles_evaluated > 0
    assert 0 < metrics.bf_mean_tested <= metrics.bf_max_tested
    text = metrics.format()
    assert "backfill cycles" in text and "bf_max_job_test" in text


def test_unknown_backfill_mode_is_rejected():
    with pytest.raises(ValueError, match="backfill_mode"):
        simulate([job(1, 0, 1)], Cluster.homogeneous(1, 1), backfill_mode="aggressive")  # type: ignore[arg-type]


# ── The remaining per-cycle caps, unit-tested on one cycle ─────────────────


def _one_cycle(queue: list[Job], **caps: float) -> tuple[list[int], BackfillCycle]:
    """Every pending job is blocked by a full-cluster job until t=1000, so a
    job gets a planned start exactly when the cycle tested it."""
    cluster = Cluster.homogeneous(4, 1)
    blocker = job(0, 0, 1000, nodes=4)
    blocker.start_time = 0.0
    nodes = cluster.find_nodes(blocker)
    assert nodes is not None
    cluster.allocate(blocker, nodes)
    scheduler = sim.SlurmScheduler(cluster, params(**caps))
    scheduler.backfill_cycle(queue, [blocker], now=1.0, state_version=0)
    tested = [j.job_id for j in queue if j.planned_start is not None]
    return tested, scheduler.stats.cycles[-1]


def test_bf_max_job_part_caps_jobs_checked_per_partition():
    queue = [job(i, 0, 10, partition="p1" if i <= 3 else "p2") for i in range(1, 7)]
    tested, cycle = _one_cycle(queue, bf_max_job_part=2)
    assert tested == [1, 2, 4, 5]
    assert cycle.skipped_by_limits == 2


def test_bf_max_job_user_part_caps_per_user_within_a_partition():
    queue = [
        job(1, 0, 10, user="u1", partition="p1"),
        job(2, 0, 10, user="u1", partition="p1"),
        job(3, 0, 10, user="u1", partition="p2"),
        job(4, 0, 10, user="u2", partition="p1"),
    ]
    tested, _ = _one_cycle(queue, bf_max_job_user_part=1)
    assert tested == [1, 3, 4]


def test_bf_max_job_assoc_caps_per_user_and_account():
    queue = [
        job(1, 0, 10, user="u1", account="a"),
        job(2, 0, 10, user="u1", account="a"),
        job(3, 0, 10, user="u1", account="b"),
    ]
    tested, _ = _one_cycle(queue, bf_max_job_assoc=1)
    assert tested == [1, 3]


def test_bf_max_job_assoc_overrides_bf_max_job_user_in_the_cycle():
    """Set directly (not via parse), both caps: only the association cap
    applies, as after backfill.c's `_load_config()` (L974-L979)."""
    queue = [
        job(1, 0, 10, user="u1", account="a"),
        job(2, 0, 10, user="u1", account="b"),
        job(3, 0, 10, user="u1", account="b"),
    ]
    tested, cycle = _one_cycle(queue, bf_max_job_assoc=1, bf_max_job_user=1)
    assert tested == [1, 2]  # the user cap of 1 would have stopped at job 1
    assert cycle.skipped_by_limits == 1


def test_partition_job_depth_skips_do_not_count_against_queue_depth():
    cluster = Cluster.homogeneous(8, 1)
    queue = [job(i, 0, 10, partition="p1") for i in (1, 2, 3)]
    queue += [job(i, 0, 10, partition="p2") for i in (4, 5)]
    scheduler = sim.SlurmScheduler(cluster, params(partition_job_depth=1))
    started = scheduler.main_pass(queue, now=0.0, depth=2)
    # One per partition; the skipped p1 jobs did not use up the depth of 2
    # (which tests three jobs).
    assert [j.job_id for j in started] == [1, 4]
