"""Invariants that must hold in every mode and configuration."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest

import schedlab.simulate as sim
from schedlab.metrics import compute
from schedlab.model import Cluster, Job
from schedlab.priority import PriorityWeights
from schedlab.simulate import simulate
from schedlab.trace import WorkloadProfile, generate
from tests._factory import clone

MODES = ["easy", "conservative"]


def _run(jobs: list[Job], mode: str, backfill: bool = True, **weights_kw):
    weights = replace(PriorityWeights(), **weights_kw)
    cluster = Cluster.homogeneous(16, 8, 2)
    placed: dict[int, list[int]] = {}
    real_allocate = cluster.allocate

    def record(job: Job, node_ids: list[int]) -> None:
        placed[job.job_id] = list(node_ids)
        real_allocate(job, node_ids)

    cluster.allocate = record  # type: ignore[method-assign]
    result = simulate(
        jobs,
        cluster,
        weights=weights,
        fairshare=weights.make_fairshare(j.account for j in jobs),
        backfill=backfill,
        backfill_mode=mode,  # type: ignore[arg-type]
    )
    return result, cluster, placed


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("algorithm", ["fair_tree", "classic"])
@pytest.mark.parametrize("backfill", [True, False])
def test_every_job_runs_once_for_exactly_its_runtime(mode, algorithm, backfill):
    jobs = generate(WorkloadProfile(job_count=150), seed=3)
    result, cluster, _ = _run(jobs, mode, backfill, fairshare_algorithm=algorithm)
    for j in jobs:
        assert j.start_time is not None and j.end_time is not None
        assert j.start_time >= j.submit_time
        assert j.end_time == pytest.approx(j.start_time + j.duration)
    assert cluster.busy_cpus == 0
    assert len(result.jobs) == len(jobs)


@pytest.mark.parametrize("mode", MODES)
def test_no_node_is_ever_overcommitted(mode):
    """Checked over time, not just at the end: sweep every node's intervals."""
    jobs = generate(WorkloadProfile(job_count=200), seed=11)
    _, cluster, placed = _run(jobs, mode)
    by_id = {j.job_id: j for j in jobs}
    for node in cluster.nodes:
        events: list[tuple[float, int, int, int]] = []
        for job_id, nodes in placed.items():
            if node.node_id in nodes:
                j = by_id[job_id]
                assert j.start_time is not None and j.end_time is not None
                events.append((j.start_time, 1, j.cpus_per_node, j.gpus_per_node))
                events.append((j.end_time, -1, j.cpus_per_node, j.gpus_per_node))
        cpus = gpus = 0
        for _, sign, c, g in sorted(events, key=lambda e: (e[0], e[1])):  # ends first
            cpus += sign * c
            gpus += sign * g
            assert 0 <= cpus <= node.cpus
            assert 0 <= gpus <= node.gpus


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("calc_period", [300.0, 0.0])
def test_simulation_is_deterministic(mode, calc_period):
    def run() -> list[float | None]:
        jobs = generate(WorkloadProfile(job_count=100), seed=42)
        _run(jobs, mode, calc_period=calc_period)
        return [j.start_time for j in jobs]

    assert run() == run()


@pytest.mark.parametrize("mode", MODES)
def test_backfill_beats_no_backfill_on_wait(mode):
    base = generate(WorkloadProfile(job_count=150), seed=7)
    on, c_on, _ = _run(clone(base), mode, backfill=True)
    off, c_off, _ = _run(clone(base), mode, backfill=False)
    assert compute(on, c_on).mean_wait < compute(off, c_off).mean_wait
    assert on.makespan <= off.makespan


def test_modes_agree_when_there_is_nothing_to_backfill():
    """A cluster big enough for everything: both modes start jobs on arrival."""
    base = generate(WorkloadProfile(job_count=50, node_choices=(1,)), seed=1)
    easy, cons = clone(base), clone(base)
    simulate(easy, Cluster.homogeneous(64, 8, 2), backfill_mode="easy")
    simulate(cons, Cluster.homogeneous(64, 8, 2))
    for e, c in zip(easy, cons, strict=True):
        assert e.start_time == c.start_time == e.submit_time


# ── Inputs the simulator must survive, or refuse clearly ───────────────────


@contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    """Fail instead of hanging: the duplicate-id bug looped forever."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def expired(signum: int, frame: object) -> None:
        raise TimeoutError(f"simulation still running after {seconds} s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("replay", [True, False])
def test_two_jobs_sharing_an_id_both_finish(monkeypatch, mode, replay):
    """Run epochs were keyed by job id, so the second job's start made the
    first job's completion look stale: it never ended and the loop never
    stopped (0.1.0 returned [(0, 100), (10, 110)] here)."""
    monkeypatch.setattr(sim, "REPLAY_UNCHANGED_BACKFILL_CYCLES", replay)
    jobs = [Job(1234, "a", 0, 100, 200), Job(1234, "a", 10, 100, 200)]
    with _deadline(20):
        result = simulate(jobs, Cluster.homogeneous(2, 1, 0), backfill_mode=mode)
    assert [(j.start_time, j.end_time) for j in result.jobs] == [(0, 100), (10, 110)]


def test_shared_ids_survive_fairshare_and_preemption():
    """Accrual and preemption bookkeeping are per job too, not per id.

    Preemption is on, over the QOS ladder the CLI builds from the synthetic
    trace (`cli._synthetic_ladder`). This test used to leave it off, so it
    could not see that victim selection was keyed by id and preempted
    nothing once two candidates shared one.
    """
    from schedlab.cli import _synthetic_ladder
    from schedlab.preempt import PreemptionConfig, PreemptMode

    weights = PriorityWeights()

    def run(shared: bool) -> tuple[list[list[tuple[float, float | None, str | None]]], int]:
        jobs = generate(WorkloadProfile(job_count=80), seed=4)
        if shared:
            for j in jobs:
                j.job_id = 7  # every job shares one id
        ladder = _synthetic_ladder(jobs, "preempt/qos", 60.0)
        with _deadline(60):
            result = simulate(
                jobs,
                Cluster.homogeneous(16, 8, 2),
                weights=weights,
                fairshare=weights.make_fairshare(j.account for j in jobs),
                preemption=PreemptionConfig(
                    "preempt/qos", PreemptMode("REQUEUE"), qos=ladder.qos
                ),
            )
        return [[(r.start, r.end, r.outcome) for r in j.runs] for j in jobs], len(
            result.preemptions
        )

    # An id is only the last tiebreak, after priority and submit time (and,
    # for preemption candidates, after QOS priority and size), and on this
    # trace no tie reaches it: sharing one id must change nothing.
    unique_runs, unique_preemptions = run(shared=False)
    assert unique_preemptions > 0, "the scenario should exercise preemption"
    assert run(shared=True) == (unique_runs, unique_preemptions)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("replay", [True, False])
def test_a_run_longer_than_its_time_limit_is_refused_clearly(monkeypatch, mode, replay):
    """Slurm kills such a job at its limit (TIMEOUT), which is not modelled.
    Conservative mode used to plan onto nodes the job still held and crash
    with "node 0 cannot fit"; EASY ran it past its limit."""
    monkeypatch.setattr(sim, "REPLAY_UNCHANGED_BACKFILL_CYCLES", replay)
    jobs = [
        Job(1, "a", 0, 500, 100),
        Job(3, "a", 0, 150, 200),
        Job(2, "a", 1, 50, 60, nodes=2),
    ]
    with pytest.raises(ValueError, match=r"job 1: runtime 500 s exceeds its time limit 100 s"):
        simulate(jobs, Cluster.homogeneous(2, 1, 0), backfill_mode=mode)
