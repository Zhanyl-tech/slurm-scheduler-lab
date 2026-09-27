"""A4: Fair Tree (Slurm's default since 19.05) alongside classic fairshare."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from schedlab.model import Cluster
from schedlab.priority import FairshareTree, PriorityWeights, compute_priority
from schedlab.simulate import simulate
from schedlab.trace import WorkloadProfile, generate
from tests._factory import clone, job, weights_only


def tree(usage: dict[str, float], shares: dict[str, float] | None = None, **kw) -> FairshareTree:
    shares = shares or {a: 1.0 for a in usage}
    return FairshareTree(shares=shares, usage=dict(usage), **kw)


def test_fair_tree_is_the_default():
    assert FairshareTree(shares={"a": 1.0}).algorithm == "fair_tree"
    assert PriorityWeights().fairshare_algorithm == "fair_tree"
    assert PriorityWeights.slurm_defaults().fairshare_algorithm == "fair_tree"


def test_level_fairshare_is_shares_over_usage():
    t = tree({"a": 60.0, "b": 40.0}, shares={"a": 3.0, "b": 1.0})
    assert t.level_fairshare("a") == pytest.approx(0.75 / 0.6)
    assert t.level_fairshare("b") == pytest.approx(0.25 / 0.4)


def test_fair_tree_factor_is_rank_over_account_count():
    t = tree({"research": 50.0, "trading": 35.0, "infra": 15.0})
    assert t.factors() == pytest.approx({"infra": 1.0, "trading": 2 / 3, "research": 1 / 3})


def test_no_usage_means_infinite_level_fairshare_and_a_shared_top_rank():
    t = tree({"a": 0.0, "b": 0.0})
    assert t.level_fairshare("a") == float("inf")
    assert t.factors() == {"a": 1.0, "b": 1.0}


def test_ties_share_a_rank_and_the_next_rank_is_skipped():
    """Competition ranking, as `rank`/`rnt` in fair_tree.c `_calc_tree_fs()`."""
    t = tree({"idle": 0.0, "b": 10.0, "c": 10.0, "d": 40.0})
    # LF: idle=inf, b=c=(1/4)/(1/6)=1.5, d=(1/4)/(4/6)=0.375
    assert t.factors() == pytest.approx({"idle": 1.0, "b": 0.75, "c": 0.75, "d": 0.25})


def test_zero_shares_rank_last_but_still_rank():
    """S == 0 gives LF 0 (fair_tree.c `_calc_assoc_fs()`), not factor 0."""
    t = tree({"a": 0.0, "z": 0.0}, shares={"a": 1.0, "z": 0.0})
    assert t.level_fairshare("z") == 0.0
    assert t.factors() == {"a": 1.0, "z": 0.5}


def test_inactive_accounts_still_take_ranks():
    """fair_tree.html: "the Fair Tree algorithm ranks all users, active or not"."""
    t = tree({"a": 70.0, "b": 30.0, "idle": 0.0})
    assert t.factors() == pytest.approx({"idle": 1.0, "b": 2 / 3, "a": 1 / 3})


def test_accounts_outside_the_tree_get_no_fairshare():
    t = tree({"a": 1.0})
    assert t.factor("stranger") == 0.0
    assert replace(t, algorithm="classic").factor("stranger") == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_flat_tree_orders_accounts_like_classic(seed):
    """On one level both are monotone in U/S, so they agree on *order*."""
    rng = random.Random(seed)
    usage = {f"acct{i}": rng.choice([0.0, rng.uniform(1, 100)]) for i in range(5)}
    shares = {a: rng.uniform(0.5, 3) for a in usage}
    ft = tree(usage, shares).factors()
    cl = tree(usage, shares, algorithm="classic").factors()
    for x in usage:
        for y in usage:
            if ft[x] > ft[y]:
                assert cl[x] >= cl[y]


def test_fair_tree_turns_a_small_usage_gap_into_a_full_rank_step():
    """The flat-tree difference that matters: spacing, not order.

    a has used 50.5% and b 49.5%. Classic barely separates them; Fair Tree
    puts a whole rank between them. With the lab's default weights that is
    enough to flip a jobsize-driven decision — the reason the Fair Tree docs
    suggest a much smaller PriorityWeightFairshare than classic needs.
    """
    usage = {"a": 50.5, "b": 49.5}
    cluster = Cluster.homogeneous(4, 1)
    big_a = job(1, 0, 10, nodes=2, account="a")
    small_b = job(2, 0, 10, nodes=1, account="b")
    weights = weights_only(fairshare=10_000.0, jobsize=10_000.0)

    for algorithm, winner in (("classic", 1), ("fair_tree", 2)):
        t = tree(usage, algorithm=algorithm)
        p = {
            j.job_id: compute_priority(j, 0.0, cluster, weights, t)
            for j in (big_a, small_b)
        }
        assert max(p, key=lambda k: p[k]) == winner, algorithm


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError):
        FairshareTree(shares={"a": 1.0}, algorithm="fairer")  # type: ignore[arg-type]


def test_classic_and_fair_tree_schedule_the_same_workload_differently():
    base = generate(WorkloadProfile(job_count=200), seed=5)
    starts = {}
    for algorithm in ("classic", "fair_tree"):
        jobs = clone(base)
        weights = replace(PriorityWeights(), fairshare_algorithm=algorithm)
        simulate(
            jobs,
            Cluster.homogeneous(16, 8, 2),
            weights=weights,
            fairshare=weights.make_fairshare(j.account for j in jobs),
        )
        assert all(j.end_time is not None for j in jobs)
        starts[algorithm] = [j.start_time for j in jobs]
    assert starts["classic"] != starts["fair_tree"]


def test_make_fairshare_wires_half_life_and_algorithm():
    weights = replace(PriorityWeights(), decay_half_life=123.0, fairshare_algorithm="classic")
    t = weights.make_fairshare(["b", "a", "a"])
    assert t.shares == {"a": 1.0, "b": 1.0}
    assert t.half_life == 123.0
    assert t.algorithm == "classic"
    assert weights.make_fairshare({"a": 3.0}).shares == {"a": 3.0}
