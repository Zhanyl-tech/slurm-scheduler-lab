"""Definition C: structural fragmentation.

`test_golden_vector` is shared VERBATIM with k8s-gpu-scheduler-lab
(tests/test_fragmentation.py there): both labs implement the same pure
function and must produce the same numbers on the same input. Do not edit the
vector or the expected values in one lab only.
"""

from __future__ import annotations

import random

import pytest

from schedlab.fragmentation import structural_fragmentation, structural_fragmentation_any
from schedlab.metrics import compute
from schedlab.model import Cluster, Job, Node
from schedlab.simulate import simulate
from schedlab.trace import WorkloadProfile, generate
from tests._factory import job

# ---- DEFINITION C GOLDEN TEST VECTOR (shared with k8s-gpu-scheduler-lab) ----
# Nodes: n0 (rack r0, switch s0, capacity 8 GPUs), n1 (rack r0, switch s0,
# capacity 8), n2 (rack r1, switch s0, capacity 4), n3 (rack r1, switch s0,
# capacity 4).
# Free-GPU samples (step function: each sample holds until the next sample):
#   t=0:  {n0: 8, n1: 3, n2: 4, n3: 0}
#   t=10: {n0: 8, n1: 8, n2: 4, n3: 4}
#   t=20: {n0: 8, n1: 8, n2: 4, n3: 4}   (horizon)
# Expected: frag_C^node = 30/390 = 1/13; frag_C^rack = 150/390 = 5/13;
#           frag_C^switch = 150/390 = 5/13.
GOLDEN_CAPACITY = {"n0": 8, "n1": 8, "n2": 4, "n3": 4}
GOLDEN_DOMAINS = {
    "rack": {"n0": "r0", "n1": "r0", "n2": "r1", "n3": "r1"},
    "switch": {"n0": "s0", "n1": "s0", "n2": "s0", "n3": "s0"},
}
GOLDEN_SAMPLES: list[tuple[float, dict[str, int]]] = [
    (0.0, {"n0": 8, "n1": 3, "n2": 4, "n3": 0}),
    (10.0, {"n0": 8, "n1": 8, "n2": 4, "n3": 4}),
    (20.0, {"n0": 8, "n1": 8, "n2": 4, "n3": 4}),
]


def test_golden_vector() -> None:
    r = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, GOLDEN_SAMPLES)
    # The integrals themselves, so a disagreement between labs can be located.
    assert r.free_gpu_seconds == 390.0
    assert r.carved_gpu_seconds == {"node": 30.0, "rack": 150.0, "switch": 150.0}
    assert r.rate("node") == 30 / 390 == pytest.approx(1 / 13, abs=1e-15)
    assert r.rate("rack") == 150 / 390 == pytest.approx(5 / 13, abs=1e-15)
    assert r.rate("switch") == 150 / 390 == pytest.approx(5 / 13, abs=1e-15)
    assert round(r.rate("node"), 7) == 0.0769231
    assert round(r.rate("rack"), 7) == 0.3846154
    assert r.levels == ("node", "rack", "switch")


def test_golden_vector_agrees_with_the_k8s_lab_when_it_is_importable() -> None:
    """Runs only where k8slab is installed (not in this repo's venv)."""
    k8s = pytest.importorskip("k8slab.fragmentation")
    theirs = k8s.structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, GOLDEN_SAMPLES)
    ours = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, GOLDEN_SAMPLES)
    assert theirs.carved_gpu_seconds == ours.carved_gpu_seconds
    assert theirs.free_gpu_seconds == ours.free_gpu_seconds


# ---- the properties the definition promises --------------------------------


def _random_nested(rng: random.Random, nodes: int) -> tuple[
    dict[str, int], dict[str, dict[str, str]], list[tuple[float, dict[str, int]]]
]:
    names = [f"x{i}" for i in range(nodes)]
    capacity = {n: rng.choice([0, 1, 2, 4, 8]) for n in names}
    racks = {n: f"r{i // rng.randint(1, 4)}" for i, n in enumerate(names)}
    switch_of_rack = {r: f"s{int(r[1:]) // 2}" for r in set(racks.values())}
    domains = {"rack": racks, "switch": {n: switch_of_rack[racks[n]] for n in names}}
    t = 0.0
    samples = []
    for _ in range(rng.randint(2, 12)):
        samples.append((t, {n: rng.randint(0, c) for n, c in capacity.items()}))
        t += rng.choice([0.0, 1.0, 2.5, 7.0])
    return capacity, domains, samples


def test_levels_are_monotone_on_random_nested_topologies() -> None:
    rng = random.Random(7)
    for _ in range(300):
        cap, dom, samples = _random_nested(rng, rng.randint(1, 12))
        r = structural_fragmentation(cap, dom, samples).rates()
        assert r["node"] <= r["rack"] + 1e-12
        assert r["rack"] <= r["switch"] + 1e-12
        assert all(0.0 <= v <= 1.0 for v in r.values())


def test_idle_fleet_is_zero_at_every_level() -> None:
    samples = [(0.0, dict(GOLDEN_CAPACITY)), (50.0, dict(GOLDEN_CAPACITY))]
    r = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, samples)
    assert r.free_gpu_seconds > 0
    assert r.rates() == {"node": 0.0, "rack": 0.0, "switch": 0.0}


def test_fully_allocated_fleet_has_no_denominator_and_reports_zero() -> None:
    full = dict.fromkeys(GOLDEN_CAPACITY, 0)
    r = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, [(0.0, full), (9.0, full)])
    assert r.free_gpu_seconds == 0.0
    assert r.rates() == {"node": 0.0, "rack": 0.0, "switch": 0.0}


def test_zero_length_intervals_and_the_final_sample_are_not_integrated() -> None:
    doubled = [GOLDEN_SAMPLES[0], GOLDEN_SAMPLES[0], *GOLDEN_SAMPLES[1:]]
    tail = [*GOLDEN_SAMPLES[:-1], (20.0, {"n0": 0, "n1": 0, "n2": 0, "n3": 0})]
    base = structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, GOLDEN_SAMPLES)
    for samples in (doubled, tail):
        assert structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, samples) == base


def test_dropping_repeated_states_changes_nothing() -> None:
    """The simulator records only state changes; that must be exact."""
    rng = random.Random(3)
    for _ in range(100):
        cap, dom, samples = _random_nested(rng, rng.randint(1, 8))
        compressed = [
            s for i, s in enumerate(samples) if i == 0 or s[1] != samples[i - 1][1]
        ]
        if compressed[-1][0] != samples[-1][0]:
            compressed.append(samples[-1])
        full = structural_fragmentation(cap, dom, samples)
        short = structural_fragmentation(cap, dom, compressed)
        assert short.free_gpu_seconds == pytest.approx(full.free_gpu_seconds)
        for level in full.levels:
            assert short.carved_gpu_seconds[level] == pytest.approx(
                full.carved_gpu_seconds[level]
            )


def test_node_level_alone_needs_no_domains() -> None:
    r = structural_fragmentation(GOLDEN_CAPACITY, {}, GOLDEN_SAMPLES)
    assert r.levels == ("node",)
    assert r.rate("node") == 30 / 390


@pytest.mark.parametrize(
    ("capacity", "domains", "samples", "match"),
    [
        (GOLDEN_CAPACITY, {"node": {}}, GOLDEN_SAMPLES, "implicit"),
        (GOLDEN_CAPACITY, {"rack": {"n0": "r0"}}, GOLDEN_SAMPLES, "no domain"),
        (
            GOLDEN_CAPACITY,
            {"rack": {**GOLDEN_DOMAINS["rack"], "zz": "r9"}},
            GOLDEN_SAMPLES,
            "unknown node",
        ),
        (
            GOLDEN_CAPACITY,
            {
                "rack": GOLDEN_DOMAINS["rack"],
                "switch": {"n0": "s0", "n1": "s1", "n2": "s1", "n3": "s1"},
            },
            GOLDEN_SAMPLES,
            "does not nest",
        ),
        (GOLDEN_CAPACITY, GOLDEN_DOMAINS, [(0.0, {"n0": 8}), (1.0, {"n0": 8})], "no free count"),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(0.0, {"n0": 9, "n1": 0, "n2": 0, "n3": 0}), (1.0, dict(GOLDEN_CAPACITY))],
            "outside",
        ),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(5.0, dict(GOLDEN_CAPACITY)), (1.0, dict(GOLDEN_CAPACITY))],
            "out of time order",
        ),
        ({"n0": -1}, {}, [], "capacity"),
        # Samples that are never integrated must still be valid: the final
        # sample, one followed by a zero-length interval, and a lone sample.
        # (The k8s lab's cases, verbatim. This port used to accept all five.)
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [GOLDEN_SAMPLES[0], (10.0, {"n0": 99, "n1": 8, "n2": 4, "n3": 4})],
            "outside",
        ),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(0.0, {"n0": -5, "n1": 0, "n2": 0, "n3": 0}), *GOLDEN_SAMPLES],
            "outside",
        ),
        (GOLDEN_CAPACITY, GOLDEN_DOMAINS, [GOLDEN_SAMPLES[0], (10.0, {"n0": 8})], "no free count"),
        (GOLDEN_CAPACITY, GOLDEN_DOMAINS, [(0.0, {"bogus": 1})], "no free count"),
        (
            GOLDEN_CAPACITY,
            GOLDEN_DOMAINS,
            [(0.0, {**GOLDEN_CAPACITY, "zz": 1}), (1.0, dict(GOLDEN_CAPACITY))],
            "unknown node",
        ),
        # The finding that exposed the gap: a final sample with a node outside
        # the fleet, and out-of-range counts, after one valid interval.
        (
            {"n0": 8, "n1": 8},
            {},
            [(0.0, {"n0": 8, "n1": 3}), (10.0, {"n0": 99, "n1": -5, "zz": 1})],
            "unknown node",
        ),
    ],
)
def test_rejects_malformed_input(capacity, domains, samples, match) -> None:
    with pytest.raises(ValueError, match=match):
        structural_fragmentation(capacity, domains, samples)


def test_malformed_input_is_refused_by_both_labs_when_k8s_is_importable() -> None:
    """Same validation, not just the same arithmetic: whatever one refuses,
    the other refuses too, with the same message."""
    k8s = pytest.importorskip("k8slab.fragmentation")
    bad = [
        [GOLDEN_SAMPLES[0], (10.0, {"n0": 99, "n1": 8, "n2": 4, "n3": 4})],
        [GOLDEN_SAMPLES[0], (10.0, {"n0": 8})],
        [(0.0, {**GOLDEN_CAPACITY, "zz": 1}), (1.0, dict(GOLDEN_CAPACITY))],
        [(5.0, dict(GOLDEN_CAPACITY)), (1.0, dict(GOLDEN_CAPACITY))],
    ]
    for samples in bad:
        with pytest.raises(ValueError) as ours:
            structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, samples)
        with pytest.raises(ValueError) as theirs:
            k8s.structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, samples)
        assert str(ours.value) == str(theirs.value)


def test_variant_validates_every_cpu_sample_too() -> None:
    gpu = {"n0": 4, "n1": 4}
    cpu = {"n0": 8, "n1": 8}
    gpu_s = [(0.0, dict(gpu)), (10.0, dict(gpu))]
    for cpu_s, match in (
        ([(0.0, dict(cpu)), (10.0, {"n0": 8, "n1": 9})], "CPU sample at t=10.0: node n1"),
        ([(0.0, dict(cpu)), (10.0, {"n0": 8})], "CPU sample at t=10.0 has no free count"),
        ([(0.0, {**cpu, "zz": 1}), (10.0, dict(cpu))], "CPU sample at t=0.0 has a free count"),
    ):
        with pytest.raises(ValueError, match=match):
            structural_fragmentation_any(gpu, cpu, {}, gpu_s, cpu_s)


# ---- the any-resource variant ----------------------------------------------


def test_variant_also_carves_nodes_whose_cpus_are_busy() -> None:
    """All GPUs free on n0 but a CPU-only job on it: carved only in the variant."""
    gpu = {"n0": 4, "n1": 4}
    cpu = {"n0": 8, "n1": 8}
    gpu_s = [(0.0, {"n0": 4, "n1": 4}), (10.0, {"n0": 4, "n1": 4})]
    cpu_s = [(0.0, {"n0": 2, "n1": 8}), (10.0, {"n0": 8, "n1": 8})]
    assert structural_fragmentation(gpu, {}, gpu_s).rate("node") == 0.0
    variant = structural_fragmentation_any(gpu, cpu, {}, gpu_s, cpu_s)
    assert variant.rate("node") == 40 / 80
    # With GPU-only carving the variant reduces to the base definition.
    same_cpu = [(0.0, dict(cpu)), (10.0, dict(cpu))]
    samples = GOLDEN_SAMPLES
    cpu_caps = dict.fromkeys(GOLDEN_CAPACITY, 1)
    idle_cpu = [(t, dict.fromkeys(GOLDEN_CAPACITY, 1)) for t, _ in samples]
    assert structural_fragmentation_any(
        GOLDEN_CAPACITY, cpu_caps, GOLDEN_DOMAINS, samples, idle_cpu
    ) == structural_fragmentation(GOLDEN_CAPACITY, GOLDEN_DOMAINS, samples)
    with pytest.raises(ValueError, match="share timestamps"):
        structural_fragmentation_any(gpu, cpu, {}, gpu_s, same_cpu[:1])


# ---- on the simulator --------------------------------------------------------


def _hom_cluster() -> Cluster:
    """4 × 8-GPU nodes, two racks of two, one switch."""
    nodes = [
        Node(i, cpus=8, gpus=8, rack=f"rack-{i // 2}", switch="switch-0") for i in range(4)
    ]
    return Cluster(nodes, name="hom", topology_declared=True)


def test_homogeneous_fleet_running_only_whole_node_jobs_has_zero_node_level() -> None:
    """The control. Whole-node jobs can never leave a node partly allocated."""
    jobs = [
        job(i, 30.0 * i, 600.0 + 97.0 * (i % 5), nodes=1 + i % 3, gpus_per_node=8)
        for i in range(1, 40)
    ]
    cluster = _hom_cluster()
    m = compute(simulate(jobs, cluster), cluster)
    assert all(j.end_time is not None for j in jobs)
    assert m.frag_c_node == 0.0
    # Not a vacuous zero: whole nodes were busy and whole racks were carved.
    assert m.frag_c_rack > 0.0
    assert m.frag_c_node <= m.frag_c_rack <= m.frag_c_switch


def test_sub_node_jobs_carve_nodes_and_levels_stay_monotone() -> None:
    cluster = _hom_cluster()
    jobs = [job(i, 20.0 * i, 400.0 + 31.0 * (i % 7), gpus_per_node=1 + i % 4) for i in range(60)]
    m = compute(simulate(jobs, cluster), cluster)
    assert 0.0 < m.frag_c_node <= m.frag_c_rack <= m.frag_c_switch <= 1.0


def test_an_idle_simulated_fleet_reads_zero() -> None:
    """GPU-less jobs never carve a GPU domain."""
    cluster = _hom_cluster()
    m = compute(simulate([job(1, 5.0, 100.0)], cluster), cluster)
    assert (m.frag_c_node, m.frag_c_rack, m.frag_c_switch) == (0.0, 0.0, 0.0)
    assert m.frag_c_any_node > 0.0  # the CPU job did carve its node


def test_samples_start_at_trace_zero_and_end_at_the_horizon() -> None:
    cluster = Cluster.homogeneous(2, 4, 2)
    jobs = [job(1, 50.0, 100.0, gpus_per_node=1), job(2, 60.0, 10.0, gpus_per_node=2)]
    r = simulate(jobs, cluster)
    assert r.node_samples[0] == (0.0, (2, 2), (4, 4))
    assert r.node_samples[1] == (50.0, (1, 2), (3, 4))
    assert r.horizon == 150.0 == r.node_samples[-1][0]
    assert r.node_samples[-1][1:] == ((2, 2), (4, 4))
    times = [t for t, _, _ in r.node_samples]
    assert times == sorted(times)
    view = r.free_gpu_samples()
    assert view[1] == (50.0, {"node-0": 1, "node-1": 2})
    assert list(view[0:1]) == [(0.0, {"node-0": 2, "node-1": 2})]


def test_synthetic_workload_levels_are_monotone() -> None:
    jobs = generate(WorkloadProfile(job_count=80), seed=4)
    cluster = Cluster.homogeneous(16, 8, 2)
    m = compute(simulate(jobs, cluster), cluster)
    assert m.frag_c_node <= m.frag_c_rack <= m.frag_c_switch
    assert m.frag_c_cpu_node <= m.frag_c_cpu_rack <= m.frag_c_cpu_switch


def test_job_type_is_the_models() -> None:
    # Guards the factory: jobs here carry GPUs per node, not per job.
    assert job(1, 0, 1, nodes=2, gpus_per_node=3).total_gpus == 6
    assert isinstance(job(1, 0, 1), Job)
