"""B2: heterogeneous clusters from the k8s lab's fleet YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from schedlab import fleet
from schedlab.fleet import _parse_yaml_subset
from schedlab.metrics import compute
from schedlab.model import Cluster, Node
from schedlab.simulate import simulate
from tests._factory import job

# A copy of k8s-gpu-scheduler-lab's fleets/default.yaml (comments trimmed),
# so the test does not depend on the sibling checkout.
DEFAULT_FLEET = """\
# The default fleet: deliberately heterogeneous.
name: default
topology:
  racksPerSwitch: 2
nodeClasses:
  - name: dgx8
    count: 12
    gpus: 8
    cpus: 96
    memoryGi: 1024
    nvlink: true
    nodesPerRack: 4
  - name: mid4
    count: 8
    gpus: 4
    cpus: 48
    memoryGi: 512
    nvlink: false
    nodesPerRack: 4
  - name: edge2
    count: 6
    gpus: 2
    cpus: 24
    memoryGi: 128
    nvlink: false
    nodesPerRack: 6
"""

K8S_FLEETS = Path(__file__).resolve().parents[2] / "k8s-gpu-scheduler-lab" / "fleets"


def _write(tmp_path: Path, text: str, name: str = "fleet.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_default_fleet_layout_matches_the_k8s_derivation(tmp_path):
    """The layout the k8s fleet file's comment spells out, node by node."""
    f = fleet.load(_write(tmp_path, DEFAULT_FLEET))
    assert (f.name, f.total_nodes, f.total_gpus, f.topology_declared) == ("default", 26, 140, True)
    cluster = f.build_cluster()
    by_name = {n.name: n for n in cluster.nodes}
    assert [n.node_id for n in cluster.nodes] == list(range(26))
    # dgx8 -> rack-0..rack-2 (4 nodes each), mid4 -> rack-3..4, edge2 -> rack-5.
    assert by_name["dgx8-0"].rack == "rack-0" and by_name["dgx8-11"].rack == "rack-2"
    assert by_name["mid4-0"].rack == "rack-3" and by_name["mid4-7"].rack == "rack-4"
    assert {by_name[f"edge2-{i}"].rack for i in range(6)} == {"rack-5"}
    # switch-0 = rack-0,1  switch-1 = rack-2,3  switch-2 = rack-4,5
    assert by_name["dgx8-4"].switch == "switch-0" and by_name["dgx8-8"].switch == "switch-1"
    assert by_name["mid4-0"].switch == "switch-1" and by_name["mid4-4"].switch == "switch-2"
    assert by_name["edge2-0"].switch == "switch-2"
    assert (by_name["dgx8-3"].gpus, by_name["dgx8-3"].cpus, by_name["dgx8-3"].nvlink) == (
        8,
        96,
        True,
    )
    assert cluster.max_node_gpus == 8 and cluster.topology_declared


@pytest.mark.parametrize("name", ["default.yaml", "homogeneous.yaml"])
def test_shipped_k8s_fleet_files_load(name):
    path = K8S_FLEETS / name
    if not path.exists():
        pytest.skip("k8s-gpu-scheduler-lab is not checked out next to this repo")
    f = fleet.load(path)
    assert f.total_nodes == sum(c.count for c in f.classes)
    assert len(f.build_cluster().nodes) == f.total_nodes
    # The YAML subset reads the file into exactly what PyYAML (the k8s lab's
    # parser) reads; PyYAML is in the dev extra, not a runtime dependency.
    yaml = pytest.importorskip("yaml")
    text = path.read_text(encoding="utf-8")
    assert _parse_yaml_subset(text) == yaml.safe_load(text)


@pytest.mark.parametrize("name", ["default.yaml", "homogeneous.yaml"])
def test_shipped_fleets_match_the_k8s_labs_own_derivation(name):
    """Every node's name, rack, switch, GPU count and NVLink flag, compared
    with `k8slab.topology.derive()` over the k8s lab's own loader. This is
    what the README's "derived exactly as" claim rests on; it needs the k8s
    lab importable (its `src` on PYTHONPATH, and PyYAML) and its fleets
    checked out next to this repo."""
    path = K8S_FLEETS / name
    if not path.exists():
        pytest.skip("k8s-gpu-scheduler-lab is not checked out next to this repo")
    k8s_fleet = pytest.importorskip("k8slab.fleet")
    k8s_topology = pytest.importorskip("k8slab.topology")
    theirs = k8s_topology.derive(k8s_fleet.load(path))
    ours = fleet.load(path).build_cluster()
    assert [n.name for n in ours.nodes] == list(theirs.nodes)
    for node in ours.nodes:
        t = theirs.node(node.name)
        assert (node.rack, node.switch, node.gpus, node.nvlink, node.node_class) == (
            t.rack, t.switch, t.gpus, t.nvlink, t.node_class,
        )  # fmt: skip
    assert ours.topology_declared == theirs.declared


def test_a_partial_last_rack_is_not_shared_with_the_next_class(tmp_path):
    """Racks never mix classes: a class of 5 at nodesPerRack 2 fills rack-0,
    rack-1 and half of rack-2, and the next class starts at rack-3. The
    shipped fleets divide evenly, so floor division passed every test."""
    f = fleet.load(
        _write(
            tmp_path,
            "topology:\n  racksPerSwitch: 2\nnodeClasses:\n"
            "  - name: a\n    count: 5\n    gpus: 1\n    nodesPerRack: 2\n"
            "  - name: b\n    count: 2\n    gpus: 1\n    nodesPerRack: 2\n",
        )
    )
    racks = {n.name: n.rack for n in f.build_cluster().nodes}
    assert racks == {
        "a-0": "rack-0", "a-1": "rack-0", "a-2": "rack-1", "a-3": "rack-1",
        "a-4": "rack-2", "b-0": "rack-3", "b-1": "rack-3",
    }  # fmt: skip
    switches = {n.name: n.switch for n in f.build_cluster().nodes}
    assert switches["a-4"] == "switch-1" and switches["b-0"] == "switch-1"


def test_topology_keys_are_optional(tmp_path):
    """A fleet file written before the k8s lab added topology still loads."""
    f = fleet.load(
        _write(
            tmp_path,
            "nodeClasses:\n  - name: a\n    count: 2\n    gpus: 4\n"
            "  - name: b\n    count: 1\n    gpus: 0\n",
            "old.yaml",
        )
    )
    assert f.name == "old" and not f.topology_declared
    cluster = f.build_cluster()
    # Flat defaults: one rack per class, one switch.
    assert [n.rack for n in cluster.nodes] == ["rack-0", "rack-0", "rack-1"]
    assert {n.switch for n in cluster.nodes} == {"switch-0"}
    assert [n.cpus for n in cluster.nodes] == [32, 32, 32]


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("name: x\nnodeClasses:\n", "non-empty"),
        ("nodeClasses:\n  - name: a\n    count: 1\n", "missing"),
        ("nodeClasses:\n  - name: a\n    count: 1\n    gpus: 1\n    nodesPerRak: 2\n", "unknown"),
        ("nodeClasses:\n  - name: a\n    count: 0\n    gpus: 1\n", "count"),
        ("nodeClasses:\n  - name: a\n    count: 1\n    gpus: 1\n    nvlink: 1\n", "nvlink"),
        (
            "nodeClasses:\n  - name: a\n    count: 1\n    gpus: 1\n    nvlink: yes\n",
            "boolean to PyYAML",
        ),
        (
            "nodeClasses:\n  - {name: a, count: 1, gpus: 1}\n",
            "does not read",
        ),
        (
            "topology:\n  racksPerSwitch: 0\n"
            "nodeClasses:\n  - name: a\n    count: 1\n    gpus: 1\n",
            "racksPerSwitch",
        ),
        (
            "topology:\n  spine: 2\nnodeClasses:\n  - name: a\n    count: 1\n    gpus: 1\n",
            "unknown",
        ),
        (
            "nodeClasses:\n  - name: a\n    count: 1\n    gpus: 1\n"
            "  - name: a\n    count: 1\n    gpus: 1\n",
            "duplicate",
        ),
        ("nodeClasses:\n  - name: a\n    count: many\n    gpus: 1\n", "integer"),
    ],
)
def test_bad_fleets_are_refused(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        fleet.load(_write(tmp_path, text))


def test_yaml_subset_reads_what_the_fleet_files_use():
    doc = _parse_yaml_subset(
        "# comment\n---\nname: 'x # y'\nn: 3\nf: 1.5\nb: false\nz: ~\nl:\n- 1\n- two\n"
        "m:\n  k: \"v\"  # trailing\n"
    )
    assert doc == {
        "name": "x # y",
        "n": 3,
        "f": 1.5,
        "b": False,
        "z": None,
        "l": [1, "two"],
        "m": {"k": "v"},
    }
    for bad in ("a: 1\n---\nb: 2\n", "k: 010\n", "k: |\n  x\n", "a:\n\tb: 1\n", "a: 1\na: 2\n"):
        with pytest.raises(ValueError):
            _parse_yaml_subset(bad)


#: PyYAML 6.0.3's bool resolver, as `yaml.safe_load(f"k: {word}")` reads each
#: word (checked in the k8s lab's venv): these twelve besides true/false.
_PYYAML_BOOLS = ("yes", "Yes", "YES", "no", "No", "NO", "on", "On", "ON", "off", "Off", "OFF")
#: Strings to PyYAML, although the YAML 1.1 spec lists y/n or they differ
#: from a bool word only in case.
_PYYAML_STRINGS = ("y", "Y", "n", "N", "yEs", "oN", "nO", "oFF", "tRUE")


@pytest.mark.parametrize("word", _PYYAML_BOOLS)
def test_words_pyyaml_reads_as_booleans_are_refused(word):
    with pytest.raises(ValueError, match="boolean to PyYAML"):
        _parse_yaml_subset(f"k: {word}\n")
    assert _parse_yaml_subset(f"k: '{word}'\n") == {"k": word}  # quoted, a string


@pytest.mark.parametrize("word", _PYYAML_STRINGS)
def test_words_pyyaml_reads_as_strings_load_as_strings(word):
    """They were refused as "a boolean to YAML 1.1 (PyYAML ...)", which PyYAML
    does not do: a node class named `n` loads in the k8s lab."""
    assert _parse_yaml_subset(f"k: {word}\n") == {"k": word}


def test_a_node_class_named_n_loads(tmp_path):
    f = fleet.load(_write(tmp_path, "nodeClasses:\n  - name: n\n    count: 1\n    gpus: 1\n"))
    assert [n.name for n in f.build_cluster().nodes] == ["n-0"]


def test_scalars_match_pyyaml_when_it_is_importable():
    yaml = pytest.importorskip("yaml")
    for word in (*_PYYAML_BOOLS, *_PYYAML_STRINGS, "true", "False", "TRUE"):
        theirs = yaml.safe_load(f"k: {word}\n")["k"]
        if isinstance(theirs, bool) and word in _PYYAML_BOOLS:
            with pytest.raises(ValueError):
                _parse_yaml_subset(f"k: {word}\n")
        else:
            assert _parse_yaml_subset(f"k: {word}\n") == {"k": theirs}


def test_heterogeneous_cluster_places_jobs_only_where_they_fit(tmp_path):
    cluster = fleet.load(_write(tmp_path, DEFAULT_FLEET)).build_cluster()
    jobs = [
        job(1, 0, 100, nodes=12, gpus_per_node=8),  # every dgx8
        job(2, 1, 100, nodes=2, gpus_per_node=4),  # mid4s only while dgx8s are busy
        job(3, 2, 100, nodes=7, gpus_per_node=2),  # more than the 6 edge2s alone
        job(4, 3, 100, nodes=1, gpus_per_node=16),  # no node has 16 GPUs
    ]
    result = simulate(jobs, cluster)
    assert [j.job_id for j in result.unschedulable] == [4]
    names = {n.node_id: n.name for n in cluster.nodes}
    assert {names[n] for n in jobs[0].runs[0].nodes} == {f"dgx8-{i}" for i in range(12)}
    assert {names[n] for n in jobs[1].runs[0].nodes} == {"mid4-0", "mid4-1"}
    assert jobs[2].start_time == 2.0
    m = compute(result, cluster)
    assert m.total_gpus == 140 and 0.0 < m.gpu_utilization < 1.0


def test_cluster_domains_default_like_the_k8s_lab():
    cluster = Cluster(
        [
            Node(0, 4, 1, node_class="a"),
            Node(1, 4, 1, node_class="b"),
            Node(2, 4, 1, node_class="a"),
        ]
    )
    assert cluster.domains() == {
        "rack": {"node-0": "rack-0", "node-1": "rack-1", "node-2": "rack-0"},
        "switch": dict.fromkeys(("node-0", "node-1", "node-2"), "switch-0"),
    }
