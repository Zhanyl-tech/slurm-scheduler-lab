"""Heterogeneous clusters from the k8s lab's fleet YAML.

The schema is k8s-gpu-scheduler-lab's (`src/k8slab/fleet.py`, `fleets/*.yaml`,
docs/metrics.md "Topology and placement"), so one file describes the hardware
for both substrates:

    name: default                  # optional; defaults to the file stem
    topology:                      # optional
      racksPerSwitch: 2            # int >= 1; default: every rack on one switch
    nodeClasses:                   # required, non-empty
      - name: dgx8                 # required, unique
        count: 12                  # required, >= 1
        gpus: 8                    # required, >= 0
        cpus: 96                   # default 32
        memoryGi: 1024             # default 256 (parsed, not modelled here)
        nvlink: true               # default false; true/false only
        nodesPerRack: 4            # default: the whole class in one rack

Unknown keys inside a node class or `topology` are rejected, as the k8s loader
rejects them: a misspelt `nodesPerRak` must not fall back to a default layout.
Every topology key is optional, so a fleet file written before the k8s lab
added topology still loads (flat layout, `topology_declared` False).

Racks and switches are derived exactly as `k8slab.topology.derive()` does:
node classes in file order, nodes `<class>-0`, `<class>-1`, ...; each class
fills its own racks `nodesPerRack` at a time (racks never mix classes), racks
are numbered `rack-0, rack-1, ...` across the fleet, and consecutive racks are
grouped `racksPerSwitch` at a time into `switch-0, ...`. Node ids follow the
same order, so this simulator's first-fit walks nodes in rack order. (The k8s
lab's degenerate baselines walk them in lexicographic name order instead;
that is a difference between the schedulers, not between the fleets.)

The topology is a declared scenario, as in the k8s lab: node selection here
ignores it, and only placement and Definition C metrics read it.

No PyYAML. This repo has no runtime dependencies, and the fleet files use a
small block-style subset, so `_parse_yaml_subset()` reads that subset and
refuses anything else (flow collections, anchors, tags, block scalars,
multiple documents) with an error rather than guessing. The plain scalars
PyYAML's resolver reads as booleans besides true/false (`yes`, `No`, `ON`,
`off`, ...: exactly the spellings in `_PYYAML_ONLY_BOOLS`) are refused too, so
a file never means one thing here and another in the k8s lab. Anything else,
`y` and `n` included, is a string, as it is to PyYAML.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import Cluster, Node

_CLASS_KEYS = {"name", "count", "gpus", "cpus", "memoryGi", "nvlink", "nodesPerRack"}
_CLASS_TOPOLOGY_KEYS = {"nvlink", "nodesPerRack"}
_TOPOLOGY_KEYS = {"racksPerSwitch"}


@dataclass(frozen=True)
class NodeClass:
    name: str
    count: int
    gpus: int
    cpus: int = 32
    memory_gi: int = 256
    nvlink: bool = False
    nodes_per_rack: int | None = None

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"node class {self.name!r}: count must be >= 1")
        if self.gpus < 0:
            raise ValueError(f"node class {self.name!r}: gpus must be >= 0")
        if self.cpus < 1:
            raise ValueError(f"node class {self.name!r}: cpus must be >= 1")
        if self.nodes_per_rack is not None and self.nodes_per_rack < 1:
            raise ValueError(f"node class {self.name!r}: nodesPerRack must be >= 1")


@dataclass(frozen=True)
class Fleet:
    name: str
    classes: tuple[NodeClass, ...]
    racks_per_switch: int | None = None
    #: True when the file declared any topology key.
    topology_declared: bool = False

    def __post_init__(self) -> None:
        if self.racks_per_switch is not None and self.racks_per_switch < 1:
            raise ValueError(f"fleet {self.name!r}: racksPerSwitch must be >= 1")

    @property
    def total_nodes(self) -> int:
        return sum(c.count for c in self.classes)

    @property
    def total_gpus(self) -> int:
        return sum(c.count * c.gpus for c in self.classes)

    def build_cluster(self) -> Cluster:
        """A fresh, idle `Cluster` with every node's rack and switch derived."""
        nodes: list[Node] = []
        placed: list[tuple[NodeClass, int, int]] = []  # (class, ordinal, rack)
        next_rack = 0
        for cls in self.classes:
            per_rack = cls.nodes_per_rack or cls.count
            for i in range(cls.count):
                placed.append((cls, i, next_rack + i // per_rack))
            next_rack += -(-cls.count // per_rack)  # ceil division
        per_switch = self.racks_per_switch or max(1, next_rack)
        for node_id, (cls, i, rack) in enumerate(placed):
            nodes.append(
                Node(
                    node_id=node_id,
                    cpus=cls.cpus,
                    gpus=cls.gpus,
                    name=f"{cls.name}-{i}",
                    node_class=cls.name,
                    rack=f"rack-{rack}",
                    switch=f"switch-{rack // per_switch}",
                    nvlink=cls.nvlink,
                )
            )
        return Cluster(nodes, name=self.name, topology_declared=self.topology_declared)

    def describe(self) -> str:
        parts = [f"{c.count}×{c.name} ({c.cpus} CPU / {c.gpus} GPU)" for c in self.classes]
        return (
            f"fleet {self.name}: {self.total_nodes} nodes, {self.total_gpus} GPUs · "
            + ", ".join(parts)
        )


def load(path: str | Path) -> Fleet:
    """Load and validate a fleet file (the k8s lab's schema; see module docstring)."""
    raw = _parse_yaml_subset(Path(path).read_text(encoding="utf-8"), str(path))
    return from_mapping(raw, default_name=Path(path).stem, source=str(path))


def from_mapping(raw: Any, default_name: str = "fleet", source: str = "fleet") -> Fleet:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected a mapping at the top level")
    name = str(raw.get("name") or default_name)
    classes_raw = raw.get("nodeClasses")
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError(f"{source}: 'nodeClasses' must be a non-empty list")

    declared = False
    classes: list[NodeClass] = []
    for entry in classes_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: each nodeClass must be a mapping")
        missing = {"name", "count", "gpus"} - set(entry)
        if missing:
            raise ValueError(f"{source}: nodeClass missing {sorted(missing)}")
        unknown = set(entry) - _CLASS_KEYS
        if unknown:
            raise ValueError(f"{source}: nodeClass has unknown key(s) {sorted(unknown)}")
        nvlink = entry.get("nvlink", False)
        if not isinstance(nvlink, bool):
            raise ValueError(f"{source}: nodeClass {entry['name']!r}: nvlink must be true/false")
        per_rack = entry.get("nodesPerRack")
        declared = declared or bool(_CLASS_TOPOLOGY_KEYS & set(entry))
        classes.append(
            NodeClass(
                name=str(entry["name"]),
                count=_int(entry["count"], "count", source),
                gpus=_int(entry["gpus"], "gpus", source),
                cpus=_int(entry.get("cpus", 32), "cpus", source),
                memory_gi=_int(entry.get("memoryGi", 256), "memoryGi", source),
                nvlink=nvlink,
                nodes_per_rack=None if per_rack is None else _int(per_rack, "nodesPerRack", source),
            )
        )
    if len({c.name for c in classes}) != len(classes):
        raise ValueError(f"{source}: duplicate nodeClass names")

    racks_per_switch: int | None = None
    topo_raw = raw.get("topology")
    if topo_raw is not None:
        if not isinstance(topo_raw, dict):
            raise ValueError(f"{source}: 'topology' must be a mapping")
        unknown = set(topo_raw) - _TOPOLOGY_KEYS
        if unknown:
            raise ValueError(f"{source}: topology has unknown key(s) {sorted(unknown)}")
        if "racksPerSwitch" in topo_raw:
            racks_per_switch = _int(topo_raw["racksPerSwitch"], "racksPerSwitch", source)
        declared = True

    return Fleet(
        name=name,
        classes=tuple(classes),
        racks_per_switch=racks_per_switch,
        topology_declared=declared,
    )


def _int(value: Any, key: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{source}: {key} must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{source}: {key} must be an integer, got {value!r}") from None


# ── A strict subset of block-style YAML ─────────────────────────────────────

#: Plain scalars PyYAML resolves to a bool, other than the true/false
#: spellings read below. Its resolver (yaml/resolver.py, PyYAML 6.0.3, checked
#: in the k8s lab's venv) matches exactly
#: `yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF`,
#: case-sensitively. The YAML 1.1 type spec also lists y/Y/n/N, but PyYAML
#: does not implement them: `yaml.safe_load("k: y")` gives the string "y", so
#: a node class named `y` or `n` loads in the k8s lab and must load here.
_PYYAML_ONLY_BOOLS = frozenset(
    {"yes", "Yes", "YES", "no", "No", "NO", "on", "On", "ON", "off", "Off", "OFF"}
)
_INT = re.compile(r"^[-+]?(0|[1-9][0-9]*)$")
_FLOAT = re.compile(r"^[-+]?([0-9]+\.[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$")
_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*:(\s+|$)(.*)$")


def _strip_comment(text: str) -> str:
    """Drop a ` #` comment that is not inside quotes."""
    quote: str | None = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1] in " \t"):
            return text[:i].rstrip()
    return text.rstrip()


def _scalar(text: str, where: str) -> Any:
    if text == "" or text in {"~", "null", "Null", "NULL"}:
        return None
    if text[0] in "'\"":
        if len(text) < 2 or text[-1] != text[0]:
            raise ValueError(f"{where}: unterminated quoted string {text!r}")
        body = text[1:-1]
        if text[0] == "'":
            return body.replace("''", "'")
        if "\\" in body:
            raise ValueError(f"{where}: escape sequences are not supported in {text!r}")
        return body
    if text[0] in "[{&*!|>%@`":
        raise ValueError(
            f"{where}: {text!r} uses YAML features this loader does not read "
            "(flow collections, anchors, tags, block scalars); write it in block style"
        )
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text in _PYYAML_ONLY_BOOLS:
        raise ValueError(
            f"{where}: {text!r} is a boolean to PyYAML (which the k8s lab uses) and a "
            "string to YAML 1.2; write true or false, or quote it for a string"
        )
    if _INT.match(text):
        return int(text)
    octal = re.match(r"^[-+]?0[0-9_]+$", text)
    underscored = "_" in text and re.match(r"^[-+]?[0-9_]+$", text)
    if octal or underscored:
        raise ValueError(f"{where}: {text!r} is ambiguous (octal or underscores); quote it")
    if _FLOAT.match(text):
        return float(text)
    return text


def _parse_yaml_subset(text: str, source: str = "yaml") -> Any:
    """Block mappings, block sequences, scalars and comments. Nothing else."""
    lines: list[tuple[int, int, str]] = []  # (line number, indent, content)
    for number, raw in enumerate(text.splitlines(), start=1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ValueError(f"{source}:{number}: tabs are not allowed in indentation")
        content = _strip_comment(raw)
        if not content.strip():
            continue
        stripped = content.strip()
        if stripped in {"---", "..."} or stripped.startswith("%"):
            if stripped == "---" and not lines:
                continue  # a single leading document marker is harmless
            raise ValueError(f"{source}:{number}: multiple documents are not supported")
        lines.append((number, len(content) - len(content.lstrip(" ")), stripped))
    if not lines:
        return None
    value, end = _parse_block(lines, 0, lines[0][1], source)
    if end != len(lines):
        number = lines[end][0]
        raise ValueError(f"{source}:{number}: unexpected indentation")
    return value


def _parse_block(
    lines: list[tuple[int, int, str]], i: int, indent: int, source: str
) -> tuple[Any, int]:
    number, _, content = lines[i]
    if content == "-" or content.startswith("- "):
        return _parse_sequence(lines, i, indent, source)
    if _KEY.match(content):
        return _parse_mapping(lines, i, indent, source)
    if i + 1 < len(lines) and lines[i + 1][1] >= indent:
        raise ValueError(f"{source}:{number}: expected a mapping or a list")
    return _scalar(content, f"{source}:{number}"), i + 1


def _parse_mapping(
    lines: list[tuple[int, int, str]], i: int, indent: int, source: str
) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while i < len(lines):
        number, ind, content = lines[i]
        if ind < indent:
            break
        if ind > indent:
            raise ValueError(f"{source}:{number}: unexpected indentation")
        match = _KEY.match(content)
        if match is None:
            raise ValueError(f"{source}:{number}: expected 'key: value', got {content!r}")
        key, rest = match.group(1), match.group(3).strip()
        if key in out:
            raise ValueError(f"{source}:{number}: duplicate key {key!r}")
        i += 1
        if rest:
            out[key] = _scalar(rest, f"{source}:{number}")
        elif i < len(lines) and (
            lines[i][1] > indent
            or (lines[i][1] == indent and lines[i][2].startswith("-"))
        ):
            out[key], i = _parse_block(lines, i, lines[i][1], source)
        else:
            out[key] = None
    return out, i


def _parse_sequence(
    lines: list[tuple[int, int, str]], i: int, indent: int, source: str
) -> tuple[list[Any], int]:
    out: list[Any] = []
    while i < len(lines):
        number, ind, content = lines[i]
        if ind < indent or not (content == "-" or content.startswith("- ")):
            if ind > indent:
                raise ValueError(f"{source}:{number}: unexpected indentation")
            break
        if ind > indent:
            raise ValueError(f"{source}:{number}: unexpected indentation")
        rest = content[1:].strip()
        if not rest:
            i += 1
            if i < len(lines) and lines[i][1] > indent:
                item, i = _parse_block(lines, i, lines[i][1], source)
            else:
                item = None
            out.append(item)
            continue
        # "- key: value" starts a mapping whose keys sit at the dash's content
        # column; re-read this line as that mapping's first line.
        item_indent = ind + (len(content) - len(rest))
        lines[i] = (number, item_indent, rest)
        if _KEY.match(rest):
            item, i = _parse_mapping(lines, i, item_indent, source)
        elif rest == "-" or rest.startswith("- "):
            item, i = _parse_sequence(lines, i, item_indent, source)
        else:
            item, i = _scalar(rest, f"{source}:{number}"), i + 1
        out.append(item)
    return out, i
