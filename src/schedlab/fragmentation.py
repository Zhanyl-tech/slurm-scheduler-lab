"""Fragmentation Definition C: structural, queue-independent, per topology level.

The same pure function as `k8slab.fragmentation.structural_fragmentation` in
the sibling k8s-gpu-scheduler-lab (its docs/metrics.md, "Definition C"), with
the same signature, the same validation and the same arithmetic in the same
order, so the two labs can be shown to agree on the shared golden vector in
`tests/test_fragmentation.py` rather than merely claimed to. Standard library
only; nothing else from this package is imported.

The definition. At level L (node, rack, switch), a level-L domain is
**carved** at time t if any node inside it has any GPU allocated (free <
capacity). The carved free GPUs are

    S_C^L(t) = sum of free GPUs on nodes whose level-L domain is carved

and

    frag_C^L = integral(S_C^L dt) / integral(total free GPUs dt),

0 when the denominator is 0. At the node level the domain is the node itself,
so a fully allocated node contributes 0 free GPUs and an idle node contributes
nothing carved. Samples are a step function: each sample holds until the next
sample's timestamp (left point), and the last sample only closes the final
interval.

Why it maps onto Slurm (the k8s lab's argument, which holds from this side
too): `--exclusive` needs a whole idle node, so frag_C^node is the share of
free GPU-time a whole-node request cannot use; with `topology/tree`, rack
plays the leaf switch.

One Slurm-specific gap, and the variant that covers it. A Slurm node is
exclusive-idle only if *nothing* is allocated on it — CPUs included. A node
whose GPUs are all free but whose CPUs run a CPU-only job is not carved under
the GPU definition, yet an `--exclusive` GPU job cannot have it. The base
definition is kept exactly (the golden values must hold in both labs), and
`structural_fragmentation_any` adds the variant in which a node is carved if
any GPU *or* CPU is allocated, still measuring free GPU-time. On a trace in
which every job asks for a GPU on every node it uses (the k8s traces), the
two agree; they differ only when CPU-only jobs share GPU nodes. The variant is
reported beside the base, never instead of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

NODE_LEVEL = "node"


@dataclass(frozen=True)
class StructuralFragmentation:
    """Numerators per level, and the shared denominator, in GPU-seconds."""

    #: Levels in the order given, finest first; always starts with "node".
    levels: tuple[str, ...]
    carved_gpu_seconds: dict[str, float]
    free_gpu_seconds: float

    def rate(self, level: str) -> float:
        """frag_C^level; 0.0 when there was no free GPU-time at all."""
        if level not in self.carved_gpu_seconds:
            raise KeyError(f"no level {level!r}; have {list(self.levels)}")
        if self.free_gpu_seconds <= 0:
            return 0.0
        return self.carved_gpu_seconds[level] / self.free_gpu_seconds

    def rates(self) -> dict[str, float]:
        return {level: self.rate(level) for level in self.levels}


def structural_fragmentation(
    capacity: Mapping[str, int],
    domains: Mapping[str, Mapping[str, str]],
    samples: Sequence[tuple[float, Mapping[str, int]]],
) -> StructuralFragmentation:
    """Definition C at the node level and at every level in `domains`.

    `capacity` is `{node: GPU capacity}` and defines the node set. `domains`
    is `{level: {node: domain id}}` for levels above the node, finest first;
    every node needs a domain at every level, and each level must nest in the
    next (non-nesting input is refused, because it could invert the
    node <= rack <= switch ordering). `samples` is `[(t, {node: free GPUs})]`
    in non-decreasing time order. Every sample, including the last one and
    any followed by a zero-length interval (neither is integrated), must give
    every node in `capacity`, and no other node, a count in `[0, capacity]`.

    Raises ValueError on any violation rather than guessing, and validates
    every sample before integrating anything, as the k8s lab's function does.
    """
    return _integrate(capacity, domains, samples, None)


def structural_fragmentation_any(
    capacity: Mapping[str, int],
    cpu_capacity: Mapping[str, int],
    domains: Mapping[str, Mapping[str, str]],
    samples: Sequence[tuple[float, Mapping[str, int]]],
    cpu_samples: Sequence[tuple[float, Mapping[str, int]]],
) -> StructuralFragmentation:
    """The variant: a node is carved if any GPU **or CPU** is allocated.

    Numerator and denominator are still free GPU-time. `cpu_samples` must
    have the same timestamps as `samples`, and is validated like it against
    `cpu_capacity` (every sample, every node, no other node).
    """
    if len(cpu_samples) != len(samples) or any(
        a[0] != b[0] for a, b in zip(samples, cpu_samples, strict=True)
    ):
        raise ValueError("GPU and CPU samples must share timestamps")
    return _integrate(capacity, domains, samples, (cpu_capacity, cpu_samples))


def _integrate(
    capacity: Mapping[str, int],
    domains: Mapping[str, Mapping[str, str]],
    samples: Sequence[tuple[float, Mapping[str, int]]],
    cpus: tuple[Mapping[str, int], Sequence[tuple[float, Mapping[str, int]]]] | None,
) -> StructuralFragmentation:
    if NODE_LEVEL in domains:
        raise ValueError("the node level is implicit; do not pass it in domains")
    nodes = sorted(capacity)
    caps = [int(capacity[n]) for n in nodes]
    if any(c < 0 for c in caps):
        raise ValueError("capacity must be >= 0 on every node")
    cpu_caps: list[int] | None = None
    if cpus is not None:
        missing_cpu = [n for n in nodes if n not in cpus[0]]
        if missing_cpu:
            raise ValueError(f"no CPU capacity for node(s) {missing_cpu[:5]}")
        cpu_caps = [int(cpus[0][n]) for n in nodes]

    # Domain ids -> dense ints per level, and the nesting check.
    level_names = tuple(domains)
    dom_idx: list[list[int]] = []
    for level in level_names:
        mapping = domains[level]
        missing = [n for n in nodes if n not in mapping]
        if missing:
            raise ValueError(f"level {level!r}: no domain for node(s) {missing[:5]}")
        extra = sorted(set(mapping) - set(capacity))
        if extra:
            raise ValueError(f"level {level!r}: domain given for unknown node(s) {extra[:5]}")
        ids: dict[str, int] = {}
        dom_idx.append([ids.setdefault(mapping[n], len(ids)) for n in nodes])
    for k in range(len(level_names) - 1):
        parent: dict[int, int] = {}
        for fine, coarse in zip(dom_idx[k], dom_idx[k + 1], strict=True):
            if parent.setdefault(fine, coarse) != coarse:
                raise ValueError(
                    f"level {level_names[k]!r} does not nest in {level_names[k + 1]!r}: "
                    f"one {level_names[k]} domain spans several {level_names[k + 1]} domains"
                )

    # Validate every sample before integrating anything, in the k8s lab's
    # order: key checks per sample, the range check per distinct state, then
    # time order. Samples that are never integrated (the last one, and any
    # followed by a zero-length interval) must be valid too. This port used
    # to read only samples that opened a positive-length interval, so a
    # malformed final sample, or a sample naming a node outside `capacity`,
    # passed silently where the k8s function refuses it.
    states = _states(samples, nodes, caps, "")
    busy_states: list[tuple[int, ...]] | None = None
    if cpus is not None and cpu_caps is not None:
        cpu_states = _states(cpus[1], nodes, cpu_caps, "CPU ")
        busy_states = [
            tuple(int(v < c) for v, c in zip(vals, cpu_caps, strict=True)) for vals in cpu_states
        ]
    for i in range(len(samples) - 1):
        t0, t1 = samples[i][0], samples[i + 1][0]
        if t1 < t0:
            raise ValueError(f"samples out of time order at index {i}: {t0} then {t1}")

    levels = (NODE_LEVEL, *level_names)
    carved = dict.fromkeys(levels, 0.0)
    free_total = 0.0
    # Identical consecutive states are common, so per-state sums are
    # memoised. Pure speed: the same arithmetic in the same order.
    memo: dict[tuple[int, ...], tuple[int, ...]] = {}

    for i in range(len(samples) - 1):
        dt = samples[i + 1][0] - samples[i][0]
        if dt == 0:
            continue
        vals = states[i]
        busy = busy_states[i] if busy_states is not None else ()
        key = vals + busy
        sums = memo.get(key)
        if sums is None:
            sums = _sums(vals, caps, dom_idx, busy)
            memo[key] = sums
        free_total += sums[0] * dt
        for level, s in zip(levels, sums[1:], strict=True):
            carved[level] += s * dt

    return StructuralFragmentation(
        levels=levels, carved_gpu_seconds=carved, free_gpu_seconds=free_total
    )


def _states(
    samples: Sequence[tuple[float, Mapping[str, int]]],
    nodes: list[str],
    caps: list[int],
    kind: str,
) -> list[tuple[int, ...]]:
    """Every sample's counts in `nodes` order, each one validated.

    `ValueError` if a sample misses a node, names a node outside the fleet,
    or has a count outside `[0, capacity]` (`k8slab.fragmentation._state` and
    `_check_range`, with the same messages). `kind` prefixes "CPU " for the
    variant's CPU samples.
    """
    states: list[tuple[int, ...]] = []
    checked: set[tuple[int, ...]] = set()
    for t, free in samples:
        try:
            vals = tuple(free[n] for n in nodes)
        except KeyError as exc:
            raise ValueError(f"{kind}sample at t={t} has no free count for node {exc}") from None
        if len(free) != len(nodes):
            extra = sorted(set(free) - set(nodes))
            raise ValueError(
                f"{kind}sample at t={t} has a free count for unknown node(s) {extra[:5]}"
            )
        states.append(vals)
    for (t, _), vals in zip(samples, states, strict=True):
        if vals in checked:
            continue
        for n, v, c in zip(nodes, vals, caps, strict=True):
            if v < 0 or v > c:
                raise ValueError(f"{kind}sample at t={t}: node {n} free {v} outside [0, {c}]")
        checked.add(vals)
    return states


def _sums(
    vals: tuple[int, ...],
    caps: list[int],
    dom_idx: list[list[int]],
    cpu_busy: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """(total free, S_C^node, S_C^<each level>) for one validated fleet state."""
    node_carved = [v < c for v, c in zip(vals, caps, strict=True)]
    if cpu_busy:
        node_carved = [g or bool(b) for g, b in zip(node_carved, cpu_busy, strict=True)]
    out = [sum(vals), sum(v for v, carved in zip(vals, node_carved, strict=True) if carved)]
    for idx in dom_idx:
        carved_domains = {d for d, carved in zip(idx, node_carved, strict=True) if carved}
        out.append(sum(v for v, d in zip(vals, idx, strict=True) if d in carved_domains))
    return tuple(out)
