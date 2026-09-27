"""Regenerate the README's S0 table ("Running S0", the contended default profile).

The traces come from the k8s lab, one per seed:

    k8slab trace --profile default --seed $s --out DIR/t$s.csv     # s = 0..4

Each S0 row runs the documented command once per seed, through `schedlab`'s
own entry point, and averages the five `--json` files:

    schedlab --k8s-trace DIR/t$s.csv --fleet FLEET --time-limit-model exact --json ...
    ... --time-limit-model padded --time-limit-factor 3
    ... --time-limit-model synthetic --seed $s

`--reference` adds the k8s lab's in-process reference-model rows (not
kube-scheduler): `k8slab.sim.run(fleet, jobs, config, seed=0)` then
`k8slab.metrics.compute` on the same CSVs, for D-fifo, D-random and
D-largest, plus D-random with `seed=s` on trace s. It needs `k8slab`
importable (run it with the k8s lab's venv, or its `src` and PyYAML on
PYTHONPATH), and it prints the k8s lab's commit and working-tree fingerprint,
so a row can be tied to the code that made it.

    python scripts/s0_table.py DIR --fleet ../k8s-gpu-scheduler-lab/fleets/default.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import statistics
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from schedlab import cli

SEEDS = range(5)
#: (label, extra flags given the seed)
MODELS: tuple[tuple[str, Callable[[int], list[str]]], ...] = (
    ("S0, exact limits", lambda s: ["--time-limit-model", "exact"]),
    ("S0, padded ×3", lambda s: ["--time-limit-model", "padded", "--time-limit-factor", "3"]),
    ("S0, synthetic", lambda s: ["--time-limit-model", "synthetic", "--seed", str(s)]),
)
#: (JSON key, scale, format) for the table's columns, in order.
COLUMNS = (
    ("utilization", 100.0, "{:.1f} %"),
    ("mean_wait", 1 / 60, "{:.1f} min"),
    ("p95_wait", 1 / 60, "{:.1f} min"),
    ("fragmentation_structural", 100.0, "{:.1f} %"),
    ("large_job_starvation_ratio", 1.0, "{:.2f}"),
)


def s0_run(trace: Path, fleet: str, extra: list[str]) -> dict[str, Any]:
    """One documented S0 command; its `--json` document."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "s0.json"
        argv = ["--k8s-trace", str(trace), "--fleet", fleet, *extra, "--json", str(out)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(argv)
        if code != 0:
            raise SystemExit(f"schedlab {' '.join(argv)} exited {code}")
        doc: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        return doc


def row(label: str, runs: list[dict[str, Any]], stranded: str) -> str:
    cells = [
        fmt.format(statistics.fmean(r[key] for r in runs) * scale) for key, scale, fmt in COLUMNS
    ]
    return f"| {label} | " + " | ".join(cells) + f" | {stranded} |"


def fingerprint(repo: Path) -> tuple[str, str]:
    """(HEAD, fingerprint): SHA-256 of `git diff HEAD --binary` followed by
    `shasum -a 256` lines of each untracked file in sorted order, first 12 hex
    digits. The recipe the README states."""

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout

    h = hashlib.sha256(git("diff", "HEAD", "--binary"))
    untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    for name in sorted(untracked):
        digest = hashlib.sha256((repo / name).read_bytes()).hexdigest()
        h.update(f"{digest}  {name}\n".encode())
    return git("rev-parse", "--short", "HEAD").decode().strip(), h.hexdigest()[:12]


def reference_rows(traces: list[Path], fleet: str) -> list[str]:
    from k8slab import fleet as k8s_fleet
    from k8slab import metrics as k8s_metrics
    from k8slab import sim as k8s_sim
    from k8slab import trace as k8s_trace

    flt = k8s_fleet.load(fleet)
    rows = []
    # (policy, whether the policy seed follows the trace seed; else 0)
    variants = [("D-fifo", False), ("D-random", False), ("D-largest", False), ("D-random", True)]
    for config, follows in variants:
        runs = []
        for s, path in zip(SEEDS, traces, strict=True):
            m = k8s_metrics.compute(
                k8s_sim.run(flt, k8s_trace.read_csv(path), config, seed=s if follows else 0)
            )
            runs.append(
                {key: getattr(m, key) for key, _, _ in COLUMNS}
                | {"gang_stranded_gpu_hours": m.gang_stranded_gpu_hours}
            )
        stranded = f"{statistics.fmean(r['gang_stranded_gpu_hours'] for r in runs):.1f}"
        label = f"k8s {config} (reference model)" + (", seed=s" if follows else "")
        rows.append(row(label, runs, stranded))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traces", type=Path, help="directory holding t0.csv ... t4.csv")
    parser.add_argument("--fleet", required=True, help="the k8s lab's fleets/default.yaml")
    parser.add_argument("--reference", action="store_true", help="add the k8s reference rows")
    args = parser.parse_args()
    traces = [args.traces / f"t{s}.csv" for s in SEEDS]

    print("| config (all: model, 5 seeds, mean) | GPU util | mean wait | p95 wait "
          "| frag C node | starvation × | gang stranded GPU-h |")  # fmt: skip
    print("| --- | --- | --- | --- | --- | --- | --- |")
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for label, extra in MODELS:
        runs = [s0_run(t, args.fleet, extra(s)) for s, t in zip(SEEDS, traces, strict=True)]
        per_seed[label] = runs
        print(row(label, runs, "0 (structural)"))
    if args.reference:
        for line in reference_rows(traces, args.fleet):
            print(line)

    print("\nper seed (mean wait min / starvation ×):")
    for label, runs in per_seed.items():
        cells = ", ".join(
            f"{r['mean_wait'] / 60:.1f} / {r['large_job_starvation_ratio']:.2f}" for r in runs
        )
        print(f"  {label}: {cells}")
    digests = [r["trace_digest"] for r in per_seed[MODELS[0][0]]]
    print(f"trace digests: {', '.join(digests)}")
    if args.reference:
        from k8slab import metrics as k8s_metrics

        # k8slab is a namespace package (no __init__.py): locate it by a module.
        repo = Path(k8s_metrics.__file__).resolve().parents[2]
        head, fp = fingerprint(repo)
        print(f"k8s lab: commit {head}, working-tree fingerprint {fp}")


if __name__ == "__main__":
    main()
