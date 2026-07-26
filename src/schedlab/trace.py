"""Job traces: synthetic generation, and parsing real `sacct` output.

Synthetic traces are seeded, so a weight sweep compares policies against an
identical workload rather than against noise.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass

from .model import Job

SECONDS_PER_HOUR = 3600.0


@dataclass
class WorkloadProfile:
    """Shape of a synthetic workload.

    Defaults approximate a mixed research cluster: mostly short single-node
    work, a long tail of large jobs, and users who over-request wall-clock by
    roughly 3x — which is what the published trace studies keep finding.
    """

    job_count: int = 400
    #: Mean seconds between arrivals (Poisson process).
    arrival_interval: float = 90.0
    #: Log-normal runtime parameters, in log-seconds.
    runtime_mu: float = 6.5
    runtime_sigma: float = 1.4
    max_runtime: float = 8 * SECONDS_PER_HOUR
    #: Node counts are drawn from powers of two, weighted toward small.
    node_choices: tuple[int, ...] = (1, 1, 1, 1, 2, 2, 4, 8, 16)
    cpus_per_node: int = 8
    gpu_fraction: float = 0.3
    gpus_per_node: int = 2
    #: Multiplier applied to true runtime to get the requested time limit.
    request_padding: float = 3.0
    padding_sigma: float = 1.0
    accounts: tuple[str, ...] = ("research", "trading", "infra")
    account_weights: tuple[float, ...] = (0.5, 0.35, 0.15)


def generate(profile: WorkloadProfile | None = None, seed: int = 0) -> list[Job]:
    profile = profile or WorkloadProfile()
    rng = random.Random(seed)

    jobs: list[Job] = []
    now = 0.0

    for job_id in range(1, profile.job_count + 1):
        now += rng.expovariate(1.0 / profile.arrival_interval)

        duration = min(
            rng.lognormvariate(profile.runtime_mu, profile.runtime_sigma),
            profile.max_runtime,
        )
        # Users pad their limits, and pad inconsistently. Never below 1.05x, or
        # jobs would be killed for exceeding their own request.
        padding = max(1.05, rng.gauss(profile.request_padding, profile.padding_sigma))

        wants_gpu = rng.random() < profile.gpu_fraction

        jobs.append(
            Job(
                job_id=job_id,
                account=rng.choices(profile.accounts, weights=profile.account_weights)[0],
                submit_time=now,
                duration=duration,
                time_limit=duration * padding,
                nodes=rng.choice(profile.node_choices),
                cpus_per_node=profile.cpus_per_node,
                gpus_per_node=profile.gpus_per_node if wants_gpu else 0,
                qos_factor=rng.choice([0.0, 0.0, 0.0, 0.5, 1.0]),
            )
        )

    return jobs


def _slurm_time(text: str) -> float:
    """Parse a Slurm time field: `[days-]HH:MM:SS`, or `UNLIMITED`."""
    text = text.strip()
    if not text or text.upper() in {"UNLIMITED", "PARTITION_LIMIT", "INVALID"}:
        return 0.0

    days = 0.0
    if "-" in text:
        day_part, _, text = text.partition("-")
        days = float(day_part)

    parts = text.split(":")
    while len(parts) < 3:
        parts.insert(0, "0")
    hours, minutes, seconds = (float(p or 0) for p in parts[-3:])
    return days * 86_400 + hours * 3600 + minutes * 60 + seconds


def _tres_gpus(tres: str) -> int:
    """Pull the GPU count out of an AllocTRES / ReqTRES string."""
    for part in tres.split(","):
        if "gres/gpu=" in part:
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return 0
    return 0


def from_sacct(path: str) -> list[Job]:
    """Load a trace from `sacct` pipe-delimited output.

    Produce the input with:

        sacct -a -X --parsable2 --starttime=now-30days \\
              --format=JobID,Account,Submit,Start,End,Elapsed,Timelimit,NNodes,ReqCPUS,ReqTRES

    Jobs that never started (cancelled while pending) are skipped — they carry
    no runtime to replay.
    """
    jobs: list[Job] = []

    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        base: float | None = None

        rows = list(reader)
        submits: list[float] = []
        for row in rows:
            try:
                submits.append(_timestamp(row.get("Submit", "")))
            except ValueError:
                submits.append(0.0)
        base = min((s for s in submits if s > 0), default=0.0)

        for row, submit in zip(rows, submits):
            elapsed = _slurm_time(row.get("Elapsed", ""))
            if elapsed <= 0:
                continue

            limit = _slurm_time(row.get("Timelimit", "")) or elapsed
            nodes = int(row.get("NNodes") or 1)
            req_cpus = int(row.get("ReqCPUS") or nodes)

            raw_id = (row.get("JobID") or "0").split(".")[0].split("_")[0]
            try:
                job_id = int(raw_id)
            except ValueError:
                job_id = len(jobs) + 1

            jobs.append(
                Job(
                    job_id=job_id,
                    account=(row.get("Account") or "unknown").strip(),
                    submit_time=submit - base,
                    duration=elapsed,
                    time_limit=max(limit, elapsed),
                    nodes=nodes,
                    cpus_per_node=max(1, req_cpus // max(nodes, 1)),
                    gpus_per_node=_tres_gpus(row.get("ReqTRES", "")) // max(nodes, 1),
                )
            )

    if not jobs:
        raise ValueError(f"no completed jobs found in {path}")
    return jobs


def _timestamp(text: str) -> float:
    """Parse Slurm's ISO-ish timestamp (`2026-07-26T09:31:04`) to epoch seconds."""
    from datetime import datetime

    text = text.strip()
    if not text or text.upper() in {"UNKNOWN", "NONE"}:
        raise ValueError("no timestamp")
    return datetime.fromisoformat(text).timestamp()
