"""Job traces: synthetic generation, and parsing real `sacct` output.

Synthetic traces are seeded, so a weight sweep compares policies against an
identical workload rather than against noise.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime

from .model import Job

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86_400.0

#: What Slurm's backfill plans a job with no finite time limit at, when its
#: partition's MaxTime is UNLIMITED too: `YEAR_MINUTES`, 365 days
#: (`_set_backfill_timelimits()`, src/plugins/sched/backfill/backfill.c
#: L2227-L2259; `YEAR_MINUTES` in src/common/slurm_protocol_defs.h L211;
#: SchedMD/slurm@9f9da53b4a7bc5b56062bc357f4f94e0d19c71cc). A running job with
#: an INFINITE limit gets the same year as its end time (`job_end_time_reset()`,
#: job_mgr.c L19402-L19414). With a finite partition MaxTime, backfill plans at
#: that instead: `--partition-max-time`.
UNLIMITED_PLANNING_LIMIT = 365 * SECONDS_PER_DAY

#: `Timelimit` values that carry no finite limit. sacct prints "UNLIMITED" for
#: INFINITE and "Partition_Limit" for NO_VAL, and leaves 0 blank
#: (src/sacct/print.c L2123-L2135, same commit); `INVALID` is kept from the
#: 0.1.0 parser. Compared upper-cased.
_NO_FINITE_LIMIT = {"", "UNLIMITED", "PARTITION_LIMIT", "INVALID"}


def whole_minutes(seconds: float) -> float:
    """`seconds` rounded up to a whole minute: a limit Slurm can hold.

    Slurm stores a job's time limit in minutes. `sbatch --time` goes through
    `arg_set_time_limit()` -> `time_str2mins()`, which rounds seconds **up**
    (`ROUNDUP(i, 60)`; src/common/slurm_opt.c L3949-L3962 and
    src/common/parse_time.c L841-L847 at SchedMD/slurm@9f9da53), and backfill
    plans with `time_limit * 60` (`_set_slot_time()`, backfill.c L1973-L1981).
    sbatch(1), `--time`: "Time resolution is one minute and second values
    are rounded up to the next minute." So every trace this repo *invents*
    limits for (the synthetic generator, and every S0 time-limit model)
    passes them through here; an sacct `Timelimit` is already whole minutes.
    The library itself (`simulate`) accepts any limit, which the hand-worked
    tests use.

    Never below `seconds`, whatever the float division does at an edge.
    """
    limit = math.ceil(seconds / 60.0) * 60.0
    return limit if limit >= seconds else limit + 60.0


@dataclass
class WorkloadProfile:
    """Shape of a synthetic workload.

    Defaults approximate a mixed research cluster: mostly short single-node
    work, a long tail of large jobs, and users who over-request wall-clock by
    a factor drawn from max(1.05, N(3, 1)). That padding is this lab's
    choice, not a figure measured from, or cited to, a trace study. The
    padded request is then rounded up to a whole minute, as Slurm stores it
    (`whole_minutes`).
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
                # Slurm holds whole minutes, rounded up (`whole_minutes`).
                time_limit=whole_minutes(duration * padding),
                nodes=rng.choice(profile.node_choices),
                cpus_per_node=profile.cpus_per_node,
                gpus_per_node=profile.gpus_per_node if wants_gpu else 0,
                qos_factor=rng.choice([0.0, 0.0, 0.0, 0.5, 1.0]),
            )
        )

    return jobs


def _slurm_time(text: str) -> float:
    """Parse a sacct time field, `[days-]HH:MM:SS`, to seconds.

    0.0 for a field with no finite value (`_NO_FINITE_LIMIT`). The caller
    decides what that means: for `Elapsed`, a job that never ran; for
    `Timelimit`, a job with no finite limit (see `load_sacct`).
    """
    text = text.strip()
    if text.upper() in _NO_FINITE_LIMIT:
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


@dataclass
class SacctTrace:
    """Jobs from an sacct file, and what the loader had to decide about them."""

    jobs: list[Job]
    #: The limit jobs without a finite `Timelimit` were planned at.
    planning_limit: float = UNLIMITED_PLANNING_LIMIT
    #: Jobs whose `Timelimit` was UNLIMITED, Partition_Limit or blank.
    no_limit_jobs: int = 0
    #: Jobs whose `Elapsed` exceeded their `Timelimit`, and the seconds cut
    #: from their runtimes to bring each down to its limit (see `load_sacct`).
    over_limit_jobs: int = 0
    over_limit_seconds: float = 0.0
    #: Seconds from the Sunday 00:00 before the earliest `Submit` to that
    #: submit, on the Submit column's own clock: where trace t=0 falls in the
    #: week, for DAILY and WEEKLY `PriorityUsageResetPeriod`
    #: (`simulate(calendar_offset=...)`).
    calendar_offset: float = 0.0
    #: The earliest `Submit`, as sacct printed it.
    origin: str = ""
    notes: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        """Lines for the run header: every count the loader had to act on."""
        lines: list[str] = []
        total = len(self.jobs)
        if self.no_limit_jobs:
            lines.append(
                f"sacct: {self.no_limit_jobs} of {total} jobs have no finite Timelimit "
                f"(UNLIMITED or Partition_Limit); planned at {_span(self.planning_limit)}, as "
                "backfill plans them, so backfill cannot fit work around them "
                "(set --partition-max-time to your partition's MaxTime)"
            )
        if self.over_limit_jobs:
            cut = self.over_limit_seconds
            amount = f"{cut:.0f} s" if cut < SECONDS_PER_HOUR else f"{cut / SECONDS_PER_HOUR:.1f} h"
            lines.append(
                f"sacct: {self.over_limit_jobs} of {total} jobs ran past their Timelimit "
                f"(TIMEOUT kill latency or OverTimeLimit); their runtimes were cut to the "
                f"limit, {amount} in all"
            )
        return lines


def _span(seconds: float) -> str:
    """`365 days`, `1 day`, `2.5 hours`, `90 minutes`: for run headers."""
    if seconds >= SECONDS_PER_DAY:
        value, unit = seconds / SECONDS_PER_DAY, "day"
    elif seconds >= SECONDS_PER_HOUR:
        value, unit = seconds / SECONDS_PER_HOUR, "hour"
    else:
        value, unit = seconds / 60.0, "minute"
    return f"{value:g} {unit}" + ("" if value == 1 else "s")


def from_sacct(path: str, *, planning_limit: float = UNLIMITED_PLANNING_LIMIT) -> list[Job]:
    """The jobs of `load_sacct(path)`."""
    return load_sacct(path, planning_limit=planning_limit).jobs


def load_sacct(path: str, *, planning_limit: float = UNLIMITED_PLANNING_LIMIT) -> SacctTrace:
    """Load a trace from `sacct` pipe-delimited output.

    Produce the input with:

        sacct -a -X --parsable2 --starttime=now-30days \\
              --format=JobID,User,Account,Submit,Elapsed,Timelimit,NNodes,ReqCPUS,ReqTRES

    `User` is optional; without it the account stands in for the user in the
    per-user backfill limits. Jobs that never started (cancelled while
    pending) are skipped — they carry no runtime to replay. `Timelimit` is
    required: it is what backfill plans with.

    **Time limits never come from `Elapsed`,** the true runtime the
    scheduler must not see. (0.1.0 used `Elapsed` wherever the limit was not
    a finite time and raised a shorter limit to it, which planned every
    UNLIMITED job, and every job that overran, with perfect foresight.) So:

    * a job with no finite limit (`UNLIMITED`, `Partition_Limit`, blank) is
      planned at `planning_limit`: by default 365 days, what Slurm's backfill
      uses when the partition's MaxTime is UNLIMITED too (see
      `UNLIMITED_PLANNING_LIMIT`). slurm.conf(5), SchedulerType:
      "Effectiveness of backfill scheduling is dependent upon users
      specifying job time limits, otherwise all jobs will have the same time
      limit and backfilling is impossible." A runtime longer than
      `planning_limit` contradicts it and is refused;
    * a job whose `Elapsed` exceeds its `Timelimit` keeps the limit, and its
      runtime is cut to it: Slurm kills a job at its limit (TIMEOUT), plus
      kill latency and any `OverTimeLimit`, none of which is modelled. The
      count and the seconds cut are on the result.

    Every job gets a distinct `job_id`. A plain numeric `JobID` keeps its
    number. Array tasks (`5000_1`, `5000_2`), heterogeneous components
    (`6000+0`) and anything else that is not a plain number, or that repeats
    an id already taken, get fresh ids above the largest plain one, in file
    order. (Collapsing `5000_1` and `5000_2` onto 5000, as this loader once
    did, gave two jobs one id.)

    Submit times are rebased so the earliest is t=0, and `calendar_offset`
    records where that falls in the week on the Submit column's clock (the
    cluster's local time, as sacct prints it by default), so DAILY and WEEKLY
    usage resets land on local midnight and Sunday 00:00 as Slurm's
    `_next_reset()` does. A daylight-saving change inside the trace moves
    the later boundaries by an hour; that is not modelled.
    """
    if not planning_limit > 0:
        raise ValueError(f"planning limit must be positive, got {planning_limit!r}")
    jobs: list[Job] = []
    trace = SacctTrace(jobs=jobs, planning_limit=planning_limit)

    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        rows = []
        for row in reader:
            check_row_width(row, path, reader.line_num)
            rows.append(row)
        if "Timelimit" not in (reader.fieldnames or []):
            raise ValueError(
                f"{path}: no Timelimit column. Backfill plans with each job's requested "
                "limit, and taking it from Elapsed would plan with the true runtime; "
                "include Timelimit in sacct --format (see README)"
            )

        stamps: list[datetime | None] = []
        for row in rows:
            try:
                stamps.append(_datetime(row.get("Submit", "")))
            except ValueError:
                stamps.append(None)
        submits = [s.timestamp() if s is not None else 0.0 for s in stamps]
        base = min((s for s in submits if s > 0), default=0.0)
        first = next(
            (s for s, t in zip(stamps, submits, strict=True) if s is not None and t == base),
            None,
        )
        if first is not None:
            trace.calendar_offset = _seconds_into_week(first)
            trace.origin = first.isoformat()

        def plain_id(row: dict[str, str]) -> int | None:
            text = (row.get("JobID") or "").strip().split(".")[0]
            return int(text) if text.isdigit() else None

        taken: set[int] = set()
        next_free = max((i for i in map(plain_id, rows) if i is not None), default=0) + 1

        for row, submit in zip(rows, submits, strict=True):
            elapsed = _slurm_time(row.get("Elapsed", ""))
            if elapsed <= 0:
                continue

            limit_text = (row.get("Timelimit") or "").strip()
            limit = _slurm_time(limit_text)
            duration = elapsed
            if limit <= 0:
                # No finite limit (a zero limit is "no limit" to sbatch --time).
                limit = planning_limit
                trace.no_limit_jobs += 1
                if elapsed > limit:
                    raise ValueError(
                        f"{path}: job {row.get('JobID', '?')} ran {elapsed:g} s with "
                        f"Timelimit {limit_text or '(blank)'}, longer than the planning limit "
                        f"{limit:g} s; raise --partition-max-time"
                    )
            elif elapsed > limit:
                trace.over_limit_jobs += 1
                trace.over_limit_seconds += elapsed - limit
                duration = limit
            nodes = int(row.get("NNodes") or 1)
            req_cpus = int(row.get("ReqCPUS") or nodes)

            parsed_id = plain_id(row)
            if parsed_id is not None and parsed_id not in taken:
                job_id = parsed_id
            else:
                job_id = next_free
                next_free += 1
            taken.add(job_id)

            user = (row.get("User") or "").strip() or None
            jobs.append(
                Job(
                    job_id=job_id,
                    account=(row.get("Account") or "unknown").strip(),
                    user=user,
                    submit_time=submit - base,
                    duration=duration,
                    time_limit=limit,
                    nodes=nodes,
                    cpus_per_node=max(1, req_cpus // max(nodes, 1)),
                    gpus_per_node=_tres_gpus(row.get("ReqTRES", "")) // max(nodes, 1),
                )
            )

    if not jobs:
        raise ValueError(f"no completed jobs found in {path}")
    trace.notes = trace.describe()
    return trace


def check_row_width(row: dict[str | None, object], path: object, line: int) -> None:
    """Refuse a row with fewer or more fields than the header.

    `csv.DictReader` fills a short row's missing fields with None and files
    extra ones under the key None. Left alone, a truncated row surfaced as a
    `TypeError` or `AttributeError` traceback from whichever parse touched
    the None first, and a row with extra fields was read silently.
    """
    if None in row:
        raise ValueError(f"{path}, line {line}: more fields than the header")
    if None in row.values():
        raise ValueError(f"{path}, line {line}: fewer fields than the header")


def _datetime(text: str) -> datetime:
    """Parse Slurm's ISO-ish timestamp (`2026-07-26T09:31:04`)."""
    text = text.strip()
    if not text or text.upper() in {"UNKNOWN", "NONE"}:
        raise ValueError("no timestamp")
    return datetime.fromisoformat(text)


def _seconds_into_week(when: datetime) -> float:
    """Seconds since the Sunday 00:00 before `when`, on `when`'s own clock.

    Slurm's WEEKLY reset is the next Sunday 00:00 and DAILY the next
    midnight, both local time (`_next_reset()`, priority_multifactor.c
    L792-L817, SchedMD/slurm@9f9da53: `tm_wday` counts from Sunday = 0).
    Python's `weekday()` counts from Monday = 0.
    """
    days = (when.weekday() + 1) % 7
    return (
        days * SECONDS_PER_DAY
        + when.hour * SECONDS_PER_HOUR
        + when.minute * 60.0
        + when.second
        + when.microsecond / 1e6
    )
