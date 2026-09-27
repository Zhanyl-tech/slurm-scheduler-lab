"""Low-level slurm.conf reading and Slurm time-string parsing.

Kept free of other `schedlab` imports so both the priority model and the
scheduler-parameter model can use it without an import cycle.

Time strings follow `time_str2secs()` / `time_str2mins()` in Slurm's
`src/common/parse_time.c`, read at SchedMD/slurm commit
9f9da53b4a7bc5b56062bc357f4f94e0d19c71cc (master, 2026-09-24):
https://github.com/SchedMD/slurm/blob/9f9da53b4a7bc5b56062bc357f4f94e0d19c71cc/src/common/parse_time.c#L799-L847

The rule that matters, and that the 0.1.0 parser got wrong: **without a day
part, a bare number is minutes and two fields are `minutes:seconds`.** Only
three colon-separated fields mean `hours:minutes:seconds`. So
`PriorityMaxAge=30` is thirty minutes, not thirty hours.
"""

from __future__ import annotations

import math

SECONDS_PER_DAY = 86_400.0

_INFINITE = {"-1", "INFINITE", "UNLIMITED"}


def _is_valid_timespec(text: str) -> bool:
    """Port of `_is_valid_timespec()` in parse_time.c (same commit, L92-L145).

    Only digits, one `-` and up to two `:`; the dash must precede any colon;
    and every colon needs a digit group on both sides.
    """
    digit_groups = dashes = colons = 0
    in_digits = False
    for ch in text:
        if ch.isdigit():
            if not in_digits:
                digit_groups += 1
                in_digits = True
        elif ch == "-":
            in_digits = False
            dashes += 1
            if colons:
                return False
        elif ch == ":":
            in_digits = False
            colons += 1
        else:
            return False
    if not digit_groups or dashes > 1 or colons > 2:
        return False
    if dashes:
        return not ((colons == 1 and digit_groups < 3) or (colons == 2 and digit_groups < 4))
    return not ((colons == 1 and digit_groups < 2) or (colons == 2 and digit_groups < 3))


def read_slurm_conf(path: str) -> dict[str, str]:
    """Return `key -> value` for every `Key=Value` line, keys lower-cased.

    slurm.conf keys are case-insensitive. Only the first `=` splits, so
    `SchedulerParameters=bf_window=2880,bf_continue` keeps its whole value.
    Comments (`#` to end of line) and blank lines are ignored. Node and
    partition definition lines are read too, but nothing here consumes them.
    """
    parsed: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            parsed[key.strip().lower()] = value.strip()
    return parsed


def parse_time_str(text: str) -> float:
    """Seconds for a Slurm time string, mirroring `time_str2secs()`.

    Accepted forms: `minutes`, `minutes:seconds`, `hours:minutes:seconds`,
    `days-hours`, `days-hours:minutes`, `days-hours:minutes:seconds`.
    `-1`, `INFINITE` and `UNLIMITED` return `math.inf`.

    Raises `ValueError` for anything Slurm would reject as `NO_VAL`.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty time string")
    if text.upper() in _INFINITE:
        return math.inf
    if not _is_valid_timespec(text):
        raise ValueError(f"not a Slurm time string: {text!r}")

    def fields(part: str) -> list[int]:
        return [int(p) if p else 0 for p in part.split(":")]

    if "-" in text:
        day_part, _, rest = text.partition("-")
        days = int(day_part)
        hms = fields(rest) if rest else []
        hms += [0] * (3 - len(hms))
        hours, minutes, seconds = hms[:3]
        return days * SECONDS_PER_DAY + hours * 3600.0 + minutes * 60.0 + seconds

    parts = fields(text)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600.0 + minutes * 60.0 + seconds
    # One or two fields: minutes[:seconds].
    minutes = parts[0]
    seconds = parts[1] if len(parts) > 1 else 0
    return minutes * 60.0 + seconds


def parse_minutes_str(text: str) -> float:
    """Seconds for a time string that Slurm stores in whole minutes.

    Mirrors `time_str2mins()`: parse to seconds, then round **up** to the next
    whole minute. `PriorityDecayHalfLife`, `PriorityMaxAge` and
    `PriorityCalcPeriod` are all read this way in `read_config.c`.
    """
    seconds = parse_time_str(text)
    if math.isinf(seconds):
        return seconds
    return math.ceil(seconds / 60.0) * 60.0
