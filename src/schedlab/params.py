"""`SchedulerParameters`: the knobs that bound Slurm's schedulers.

Defaults and ranges are from slurm.conf(5) for Slurm 26.05
(https://slurm.schedmd.com/slurm.conf.html#OPT_SchedulerParameters) and were
cross-checked against `backfill.c` at SchedMD/slurm@9f9da53 (master,
2026-09-24): `BACKFILL_RESOLUTION 60`, `BACKFILL_WINDOW (24*60*60)`,
`DEF_BF_MAX_JOB_TEST 500` (L103-L116).

One documentation conflict, stated rather than hidden: the Scheduling
Configuration Guide (https://slurm.schedmd.com/sched_config.html) still says
`bf_max_job_test` defaults to 100. slurm.conf(5) and the source both say 500;
this module uses 500.

All intervals are **simulated** seconds. This simulator advances an event
clock and has no wall clock to compress, so `bf_interval=30` means a backfill
cycle every 30 simulated seconds, exactly as on a controller. Contrast the
sibling k8s lab, which replays against a live kube-scheduler with timestamps
divided by `--speedup` and must rescale real-time controller constants to
match (see its `src/k8slab/timescale.py`). Nothing here needs that.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace

from .slurmconf import read_slurm_conf

_LEADING_INT = re.compile(r"^[+-]?\d+")

#: Keys this simulator acts on. Everything else found in the string is kept in
#: `SchedulerParameters.unmodelled` so a run can say what it ignored.
_MODELLED = {
    "bf_interval",
    "bf_window",
    "bf_resolution",
    "bf_max_job_test",
    "bf_max_job_start",
    "bf_max_job_user",
    "bf_max_job_part",
    "bf_max_job_user_part",
    "bf_max_job_assoc",
    "default_queue_depth",
    "sched_interval",
    "partition_job_depth",
    "requeue_delay",
}


@dataclass(frozen=True)
class SchedulerParameters:
    """Slurm's `SchedulerParameters`, as used by the conservative mode.

    Times are stored in seconds even where slurm.conf takes minutes
    (`bf_window`); `parse()` converts.
    """

    #: Seconds between backfill cycles. Default 30, range 1-10800. -1 disables
    #: the backfill loop.
    bf_interval: float = 30.0
    #: How far ahead the backfill plan looks, in **seconds** (slurm.conf gives
    #: minutes: default 1440 = one day, range 1-43200). A job whose earliest
    #: start lies beyond it gets no reservation — backfill.c logs "StartTime
    #: set to time after current backfill window. No reservation created"
    #: (L3700-L3717) — so lower-priority work may take its nodes.
    bf_window: float = 1440 * 60.0
    #: Granularity of the plan, seconds. Default 60, range 1-3600. Planned
    #: starts are floored and planned ends rounded up to it (`_set_slot_time`,
    #: backfill.c L1973-L1981; running jobs: `_bf_reserve_running`, L1683).
    bf_resolution: float = 60.0
    #: Jobs tried per backfill cycle (the queue depth). Default 500, range
    #: 1-1,000,000. Counts jobs that reach the placement test, not jobs
    #: skipped by per-user/partition limits (`job_test_cnt`, backfill.c L3233).
    bf_max_job_test: int = 500
    #: Jobs started per backfill cycle. Default 0 (no limit), max 10000.
    bf_max_job_start: int = 0
    #: Jobs *checked* per user per cycle; later ones are skipped ("have already
    #: checked %u jobs for user", backfill.c L1875). Default 0 (no limit).
    bf_max_job_user: int = 0
    #: Jobs checked per partition per cycle. Default 0 (no limit).
    bf_max_job_part: int = 0
    #: Jobs checked per user per partition per cycle. Default 0 (no limit).
    bf_max_job_user_part: int = 0
    #: Jobs checked per association per cycle. The simulator's association is
    #: (user, account). Default 0 (no limit). When both this and
    #: `bf_max_job_user` are set, this one wins and the per-user cap is
    #: dropped (backfill.c `_load_config()`, L974-L979).
    bf_max_job_assoc: int = 0
    #: Depth of the *event-triggered* main scheduler on each submit or
    #: completion. Default 100, and 0 is allowed; Slurm rejects only negative
    #: values (job_scheduler.c L1273-L1284). The pass tests up to this many
    #: jobs **plus one**: it breaks on `job_depth++ > def_job_limit` (L1578).
    default_queue_depth: int = 100
    #: Seconds between full main-scheduler passes (which ignore
    #: `default_queue_depth`). Default 60. 0 makes every pass a full one: the
    #: controller asks for one whenever `now - last_full_sched_time >=
    #: sched_interval` (controller.c L2939). -1 disables the main scheduler
    #: altogether, event-triggered passes included — `_schedule()` returns at
    #: once (job_scheduler.c L1360-L1372) — so only backfill starts jobs.
    #: slurm.conf(5): "A setting of -1 will disable the main scheduling loop."
    sched_interval: float = 60.0
    #: Jobs the main scheduler considers per partition. Default 0 (no limit).
    partition_job_depth: int = 0
    #: Seconds a requeued job waits before it is eligible again ("Delay before
    #: a non-Expedited Requeue job is eligible to run after being requeued",
    #: slurm.conf(5)). None is the default, `AuthInfo=cred_expire` — 120 s
    #: unless set, which this simulator does not read. Only preemption with
    #: PreemptMode=REQUEUE requeues jobs here.
    requeue_delay: float | None = None

    #: Keys present in the parsed string that the simulator does not model,
    #: e.g. `bf_continue`, `bf_max_time`, `defer`, `bf_min_prio_reserve`.
    unmodelled: tuple[str, ...] = ()
    #: Values Slurm would reject and replace with the default, as it does.
    warnings: tuple[str, ...] = ()

    @property
    def backfill_enabled(self) -> bool:
        return self.bf_interval > 0

    @property
    def sched_interval_enabled(self) -> bool:
        """True when the periodic full pass is a timer (`sched_interval > 0`)."""
        return self.sched_interval > 0

    @property
    def main_scheduler_enabled(self) -> bool:
        """False for `sched_interval=-1`, which disables every main pass."""
        return self.sched_interval != -1

    # ── Parsing ─────────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, text: str) -> SchedulerParameters:
        """Parse a `SchedulerParameters=` value: `key=value,flag,key=value`.

        Keys are case-insensitive, as Slurm matches them with `xstrcasestr`.
        Out-of-range values fall back to the default with a warning, which is
        what `backfill.c` does for the backfill options it validates
        (L801-L872); for the others the fallback is this module's choice and
        says so in the warning.
        """
        defaults = cls()
        values: dict[str, float | int] = {}
        unmodelled: list[str] = []
        warnings: list[str] = []

        for raw in text.split(","):
            item = raw.strip()
            if not item:
                continue
            key, has_value, value = item.partition("=")
            key = key.strip().lower()
            if key not in _MODELLED:
                unmodelled.append(key)
                continue
            if not has_value:
                warnings.append(f"{key} needs a value; using the default")
                continue
            match = _LEADING_INT.match(value.strip())
            if match is None:
                warnings.append(f"{key}={value!r} is not a number; using the default")
                continue
            # backfill.c reads these with atoi(): "2.5" is 2.
            checked = _validate(key, float(match.group()))
            if checked is None:
                warnings.append(f"{key}={value.strip()} is out of range; using the default")
                continue
            values[key] = checked

        if values.get("bf_max_job_assoc") and values.get("bf_max_job_user"):
            # backfill.c `_load_config()` (L974-L979) logs this error and
            # zeroes the per-user limit.
            warnings.append(
                "Both bf_max_job_user and bf_max_job_assoc are set: bf_max_job_assoc taking "
                "precedence (bf_max_job_user ignored, as backfill.c does)"
            )
            values["bf_max_job_user"] = 0

        params = replace(defaults, **values)  # type: ignore[arg-type]
        return replace(params, unmodelled=tuple(unmodelled), warnings=tuple(warnings))

    @classmethod
    def from_conf(cls, parsed: Mapping[str, str]) -> SchedulerParameters:
        """From a `read_slurm_conf()` mapping; defaults if the key is absent."""
        return cls.parse(parsed.get("schedulerparameters", ""))

    @classmethod
    def from_slurm_conf(cls, path: str) -> SchedulerParameters:
        return cls.from_conf(read_slurm_conf(path))

    def describe(self) -> str:
        """One line naming every modelled value, for run headers."""
        parts = []
        for f in fields(self):
            if f.name in {"unmodelled", "warnings"}:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "bf_window":
                parts.append(f"bf_window={value / 60:g}min")
            elif isinstance(value, float):
                parts.append(f"{f.name}={value:g}s")
            else:
                parts.append(f"{f.name}={value}")
        return " ".join(parts)


def _validate(key: str, value: float) -> float | int | None:
    """Convert to the stored unit and range-check. None means "use default"."""
    if key == "bf_interval":
        # slurm.conf(5): Min 1, Max 10800; -1 disables the loop.
        return value if value == -1 or 1 <= value <= 10_800 else None
    if key == "bf_window":
        # Minutes in slurm.conf; Min 1, Max 43200.
        return value * 60.0 if 1 <= value <= 43_200 else None
    if key == "bf_resolution":
        return value if 1 <= value <= 3_600 else None
    if key == "bf_max_job_test":
        return int(value) if 1 <= value <= 1_000_000 else None
    if key == "bf_max_job_start":
        return int(value) if 0 <= value <= 10_000 else None
    if key in {"bf_max_job_user", "bf_max_job_part", "bf_max_job_user_part", "bf_max_job_assoc"}:
        return int(value) if value >= 0 else None
    if key == "default_queue_depth":
        # job_scheduler.c L1273-L1284: only a negative value is "ignoring
        # SchedulerParameters: default_queue_depth" (back to 100). 0 is valid
        # and still tests one job per event pass (see the field).
        return int(value) if value >= 0 else None
    if key == "sched_interval":
        # job_scheduler.c L1360-L1380: -1 disables the main scheduler, any
        # other negative is "Invalid sched_interval" (back to 60), and 0 is
        # accepted (a full pass at every event here; see the field).
        return value if value == -1 or value >= 0 else None
    if key == "partition_job_depth":
        return int(value) if value >= 0 else None
    if key == "requeue_delay":
        # Read with atoi() in job_mgr.c `_requeue_delay()`; a negative delay
        # would make a job eligible before it was requeued.
        return value if value >= 0 else None
    return None
