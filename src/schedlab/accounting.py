"""When usage becomes visible to priority: a model of Slurm's decay thread.

0.1.0 charged a job's *entire* usage — `total_cpus * duration` — the instant it
was dispatched. That was wrong twice over: it read `duration`, the true runtime
the scheduler is never allowed to see, and it made future usage count against
an account before a single second of it had been consumed.

Slurm does neither. In `priority/multifactor`, a dedicated thread wakes every
`PriorityCalcPeriod` (default 5 minutes). Each wake, per `_decay_thread()` in
priority_multifactor.c (SchedMD/slurm@9f9da53b4a7bc5b56062bc357f4f94e0d19c71cc,
L1348-L1506):

1. applies a `PriorityUsageResetPeriod` reset if one is due;
2. (classic only) computes every *account's* effective usage from raw usage
   *as it stands*, and marks every *user's* effective usage stale (NO_VAL) —
   `_set_children_usage_efctv()` via `_foreach_set_usage_efctv()`
   (L368-L380, called at L1431-L1436);
3. decays all recorded usage by the time since the previous wake;
4. for each running job, adds usage for the time it ran since the previous
   wake — `_apply_new_usage()`, clipped to the job's start (L1084-L1291);
5. recomputes the priority of every pending job, and the fairshare factors.

When a job ends, `priority_p_job_end()` (L2057-L2062) applies the final partial
period immediately. Usage is billed as allocated CPUs × elapsed seconds, the
default when `TRESBillingWeights` is unset
(https://slurm.schedmd.com/priority_multifactor.html).

The consequence is a **usage-visibility lag**, and it is a first-class property
of this model rather than an artefact:

* A pending job's priority is a snapshot from its submit time or the most
  recent tick, whichever is later. Nothing in between changes it. Slurm
  stores it as an integer (see `priority.controller_priority`).
* Fair Tree (the default) recomputes factors *before* job priorities in the
  same tick (`fair_tree_decay()`, fair_tree.c L56-L78), so priorities see
  usage up to the tick: lag ≤ one `PriorityCalcPeriod`.
* Classic sees usage as it stood at the *start* of the tick, before step 4's
  accrual: one period more than Fair Tree. In step 5 each pending job's
  fairshare comes from its user association; step 2 left that NO_VAL, so
  `_get_fairshare_priority()` recomputes it on the spot (L417-L418) with
  `_set_usage_efctv()` (L1788-L1805): `UE_user = UA_user + (UE_account -
  UA_user) × S_user / S_siblings`. For the one-user accounts this model
  has, the share ratio is 1 and `UE_user` is the account's step-2 value,
  whatever the user's raw usage has become since. A job submitted between
  ticks gets the same value (the account is only recomputed at step 2). So
  a job's classic factor is always the one computed at the start of the most
  recent tick. (Until this was re-read, the model gave the tick's own
  priorities the *previous* tick's factor, a period staler than Slurm.)

Slurm's own period is wall-clock; here it is simulated seconds, and the tick
is a discrete event. There is no time compression in this simulator (see
README), so no rescaling is needed: a 5-minute period is 300 simulated seconds
by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import Cluster, Job
from .priority import FairshareTree, PriorityWeights, compute_priority, controller_priority

#: Seconds in the DAILY and WEEKLY reset cycles. The calendar-month cycles
#: have no fixed length and need a real calendar, which the simulator lacks.
_RESET_CYCLE = {"DAILY": 86_400.0, "WEEKLY": 7 * 86_400.0}


@dataclass(frozen=True)
class FairshareSnapshot:
    """State right after one `PriorityCalcPeriod` tick."""

    time: float
    #: Decayed raw usage per account (CPU-seconds), after this tick's accrual.
    usage: dict[str, float]
    #: Fairshare factor each account's pending jobs were re-prioritised with
    #: at this tick. For classic it is computed from usage as it stood at the
    #: start of the tick, so it lags `usage` (see module docstring).
    factors: dict[str, float]


class PriorityEngine:
    """Owns job priorities, usage accrual and decay for one simulation.

    The simulator calls `submit()`, `job_started()`, `job_ended()` and `tick()`
    at the matching events. Scheduling code only ever reads `job.priority`.

    Nothing here reads `job.duration`: accrual uses the elapsed time between
    events the simulator has already reached, never a job's future.
    """

    def __init__(
        self,
        cluster: Cluster,
        weights: PriorityWeights,
        fairshare: FairshareTree | None = None,
        partitions: dict[str, float] | None = None,
        calendar_offset: float = 0.0,
    ) -> None:
        if weights.calc_period < 0:
            raise ValueError("calc_period must be >= 0")
        period = weights.usage_reset_period
        if period in {"MONTHLY", "QUARTERLY", "YEARLY"}:
            raise ValueError(
                f"PriorityUsageResetPeriod={period} needs a calendar; the simulator only "
                "models NONE, NOW, DAILY and WEEKLY"
            )
        self.cluster = cluster
        self.weights = weights
        self.fairshare = fairshare
        self.partitions = partitions or {}
        #: Seconds past a Sunday 00:00 at which simulated t=0 falls. DAILY and
        #: WEEKLY resets fire when `t + calendar_offset` crosses a multiple of
        #: one day / one week — Slurm resets at local midnight / Sunday 00:00
        #: (`_next_reset()`, priority_multifactor.c L792-L849).
        self.calendar_offset = calendar_offset

        self.last_tick: float | None = None
        self.snapshots: list[FairshareSnapshot] = []
        #: Per running job (by identity; ids need not be unique): simulated
        #: time up to which usage has been charged.
        self._accrued_until: dict[Job, float] = {}
        self._reset_now_pending = period == "NOW"
        #: The factors job priorities are computed with right now.
        self.visible_factors: dict[str, float] = fairshare.factors() if fairshare else {}

    @property
    def idealised(self) -> bool:
        """True for `calc_period == 0`: no lag.

        The simulator then ticks at every event with a job pending — every
        submit, completion, main pass, backfill cycle and requeue begin time
        — so every scheduling decision sees usage and ages as of that moment.
        """
        return self.weights.calc_period == 0

    # ── Events ──────────────────────────────────────────────────────────────

    def submit(self, job: Job, now: float) -> None:
        """Initial priority, computed with whatever factors are visible now.

        Slurm assigns a priority at submit (Fair Tree docs: "New jobs are
        immediately assigned a priority"), using the association's current
        fairshare factor — itself a product of the last tick.
        """
        job.priority = self._priority(job, now)

    def job_started(self, job: Job, now: float) -> None:
        self._accrued_until[job] = now

    def job_ended(self, job: Job, now: float) -> None:
        """Final partial accrual, from the last tick to the job's end.

        Mirrors `priority_p_job_end()`: the usage lands in the raw total
        immediately, but job priorities do not see it until the next tick.
        """
        self._accrue(job, now)
        self._accrued_until.pop(job, None)

    def tick(self, now: float, pending: list[Job], running: list[Job]) -> None:
        """One wake of the decay thread at simulated time `now`."""
        tree = self.fairshare
        previous = self.last_tick

        if tree is not None and self._reset_due(previous, now):
            tree.reset()

        classic = tree is not None and tree.algorithm == "classic" and not self.idealised
        if tree is not None and classic:
            # Step 2: from usage as it stands, before this tick's decay and
            # accrual. It is what this tick's job priorities use, and what
            # jobs submitted before the next tick get (module docstring).
            self.visible_factors = tree.factors()

        if tree is not None and previous is not None:
            tree.decay_by(now - previous)

        for job in running:
            self._accrue(job, now)

        if tree is not None and not classic:
            # Fair Tree (and the idealised mode): factors after this tick's
            # accrual, then jobs.
            self.visible_factors = tree.factors()

        for job in pending:
            job.priority = self._priority(job, now)

        if tree is not None:
            self.snapshots.append(
                FairshareSnapshot(
                    time=now, usage=dict(tree.usage), factors=dict(self.visible_factors)
                )
            )
        self.last_tick = now

    # ── Internals ───────────────────────────────────────────────────────────

    def _priority(self, job: Job, now: float) -> int:
        """The integer the controller stores, not the raw weighted sum."""
        fs = self.visible_factors.get(job.account, 0.0) if self.fairshare else 0.0
        return controller_priority(
            compute_priority(
                job, now, self.cluster, self.weights, None, self.partitions, fs_factor=fs
            )
        )

    def _accrue(self, job: Job, until: float) -> None:
        """Charge `job` for the time it has run since its last charge.

        Reads only `start_time` and the current simulated time. The new usage
        is itself decayed over its own span, as `_apply_new_usage()` does
        (`run_decay = run_delta * pow(decay_factor, run_delta)`, L1184).
        """
        if self.fairshare is None or job.start_time is None:
            return
        since = max(self._accrued_until.get(job, job.start_time), job.start_time)
        elapsed = until - since
        if elapsed <= 0:
            return
        amount = job.total_cpus * elapsed * self.fairshare.decay_multiplier(elapsed)
        self.fairshare.charge(job.account, amount)
        self._accrued_until[job] = until

    def _reset_due(self, previous: float | None, now: float) -> bool:
        if self._reset_now_pending:
            # NOW: "Clear the historic usage now. Executed at startup."
            self._reset_now_pending = False
            return True
        cycle = _RESET_CYCLE.get(self.weights.usage_reset_period)
        if cycle is None or previous is None:
            return False
        before = math.floor((previous + self.calendar_offset) / cycle)
        after = math.floor((now + self.calendar_offset) / cycle)
        return after > before
