"""Slurm's multifactor priority plugin.

Mirrors `priority/multifactor` as documented in slurm.conf(5):

    priority = weight_age       * age_factor
             + weight_fairshare * fairshare_factor
             + weight_jobsize   * jobsize_factor
             + weight_partition * partition_factor
             + weight_qos       * qos_factor
             - nice

Every factor is normalised to [0, 1], so the weights alone decide what the
queue optimises for. That is the whole point: the weights are the policy, and
this module lets you change them without touching a live controller.

The sum is a float here (`compute_priority`); the controller stores it as an
integer of at least 1 (`controller_priority`), and that integer is what the
queue sorts by.

*When* those factors are refreshed is modelled in `accounting.py`: usage
accrues and job priorities are recomputed on `PriorityCalcPeriod` ticks, not
continuously. This module holds the static parts — the weights, the fairshare
tree and the per-factor formulas.

Two fairshare algorithms are implemented, because Slurm ships two:

* **Fair Tree** — the default since Slurm 19.05 ("As of the 19.05 release, the
  'Fair Tree' fairshare algorithm has been made the default",
  https://slurm.schedmd.com/priority_multifactor.html). Accounts are ranked by
  Level Fairshare `LF = S / U` and the factor is `rank / N`
  (https://slurm.schedmd.com/fair_tree.html).
* **Classic** — `F = 2^(-U/S)`, used only with `PriorityFlags=NO_FAIR_TREE`
  (https://slurm.schedmd.com/classic_fair_share.html).

The tree is flat: one level of accounts under root, each account treated as a
single user association. See README "Scope and simplifications".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, get_args

from .model import Cluster, Job
from .slurmconf import SECONDS_PER_DAY, parse_minutes_str, read_slurm_conf

FairshareAlgorithm = Literal["classic", "fair_tree"]
FAIRSHARE_ALGORITHMS: tuple[str, ...] = get_args(FairshareAlgorithm)

#: `PriorityUsageResetPeriod` values, from slurm.conf(5) and the parser in
#: `read_config.c` (SchedMD/slurm@9f9da53, L4830-L4852).
UsageResetPeriod = Literal["NONE", "NOW", "DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "YEARLY"]
USAGE_RESET_PERIODS: tuple[str, ...] = get_args(UsageResetPeriod)

#: `PriorityFlags` this simulator acts on. Anything else is reported, not
#: silently ignored.
_MODELLED_PRIORITY_FLAGS = {"NO_FAIR_TREE", "DEPTH_OBLIVIOUS"}


@dataclass
class PriorityWeights:
    """Maps onto the `Priority*` settings in slurm.conf.

    The *weight* defaults below are the lab's demo values, not Slurm's: Slurm
    defaults every `PriorityWeight*` to 0, which makes every job tie and the
    queue FIFO. `slurm_defaults()` gives the real defaults, and
    `from_slurm_conf()` starts from them, so a key your config omits behaves as
    it would on the controller. The non-weight defaults (`max_age`,
    `decay_half_life`, `calc_period`, reset period, algorithm) are Slurm's.
    """

    age: float = 1_000.0
    fairshare: float = 10_000.0
    jobsize: float = 10_000.0
    partition: float = 1_000.0
    qos: float = 10_000.0

    #: `PriorityMaxAge` — the age at which age_factor saturates at 1.0.
    max_age: float = 7 * SECONDS_PER_DAY
    #: `PriorityFavorSmall` — when true, small jobs get the larger factor.
    favor_small: bool = False
    #: `PriorityDecayHalfLife` — Slurm default 7-0 (7 days). 0 disables decay.
    #: slurmctld accepts 0 only with `PriorityUsageResetPeriod` set ("If set
    #: to 0 PriorityUsageResetPeriod must be set to some interval",
    #: slurm.conf(5)); `from_conf` refuses the pair as read_config.c does.
    decay_half_life: float = 7 * SECONDS_PER_DAY
    #: `PriorityCalcPeriod`, in seconds (slurm.conf takes minutes; default 5).
    #: Usage accrues, decays, and job priorities are recomputed only on these
    #: ticks. Slurm reads it with `time_str2mins()`, which rounds *up* to
    #: whole minutes (parse_time.c L841-L847), and rejects only a result
    #: below 1 (`read_config.c`, SchedMD/slurm@9f9da53 L4756-L4766): `0:30`
    #: runs as one minute, `0` is an error. So a Slurm period is always a
    #: positive whole number of minutes. `0` here is a lab-only idealisation —
    #: refresh at every scheduling event — that Slurm cannot run.
    calc_period: float = 300.0
    #: `PriorityUsageResetPeriod`. The simulator implements NONE, NOW, DAILY
    #: and WEEKLY; the calendar-month values need a real calendar and raise.
    usage_reset_period: UsageResetPeriod = "NONE"
    #: Fair Tree unless `PriorityFlags` contains NO_FAIR_TREE (or
    #: DEPTH_OBLIVIOUS, which slurm.conf(5) says "automatically enables
    #: NO_FAIR_TREE").
    fairshare_algorithm: FairshareAlgorithm = "fair_tree"

    @classmethod
    def slurm_defaults(cls) -> PriorityWeights:
        """Every value at its slurm.conf(5) default (all weights 0)."""
        return cls(age=0.0, fairshare=0.0, jobsize=0.0, partition=0.0, qos=0.0)

    @classmethod
    def from_slurm_conf(cls, path: str) -> PriorityWeights:
        """Read the `Priority*` lines out of a real slurm.conf.

        Lets you point the simulator at the config you are about to deploy
        instead of re-typing its numbers. Keys the file omits take Slurm's
        default, not the lab's. Use `config.SlurmConfig.load()` to also see the
        warnings about settings that are parsed but not modelled.
        """
        weights, _ = cls.from_conf(read_slurm_conf(path))
        return weights

    @classmethod
    def from_conf(cls, parsed: Mapping[str, str]) -> tuple[PriorityWeights, list[str]]:
        """Build from an already-parsed `read_slurm_conf()` mapping.

        Returns the weights and human-readable warnings for anything present
        in the config that the simulator does not model.
        """
        warnings: list[str] = []
        weights = cls.slurm_defaults()

        for attr, key in (
            ("age", "priorityweightage"),
            ("fairshare", "priorityweightfairshare"),
            ("jobsize", "priorityweightjobsize"),
            ("partition", "priorityweightpartition"),
            ("qos", "priorityweightqos"),
        ):
            if key in parsed:
                setattr(weights, attr, float(parsed[key]))

        for key in ("priorityweightassoc", "priorityweighttres"):
            if key in parsed:
                warnings.append(f"{key} is parsed but not modelled")

        if "prioritymaxage" in parsed:
            weights.max_age = _parse_duration(parsed["prioritymaxage"])
        if "prioritydecayhalflife" in parsed:
            weights.decay_half_life = _parse_duration(parsed["prioritydecayhalflife"])
            if weights.decay_half_life == 0 and "priorityusageresetperiod" not in parsed:
                # read_config.c (SchedMD/slurm@9f9da53 L4851-L4860): with no
                # PriorityUsageResetPeriod line and a zero half-life it logs
                # "You have to either have PriorityDecayHalfLife != 0 or
                # PriorityUsageResetPeriod set to something or the priority
                # plugin will result in rolling over." and returns
                # SLURM_ERROR, whatever PriorityType is. slurmctld would not
                # start, so refuse rather than run with usage that never decays.
                raise ValueError(
                    f"PriorityDecayHalfLife={parsed['prioritydecayhalflife']!r} disables "
                    "decay, and without PriorityUsageResetPeriod Slurm refuses the config "
                    "(usage would only ever grow)"
                )
        if "priorityfavorsmall" in parsed:
            weights.favor_small = parsed["priorityfavorsmall"].upper() in {"YES", "1", "TRUE"}
        if "prioritycalcperiod" in parsed:
            # Already rounded up to whole minutes, so only a value that rounds
            # to 0 (or -1/INFINITE) lands here.
            period = _parse_duration(parsed["prioritycalcperiod"])
            if not math.isfinite(period) or period < 60:
                # read_config.c returns SLURM_ERROR here: slurmctld would not
                # start. Refuse rather than guess.
                raise ValueError(
                    f"PriorityCalcPeriod={parsed['prioritycalcperiod']!r}: Slurm requires "
                    "at least 1 minute"
                )
            weights.calc_period = period
        if "priorityusageresetperiod" in parsed:
            value = parsed["priorityusageresetperiod"].upper()
            if value not in USAGE_RESET_PERIODS:
                raise ValueError(f"PriorityUsageResetPeriod={value!r} is not a Slurm value")
            weights.usage_reset_period = value  # type: ignore[assignment]

        flags = {
            f.strip().upper() for f in parsed.get("priorityflags", "").split(",") if f.strip()
        }
        if flags & {"NO_FAIR_TREE", "DEPTH_OBLIVIOUS"}:
            weights.fairshare_algorithm = "classic"
        if "DEPTH_OBLIVIOUS" in flags:
            warnings.append(
                "PriorityFlags=DEPTH_OBLIVIOUS is simulated as classic fairshare; on a flat "
                "account tree its depth correction has nothing to act on (unverified against "
                "Slurm's implementation)"
            )
        for flag in sorted(flags - _MODELLED_PRIORITY_FLAGS):
            warnings.append(f"PriorityFlags={flag} is parsed but not modelled")

        priority_type = parsed.get("prioritytype", "priority/multifactor").lower()
        if priority_type == "priority/basic":
            # slurm.conf(5): "Jobs are evaluated in a First In, First Out
            # (FIFO) manner." Zero weights leave only the submit-time tiebreak.
            weights = replace(weights, age=0.0, fairshare=0.0, jobsize=0.0, partition=0.0, qos=0.0)
        elif priority_type != "priority/multifactor":
            warnings.append(f"PriorityType={priority_type} is not modelled; using multifactor")

        return weights, warnings

    def make_fairshare(self, accounts: Iterable[str] | Mapping[str, float]) -> FairshareTree:
        """A fairshare tree wired to these settings (half-life, algorithm).

        Pass a mapping for explicit shares, or any iterable of account names
        for equal shares.
        """
        if isinstance(accounts, Mapping):
            shares = {str(a): float(s) for a, s in accounts.items()}
        else:
            shares = {a: 1.0 for a in sorted(set(accounts))}
        return FairshareTree(
            shares=shares, half_life=self.decay_half_life, algorithm=self.fairshare_algorithm
        )


def _parse_duration(text: str) -> float:
    """Seconds for a `Priority*` duration, parsed as Slurm parses it.

    `days-hr:min:sec`, `hr:min:sec`, `min:sec` or bare `min`, rounded up to a
    whole minute (`time_str2mins()`). A bare number is **minutes**.
    """
    return parse_minutes_str(text)


@dataclass
class FairshareTree:
    """Flat single-level fairshare — one share allocation per account.

    `usage` is Slurm's `usage_raw`: allocated CPU-seconds, decayed on the
    half-life. Nothing in here knows about time; `accounting.PriorityEngine`
    decides when usage accrues and when factors become visible to jobs.
    """

    shares: dict[str, float]
    usage: dict[str, float] = field(default_factory=dict)
    half_life: float = 7 * SECONDS_PER_DAY
    algorithm: FairshareAlgorithm = "fair_tree"
    _last_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.algorithm not in FAIRSHARE_ALGORITHMS:
            raise ValueError(f"unknown fairshare algorithm {self.algorithm!r}")

    # ── Usage bookkeeping ───────────────────────────────────────────────────

    def decay_multiplier(self, elapsed: float) -> float:
        """Factor that `elapsed` seconds of half-life decay multiplies usage by.

        Exactly `0.5 ** (elapsed / half_life)`. Slurm uses the first-order
        approximation `D = 1 - 0.693/half_life` per second
        (priority_multifactor.c, SchedMD/slurm@9f9da53 L1394), which leaves
        0.50007 rather than 0.5 after one half-life — a 0.015% difference this
        model does not reproduce. Half-life 0 means no decay, as in Slurm.
        """
        if elapsed <= 0 or self.half_life <= 0 or math.isinf(self.half_life):
            return 1.0
        return float(0.5 ** (elapsed / self.half_life))

    def decay_by(self, elapsed: float) -> None:
        """Decay every account's usage by `elapsed` seconds of half-life."""
        factor = self.decay_multiplier(elapsed)
        if factor == 1.0:
            return
        for account in self.usage:
            self.usage[account] *= factor

    def decay(self, now: float) -> None:
        """Decay by the time since the previous `decay()` call.

        Kept for API compatibility. The simulator no longer calls it; decay is
        driven by `PriorityCalcPeriod` ticks through `decay_by()`.
        """
        elapsed = now - self._last_decay
        if elapsed <= 0:
            return
        self.decay_by(elapsed)
        self._last_decay = now

    def charge(self, account: str, cpu_seconds: float) -> None:
        """Add already-decayed usage to an account."""
        self.usage[account] = self.usage.get(account, 0.0) + cpu_seconds

    def reset(self) -> None:
        """Zero all usage, as `PriorityUsageResetPeriod` does."""
        for account in self.usage:
            self.usage[account] = 0.0

    # ── Shares and usage, normalised ────────────────────────────────────────

    def normalized_shares(self, account: str) -> float:
        total = sum(self.shares.values())
        if total <= 0:
            return 0.0
        return self.shares.get(account, 0.0) / total

    def normalized_usage(self, account: str) -> float:
        """`U = usage_raw(account) / usage_raw(root)`; 0 when nothing is used."""
        total = sum(self.usage.values())
        if total <= 0:
            return 0.0
        return self.usage.get(account, 0.0) / total

    # ── Factors ─────────────────────────────────────────────────────────────

    def classic_factor(self, account: str) -> float:
        """Slurm's classic fairshare: F = 2^(-U_norm / S_norm).

        F = 0.5 means the account is using exactly its share. Above means
        under-served (priority boost), below means over-served.
        `FairShareDampeningFactor` is not modelled (its default is 1).
        """
        shares = self.normalized_shares(account)
        if shares <= 0:
            return 0.0

        total_usage = sum(self.usage.values())
        if total_usage <= 0:
            return 1.0

        usage_norm = self.usage.get(account, 0.0) / total_usage
        return float(2.0 ** (-usage_norm / shares))

    def level_fairshare(self, account: str) -> float:
        """Fair Tree's Level Fairshare, `LF = S / U`, among sibling accounts.

        Follows `_calc_assoc_fs()` in fair_tree.c (SchedMD/slurm@9f9da53
        L176-L211): `S == 0` gives 0 (lowest); `U == 0` with `S > 0` gives
        infinity (highest). "Under-served associations will have a value
        greater than 1.0."
        """
        shares = self.normalized_shares(account)
        if shares <= 0:
            return 0.0
        usage = self.normalized_usage(account)
        if usage <= 0:
            return math.inf
        return shares / usage

    def fair_tree_factors(self) -> dict[str, float]:
        """Fair Tree factors for every account in `shares`.

        Accounts are sorted by Level Fairshare (highest first) and each gets
        `rank / N`, rank counting down from N. Ties share a rank and the next
        distinct value drops to the rank it would have had without the tie —
        the `rank`/`rnt` pair in `_calc_tree_fs()` (fair_tree.c L338-L409).
        On a flat tree each account is its own leaf, so there is nothing to
        merge on a tie.
        """
        accounts = list(self.shares)
        n = len(accounts)
        if n == 0:
            return {}
        level = {a: self.level_fairshare(a) for a in accounts}
        ordered = sorted(accounts, key=lambda a: -level[a])

        factors: dict[str, float] = {}
        rank = rnt = n
        previous: float | None = None
        for account in ordered:
            if previous is None or level[account] != previous:
                rank = rnt
            factors[account] = rank / n
            rnt -= 1
            previous = level[account]
        return factors

    def factors(self) -> dict[str, float]:
        """Current factor for every account in `shares`, per `algorithm`."""
        if self.algorithm == "fair_tree":
            return self.fair_tree_factors()
        return {account: self.classic_factor(account) for account in self.shares}

    def factor(self, account: str) -> float:
        """Current factor for one account, computed from usage as it is now.

        Accounts with no share allocation get 0, as a job without an
        association gets no fairshare priority in Slurm.
        """
        if self.algorithm == "fair_tree":
            return self.fair_tree_factors().get(account, 0.0)
        return self.classic_factor(account)


def age_factor(job: Job, now: float, max_age: float) -> float:
    """Time eligible in the queue over `PriorityMaxAge`, capped at 1.

    Measured from submit — or, for a requeued job, from its new begin time:
    `batch_requeue_fini()` (job_mgr.c) zeroes `details->accrue_time`, and it is
    set back to `begin_time` when the job becomes eligible again
    (`acct_policy_handle_accrue_time()`, acct_policy.c; SchedMD/slurm@9f9da53).
    """
    if max_age <= 0:
        return 0.0
    return min(max(now - job.accrue_start, 0.0) / max_age, 1.0)


def jobsize_factor(job: Job, cluster: Cluster, favor_small: bool) -> float:
    """Slurm's job size factor: the node fraction and CPU fraction, averaged.

    `set_priority_factors()` in priority_multifactor.c (SchedMD/slurm@9f9da53
    L2175-L2244), without `PriorityFlags=SMALL_RELATIVE_TO_TIME`:

    * favor large (the default): `(min_nodes / node_count + cpu_cnt /
      cluster_cpus) / 2`;
    * `PriorityFavorSmall`: `((node_count - min_nodes) / node_count` (0 once
      the job asks for every node) `+ (cluster_cpus - cpu_cnt) /
      cluster_cpus) / 2`;

    clamped to [0, 1]. `node_count` is the controller's active node count;
    every node here is active, so it is the cluster's node count. `cpu_cnt`
    for a pending job comes from its request (`max_cpus`, else `min_cpus`);
    here that is `nodes × cpus_per_node`. The docs: the factor "correlates to
    the number of nodes or CPUs the job has requested"
    (https://slurm.schedmd.com/priority_multifactor.html).

    For a whole-node job on a homogeneous cluster the two fractions are
    equal, so this is the CPU fraction (the formula 0.1.0 used, and still
    the value for every job of this repo's synthetic workload). They differ
    for sub-node jobs: an S0 gang of G one-CPU pods on a 26-node, 1680-CPU
    fleet gets about G/52 here, where the CPU fraction alone gave G/1680.
    """
    node_count = len(cluster.nodes)
    total = cluster.total_cpus
    if node_count <= 0 or total <= 0:
        return 0.0
    cpus = job.total_cpus
    if favor_small:
        factor = (node_count - job.nodes) / node_count if node_count > job.nodes else 0.0
        if cpus:
            factor = (factor + (total - cpus) / total) / 2
    else:
        factor = job.nodes / node_count
        if cpus:
            factor = (factor + cpus / total) / 2
    return min(max(factor, 0.0), 1.0)


#: Largest priority Slurm can store: `job_ptr->priority` is a `uint32_t`.
MAX_PRIORITY = 0xFFFF_FFFF


def controller_priority(value: float) -> int:
    """The priority the controller stores for a weighted sum `value`.

    `_get_priority_internal()` (priority_multifactor.c, SchedMD/slurm@9f9da53
    L655-L664, L787) raises anything below 1 to 1 ("Priority 0 is reserved
    for held jobs"), caps at 2^32 - 1 and returns `(uint32_t)priority` — a
    truncation, which for values of at least 1 is the floor.
    `sort_job_queue2()` (job_scheduler.c L2179) compares those integers, so
    two jobs whose sums differ by less than one can tie, and then the earlier
    submit goes first.
    """
    if not value >= 1:  # also catches NaN
        return 1
    if value >= MAX_PRIORITY:
        return MAX_PRIORITY
    return int(value)


def partition_factor(job: Job, partitions: dict[str, float]) -> float:
    return partitions.get(job.partition, 0.0)


def compute_priority(
    job: Job,
    now: float,
    cluster: Cluster,
    weights: PriorityWeights,
    fairshare: FairshareTree | None = None,
    partitions: dict[str, float] | None = None,
    *,
    fs_factor: float | None = None,
) -> float:
    """The weighted sum for one pending job at time `now`, as a float.

    `fs_factor`, when given, overrides reading the tree. The simulator passes
    the factor that was *visible* at the last `PriorityCalcPeriod` tick, which
    is what makes usage lag a modelled property rather than an accident.

    This is the sum before the controller stores it; the simulator orders
    the queue by `controller_priority()` of it, as Slurm does.
    """
    partitions = partitions or {}
    if fs_factor is None:
        fs_factor = fairshare.factor(job.account) if fairshare else 0.0

    return (
        weights.age * age_factor(job, now, weights.max_age)
        + weights.fairshare * fs_factor
        + weights.jobsize * jobsize_factor(job, cluster, weights.favor_small)
        + weights.partition * partition_factor(job, partitions)
        + weights.qos * job.qos_factor
        - job.nice
    )
