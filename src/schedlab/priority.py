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
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import Cluster, Job

SECONDS_PER_DAY = 86_400.0


@dataclass
class PriorityWeights:
    """Maps 1:1 onto the `PriorityWeight*` settings in slurm.conf."""

    age: float = 1_000.0
    fairshare: float = 10_000.0
    jobsize: float = 10_000.0
    partition: float = 1_000.0
    qos: float = 10_000.0

    #: `PriorityMaxAge` — the age at which age_factor saturates at 1.0.
    max_age: float = 7 * SECONDS_PER_DAY
    #: `PriorityFavorSmall` — when true, small jobs get the larger factor.
    favor_small: bool = False
    #: `PriorityDecayHalfLife` — how fast recorded usage decays for fairshare.
    decay_half_life: float = 5 * SECONDS_PER_DAY

    @classmethod
    def from_slurm_conf(cls, path: str) -> PriorityWeights:
        """Read the `PriorityWeight*` lines out of a real slurm.conf.

        Lets you point the simulator at the config you are about to deploy
        instead of re-typing its numbers.
        """
        parsed: dict[str, str] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                parsed[key.strip().lower()] = value.strip()

        weights = cls()
        for attr, key in (
            ("age", "priorityweightage"),
            ("fairshare", "priorityweightfairshare"),
            ("jobsize", "priorityweightjobsize"),
            ("partition", "priorityweightpartition"),
            ("qos", "priorityweightqos"),
        ):
            if key in parsed:
                setattr(weights, attr, float(parsed[key]))

        if "prioritymaxage" in parsed:
            weights.max_age = _parse_duration(parsed["prioritymaxage"])
        if "prioritydecayhalflife" in parsed:
            weights.decay_half_life = _parse_duration(parsed["prioritydecayhalflife"])
        if "priorityfavorsmall" in parsed:
            weights.favor_small = parsed["priorityfavorsmall"].upper() in {"YES", "1", "TRUE"}

        return weights


def _parse_duration(text: str) -> float:
    """Parse Slurm's duration syntax: `days-hours:minutes:seconds`."""
    days = 0.0
    if "-" in text:
        day_part, _, text = text.partition("-")
        days = float(day_part)

    parts = [float(p) for p in text.split(":")] if text else [0.0]
    while len(parts) < 3:
        parts.append(0.0)
    hours, minutes, seconds = parts[:3]
    return days * SECONDS_PER_DAY + hours * 3600 + minutes * 60 + seconds


@dataclass
class FairshareTree:
    """Flat single-level fairshare — one share allocation per account.

    Slurm supports a hierarchy; a flat tree is enough to reproduce the effect
    people actually feel, which is a heavy account being pushed down the queue.
    """

    shares: dict[str, float]
    usage: dict[str, float] = field(default_factory=dict)
    half_life: float = 5 * SECONDS_PER_DAY
    _last_decay: float = 0.0

    def normalized_shares(self, account: str) -> float:
        total = sum(self.shares.values())
        if total <= 0:
            return 0.0
        return self.shares.get(account, 0.0) / total

    def decay(self, now: float) -> None:
        """Apply exponential decay so old usage stops counting against you."""
        elapsed = now - self._last_decay
        if elapsed <= 0 or self.half_life <= 0:
            return
        factor = 0.5 ** (elapsed / self.half_life)
        for account in self.usage:
            self.usage[account] *= factor
        self._last_decay = now

    def charge(self, account: str, cpu_seconds: float) -> None:
        self.usage[account] = self.usage.get(account, 0.0) + cpu_seconds

    def factor(self, account: str) -> float:
        """Slurm's classic fairshare: F = 2^(-U_norm / S_norm).

        F = 0.5 means the account is using exactly its share. Above means
        under-served (priority boost), below means over-served.
        """
        shares = self.normalized_shares(account)
        if shares <= 0:
            return 0.0

        total_usage = sum(self.usage.values())
        if total_usage <= 0:
            return 1.0

        usage_norm = self.usage.get(account, 0.0) / total_usage
        return 2.0 ** (-usage_norm / shares)


def age_factor(job: Job, now: float, max_age: float) -> float:
    if max_age <= 0:
        return 0.0
    return min(max(now - job.submit_time, 0.0) / max_age, 1.0)


def jobsize_factor(job: Job, cluster: Cluster, favor_small: bool) -> float:
    total = cluster.total_cpus
    if total <= 0:
        return 0.0
    factor = min(job.total_cpus / total, 1.0)
    return 1.0 - factor if favor_small else factor


def partition_factor(job: Job, partitions: dict[str, float]) -> float:
    return partitions.get(job.partition, 0.0)


def compute_priority(
    job: Job,
    now: float,
    cluster: Cluster,
    weights: PriorityWeights,
    fairshare: FairshareTree | None = None,
    partitions: dict[str, float] | None = None,
) -> float:
    """Composite priority for one pending job at time `now`."""
    partitions = partitions or {}
    fs_factor = fairshare.factor(job.account) if fairshare else 0.0

    return (
        weights.age * age_factor(job, now, weights.max_age)
        + weights.fairshare * fs_factor
        + weights.jobsize * jobsize_factor(job, cluster, weights.favor_small)
        + weights.partition * partition_factor(job, partitions)
        + weights.qos * job.qos_factor
        - job.nice
    )
