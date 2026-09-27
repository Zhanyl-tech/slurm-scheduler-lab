"""Everything the simulator reads from one slurm.conf, in one place."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from .params import SchedulerParameters
from .preempt import PREEMPT_TYPES, PreemptionConfig, PreemptMode
from .priority import PriorityWeights
from .slurmconf import parse_time_str, read_slurm_conf

#: slurm.conf(5) / sched_config.html: "Options are sched/backfill [...] and
#: sched/builtin". "The backfill scheduling plugin is loaded by default."
SCHEDULER_TYPES = ("sched/backfill", "sched/builtin")

#: `PreemptParameters` options acted on. The rest are reported.
_MODELLED_PREEMPT_PARAMS = frozenset({"youngest_first", "reorder_count", "strict_order"})

#: slurm.conf keys that change job priority, queue order, node choice or run
#: time but are not modelled, so they are reported when present. Each maps to
#: the value at which the key has no effect (no warning then; None: always
#: warn) and what it does that the simulator leaves out. Keys with no
#: scheduling effect (SlurmctldHost, log levels, paths) are not listed and
#: are ignored silently. Semantics from slurm.conf(5), Slurm 26.05.
_UNMODELLED_KEYS: dict[str, tuple[str | None, str]] = {
    "fairsharedampeningfactor": (
        "1",
        "dampens the fairshare factor of over-served associations; the simulator uses "
        "Slurm's default, 1",
    ),
    "prioritysitefactorplugin": (None, "adds a site factor to every job's priority"),
    "prioritysitefactorparameters": (None, "configures the site factor plugin"),
    "priorityparameters": (None, "is passed to the priority plugin"),
    "overtimelimit": (
        "0",
        "lets a job run past its time limit; here a runtime never exceeds its limit",
    ),
    "topologyplugin": (
        "topology/flat",
        "changes node selection; here it is first-fit by node id (declare racks and "
        "switches in a --fleet file for the placement metrics)",
    ),
    "selecttypeparameters": (
        None,
        "changes what an allocation consumes; here a job takes its CPUs and GPUs on "
        "each node, and memory and cores are not tracked",
    ),
    "partitionname": (
        None,
        "lines are not parsed: every job shares one partition spanning every node, so "
        "a partition's Nodes, PriorityTier, PriorityJobFactor, MaxTime, GraceTime and "
        "PreemptMode have no effect",
    ),
    "nodename": (
        None,
        "lines are not read: the cluster comes from --nodes/--cpus/--gpus or --fleet",
    ),
}

#: slurm.conf(5) spellings for the warning text; the parser lower-cases keys.
_KEY_NAMES = {
    "fairsharedampeningfactor": "FairShareDampeningFactor",
    "prioritysitefactorplugin": "PrioritySiteFactorPlugin",
    "prioritysitefactorparameters": "PrioritySiteFactorParameters",
    "priorityparameters": "PriorityParameters",
    "overtimelimit": "OverTimeLimit",
    "topologyplugin": "TopologyPlugin",
    "selecttypeparameters": "SelectTypeParameters",
    "partitionname": "PartitionName",
    "nodename": "NodeName",
}


def unmodelled_key_warnings(parsed: Mapping[str, str]) -> list[str]:
    """One warning per scheduling-relevant key that is present but not applied.

    `read_slurm_conf()` keeps the last of repeated keys, so several
    PartitionName lines give one warning.
    """
    warnings: list[str] = []
    for key, (neutral, effect) in _UNMODELLED_KEYS.items():
        if key not in parsed:
            continue
        value = parsed[key].strip()
        if neutral is not None and value.lower() == neutral:
            continue
        name = _KEY_NAMES.get(key, key)
        if key in {"partitionname", "nodename"}:
            warnings.append(f"{name} {effect}")
        else:
            warnings.append(f"{name}={value} is not modelled: it {effect}")

    weight = parsed.get("priorityweightpartition", "0").strip()
    try:
        partition_weight = float(weight)
    except ValueError:
        partition_weight = 0.0  # PriorityWeights.from_conf has already refused it
    if partition_weight:
        warnings.append(
            f"PriorityWeightPartition={weight} has no effect: the partition factor comes "
            "from PartitionName PriorityJobFactor, which is not parsed, so every job's "
            "partition factor is 0"
        )
    return warnings


@dataclass
class SlurmConfig:
    """Priority settings, scheduler parameters and scheduler type together.

    `warnings` lists every setting that was present, would change what Slurm
    schedules, and is not modelled, so a run can print what it ignored
    instead of silently pretending. The keys checked are the priority,
    scheduler and preemption keys this module reads, plus `_UNMODELLED_KEYS`;
    keys with no scheduling effect are ignored silently.

    `preemption` holds the cluster-wide preemption settings. Partition and
    QOS tables are not read from slurm.conf (QOSes live in the accounting
    database, and `PartitionName` lines are not parsed); the CLI builds them
    from the trace (see README "Preemption").
    """

    priority: PriorityWeights = field(default_factory=PriorityWeights)
    scheduler: SchedulerParameters = field(default_factory=SchedulerParameters)
    scheduler_type: str = "sched/backfill"
    warnings: list[str] = field(default_factory=list)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)

    @property
    def backfill_enabled(self) -> bool:
        """True unless `SchedulerType=sched/builtin` or `bf_interval=-1`."""
        return self.scheduler_type == "sched/backfill" and self.scheduler.backfill_enabled

    @classmethod
    def load(cls, path: str, sched_params: str | None = None) -> SlurmConfig:
        return cls.from_conf(read_slurm_conf(path), sched_params=sched_params)

    @classmethod
    def from_conf(
        cls, parsed: Mapping[str, str], sched_params: str | None = None
    ) -> SlurmConfig:
        """Read `parsed`; `sched_params`, if given, replaces its SchedulerParameters.

        Replaced whole, before anything reads it: the warnings about the
        config's own string and its `requeue_delay` go with it. (The CLI's
        `--sched-params` used to swap the parsed parameters in afterwards,
        so the header printed warnings about the discarded string and the
        config's requeue_delay stayed in force.)
        """
        priority, warnings = PriorityWeights.from_conf(parsed)
        if sched_params is None:
            scheduler = SchedulerParameters.from_conf(parsed)
        else:
            scheduler = SchedulerParameters.parse(sched_params)
        warnings = list(warnings) + scheduler_warnings(scheduler)
        scheduler_type = parsed.get("schedulertype", "sched/backfill").lower()
        if scheduler_type not in SCHEDULER_TYPES:
            warnings.append(f"SchedulerType={scheduler_type} is not modelled; using sched/backfill")
            scheduler_type = "sched/backfill"
        preemption, preempt_warnings = preemption_from_conf(parsed, scheduler)
        warnings += preempt_warnings
        warnings += unmodelled_key_warnings(parsed)
        return cls(priority, scheduler, scheduler_type, warnings, preemption)


def scheduler_warnings(scheduler: SchedulerParameters) -> list[str]:
    """What a run header says about one SchedulerParameters string."""
    warnings = list(scheduler.warnings)
    if scheduler.unmodelled:
        warnings.append("SchedulerParameters not modelled: " + ", ".join(scheduler.unmodelled))
    return warnings


def preemption_from_conf(
    parsed: Mapping[str, str], scheduler: SchedulerParameters | None = None
) -> tuple[PreemptionConfig, list[str]]:
    """The cluster-wide preemption keys of a `read_slurm_conf()` mapping.

    Mirrors read_config.c (SchedMD/slurm@9f9da53): `PreemptType` unset or
    `preempt/none` with a PreemptMode other than OFF, or a preempt plugin with
    PreemptMode OFF, is a configuration slurmctld refuses, so this raises.
    """
    warnings: list[str] = []
    ptype = parsed.get("preempttype", "preempt/none").strip().lower() or "preempt/none"
    if ptype not in PREEMPT_TYPES:
        raise ValueError(f"PreemptType={ptype} is not modelled (have {', '.join(PREEMPT_TYPES)})")
    mode = PreemptMode.parse(parsed.get("preemptmode", "OFF"))
    if mode.gang:
        warnings.append("PreemptMode=GANG (time-slicing) is not modelled")

    exempt = 0.0
    if "preemptexempttime" in parsed:
        # read_config.c parses it with time_str2secs(): a bare number is minutes.
        value = parse_time_str(parsed["preemptexempttime"])
        exempt = 0.0 if math.isinf(value) or value < 0 else value

    youngest = strict = False
    reorder = 1
    for raw in parsed.get("preemptparameters", "").split(","):
        item = raw.strip()
        if not item:
            continue
        key, _, arg = item.partition("=")
        key = key.strip().lower()
        # `_MODELLED_PREEMPT_PARAMS` alone decides what is reported; each of
        # its keys has a branch below (a test checks each one has an effect).
        if key not in _MODELLED_PREEMPT_PARAMS:
            warnings.append(f"PreemptParameters {key} is parsed but not modelled")
        elif key == "youngest_first":
            youngest = True
        elif key == "strict_order":
            strict = True
        elif key == "reorder_count":
            try:
                reorder = int(arg)
            except ValueError:
                raise ValueError(f"PreemptParameters reorder_count={arg!r}") from None

    job_requeue = parsed.get("jobrequeue", "1").strip() != "0"
    requeue_delay = 120.0
    if scheduler is not None and scheduler.requeue_delay is not None:
        requeue_delay = scheduler.requeue_delay
    elif "authinfo" in parsed and "cred_expire" in parsed["authinfo"].lower():
        warnings.append(
            "AuthInfo cred_expire is not read; requeue_delay defaults to 120 s "
            "(set SchedulerParameters=requeue_delay=N to override)"
        )

    config = PreemptionConfig(
        preempt_type=ptype,  # type: ignore[arg-type]
        mode=mode,
        exempt_time=exempt,
        youngest_first=youngest,
        reorder_count=reorder,
        strict_order=strict,
        job_requeue=job_requeue,
        requeue_delay=requeue_delay,
        kill_wait=float(parsed.get("killwait", "30")),
        message_timeout=float(parsed.get("messagetimeout", "10")),
    )
    return config, warnings
