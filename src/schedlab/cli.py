"""Command line entry point."""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from . import fleet as fleet_mod
from . import k8strace, parity
from .config import SlurmConfig, scheduler_warnings
from .metrics import Metrics, compute
from .model import Cluster, Job
from .params import SchedulerParameters
from .preempt import (
    PartitionSpec,
    PreemptionConfig,
    PreemptMode,
    QOSSpec,
    partition_ladder,
    qos_ladder,
)
from .priority import FAIRSHARE_ALGORITHMS, PriorityWeights
from .simulate import BACKFILL_MODES, BackfillMode, SimulationResult, simulate
from .slurmconf import parse_minutes_str, parse_time_str
from .trace import UNLIMITED_PLANNING_LIMIT, WorkloadProfile, generate, load_sacct


@dataclass
class Workload:
    """The jobs to run, and what the trace source added to the config."""

    jobs: list[Job]
    source: str
    digest: str | None = None
    qos: dict[str, QOSSpec] = field(default_factory=dict)
    partitions: dict[str, PartitionSpec] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    #: Where t=0 falls in the week (seconds after Sunday 00:00), for DAILY and
    #: WEEKLY usage resets, and the timestamp it came from. Known for an sacct
    #: trace only; otherwise 0 and "" (t=0 is taken as Sunday 00:00).
    calendar_offset: float = 0.0
    calendar_origin: str = ""


def _load_workload(args: argparse.Namespace, grace: float) -> Workload:
    if args.k8s_trace:
        s0 = k8strace.load(
            args.k8s_trace,
            time_limit_model=args.time_limit_model,
            time_limit_factor=args.time_limit_factor,
            seed=args.seed,
            priority_mapping=args.k8s_priority,
            cpus_per_pod=args.k8s_cpus_per_pod,
            grace_time=grace,
        )
        return Workload(
            s0.jobs,
            f"{args.k8s_trace} (k8s lab CSV)",
            digest=s0.digest,
            qos=s0.qos,
            partitions=s0.partitions,
            notes=[s0.describe()],
            model={
                "trace": "k8s",
                "time_limit_model": s0.time_limit_model,
                "time_limit_factor": s0.time_limit_factor,
                # Seeds the synthetic limits; the CSV itself is in the digest.
                "seed": args.seed if s0.time_limit_model == "synthetic" else None,
                "priority_mapping": s0.priority_mapping,
                "cpus_per_pod": s0.cpus_per_pod,
            },
        )
    if args.sacct:
        planning = UNLIMITED_PLANNING_LIMIT
        if args.partition_max_time is not None:
            # slurm.conf MaxTime is read with time_str2mins() (read_config.c):
            # whole minutes, rounded up, like every other Slurm time limit.
            planning = parse_minutes_str(args.partition_max_time)
            if math.isinf(planning):
                planning = UNLIMITED_PLANNING_LIMIT  # an UNLIMITED MaxTime: backfill's year
            elif planning <= 0:
                raise ValueError("--partition-max-time must be a positive time")
        sacct = load_sacct(args.sacct, planning_limit=planning)
        return Workload(
            sacct.jobs,
            args.sacct,
            notes=list(sacct.notes),
            model={
                "trace": "sacct",
                "planning_limit_seconds": sacct.planning_limit,
                "no_limit_jobs": sacct.no_limit_jobs,
                "over_limit_jobs": sacct.over_limit_jobs,
                "over_limit_seconds": sacct.over_limit_seconds,
                "calendar_offset_seconds": sacct.calendar_offset,
            },
            calendar_offset=sacct.calendar_offset,
            calendar_origin=sacct.origin,
        )
    jobs = _synthetic_jobs(args, args.seed)
    work = Workload(
        jobs,
        f"synthetic (seed={args.seed})",
        # The seed and job count are the whole synthetic trace.
        model={"trace": "synthetic", "seed": args.seed, "jobs": args.jobs},
    )
    if args.time_limit_model:
        work.notes.append(f"time limits: {_limit_label(args)}")
        work.model["time_limit_model"] = args.time_limit_model
        work.model["time_limit_factor"] = args.time_limit_factor
    return work


def _limit_label(args: argparse.Namespace) -> str:
    if args.time_limit_model == "padded":
        return f"padded ×{args.time_limit_factor:g}"
    if args.time_limit_model == "exact":
        return "exact (limit = runtime, rounded up to a whole minute)"
    return "synthetic (the generator's own padding)"


def _synthetic_jobs(args: argparse.Namespace, seed: int) -> list[Job]:
    """The synthetic trace for `seed`, with `--time-limit-model` applied.

    Only the requested limits change: every job keeps the generator's
    submit time, runtime, shape and account, so a comparison between models
    isolates what users ask for. `synthetic` is the generator's own padding
    (the same max(1.05, N(3, 1)) distribution the S0 adapter uses), which
    is also what omitting the flag gives. `exact` and `padded` are the S0
    adapter's own definitions (`k8strace.time_limits`), factor check included.
    """
    jobs = generate(WorkloadProfile(job_count=args.jobs), seed=seed)
    if args.time_limit_model in {"exact", "padded"}:
        limits = k8strace.time_limits(jobs, args.time_limit_model, args.time_limit_factor)
        for job, limit in zip(jobs, limits, strict=True):
            job.time_limit = limit
    return jobs


def _synthetic_ladder(jobs: list[Job], kind: str, grace: float) -> Workload:
    """QOSes or partitions from the synthetic trace's qos_factor levels.

    The generator draws `qos_factor` from a few levels; each level becomes a
    QOS (`preempt/qos`: every level preempts the lower ones) or a partition
    (`preempt/partition_prio`: PriorityTier = rank). `qos_factor` itself is
    unchanged, so priorities are too.
    """
    levels = sorted({j.qos_factor for j in jobs})
    names = {level: f"q{level:g}" for level in levels}
    out = Workload(jobs, "")
    if kind == "preempt/qos":
        out.qos = qos_ladder({names[lv]: rank for rank, lv in enumerate(levels)}, grace)
        for j in jobs:
            j.qos = names[j.qos_factor]
    else:
        out.partitions = partition_ladder(
            {names[lv]: rank for rank, lv in enumerate(levels, start=1)}, grace
        )
        for j in jobs:
            j.partition = names[j.qos_factor]
    return out


def _cluster_factory(args: argparse.Namespace) -> tuple[Any, str]:
    """A callable returning a fresh cluster per run, and a header line."""
    if args.fleet:
        flt = fleet_mod.load(args.fleet)
        return flt.build_cluster, flt.describe()
    return (
        lambda: Cluster.homogeneous(args.nodes, args.cpus, args.gpus),
        f"{args.nodes} nodes × {args.cpus} CPU / {args.gpus} GPU",
    )


def _config(args: argparse.Namespace) -> SlurmConfig:
    """The slurm.conf (or lab defaults), with command-line overrides applied."""
    if args.slurm_conf:
        # --sched-params replaces the config's SchedulerParameters before
        # anything reads them: its warnings and requeue_delay go too.
        cfg = SlurmConfig.load(args.slurm_conf, sched_params=args.sched_params)
    else:
        cfg = SlurmConfig()
        if args.sched_params is not None:
            cfg.scheduler = SchedulerParameters.parse(args.sched_params)
            cfg.warnings += scheduler_warnings(cfg.scheduler)
            if cfg.scheduler.requeue_delay is not None:
                cfg.preemption = replace(
                    cfg.preemption, requeue_delay=cfg.scheduler.requeue_delay
                )
    weights = cfg.priority
    if args.fairshare_algorithm:
        weights = replace(weights, fairshare_algorithm=args.fairshare_algorithm)
    if args.calc_period is not None:
        if args.calc_period < 0:
            raise ValueError("--calc-period must be >= 0")
        minutes = float(math.ceil(args.calc_period))
        if minutes != args.calc_period:
            # time_str2mins() rounds up to whole minutes (parse_time.c
            # L841-L847); a Slurm period is never a fraction of a minute.
            cfg.warnings.append(
                f"--calc-period {args.calc_period:g}: PriorityCalcPeriod is whole minutes in "
                f"Slurm, which rounds up; using {minutes:g} min"
            )
        weights = replace(weights, calc_period=minutes * 60.0)
    if weights.usage_reset_period in {"MONTHLY", "QUARTERLY", "YEARLY"}:
        # The library (PriorityEngine) refuses these. From a slurm.conf the
        # CLI does what it does for other config settings it cannot apply:
        # it says so, and runs without.
        cfg.warnings.append(
            f"PriorityUsageResetPeriod={weights.usage_reset_period} needs a calendar the "
            "simulator does not have; simulating as NONE"
        )
        weights = replace(weights, usage_reset_period="NONE")
    cfg.priority = weights

    p = cfg.preemption
    ptype: str = p.preempt_type
    if args.preempt_type is not None:
        ptype = "preempt/none" if args.preempt_type == "none" else f"preempt/{args.preempt_type}"
    mode = PreemptMode.parse(args.preempt_mode) if args.preempt_mode else p.mode
    if ptype == "preempt/none" and args.preempt_type == "none":
        mode = PreemptMode()
    elif args.preempt_mode and mode.gang:
        # As config.preemption_from_conf says for a slurm.conf GANG.
        cfg.warnings.append("PreemptMode=GANG (time-slicing) is not modelled")
    exempt = p.exempt_time
    if args.preempt_exempt_time is not None:
        exempt = parse_time_str(args.preempt_exempt_time)
    if ptype != "preempt/none" and mode.base == "OFF":
        raise ValueError(
            f"{ptype} is not compatible with PreemptMode=OFF (slurmctld refuses it); "
            "pass --preempt-mode REQUEUE or CANCEL"
        )
    cfg.preemption = PreemptionConfig(
        preempt_type=ptype,  # type: ignore[arg-type]
        mode=mode,
        exempt_time=exempt,
        youngest_first=p.youngest_first,
        reorder_count=p.reorder_count,
        strict_order=p.strict_order,
        job_requeue=p.job_requeue,
        requeue_delay=p.requeue_delay,
        kill_wait=p.kill_wait,
        message_timeout=p.message_timeout,
        checkpoint_fraction=(
            args.checkpoint_fraction
            if args.checkpoint_fraction is not None
            else p.checkpoint_fraction
        ),
    )
    return cfg


def _attach_tables(cfg: SlurmConfig, work: Workload, args: argparse.Namespace) -> None:
    """Give the preemption config the QOS / partition tables the trace implies."""
    p = cfg.preemption
    grace = args.grace_time or 0.0
    if not p.enabled:
        unused = [
            flag
            for flag, value in (
                # With --preempt-type none, _config discards a --preempt-mode.
                ("--preempt-mode", args.preempt_mode),
                ("--grace-time", args.grace_time),
                ("--checkpoint-fraction", args.checkpoint_fraction),
                ("--preempt-exempt-time", args.preempt_exempt_time),
            )
            if value is not None
        ]
        if unused:
            cfg.warnings.append(f"{', '.join(unused)} ignored: preemption is off")
    if p.enabled:
        reason = None
        if args.sacct:
            reason = (
                "sacct traces are not read for per-job QOS or partitions, so there is "
                "nothing to preempt by"
            )
        elif args.backfill_mode == "easy":
            reason = "preemption is modelled in the conservative mode only"
        if reason is not None:
            if args.preempt_type is not None:
                raise ValueError(reason)
            # From a slurm.conf: say so and run without, rather than refuse the run.
            cfg.warnings.append(f"{p.preempt_type} in the config is not applied: {reason}")
            cfg.preemption = PreemptionConfig()
            return
        if args.k8s_trace:
            want = "qos" if p.preempt_type == "preempt/qos" else "tier"
            if args.k8s_priority != want:
                raise ValueError(f"{p.preempt_type} needs --k8s-priority {want}")
        else:
            ladder = _synthetic_ladder(work.jobs, p.preempt_type, grace)
            work.qos, work.partitions = ladder.qos, ladder.partitions
    if work.qos or work.partitions:
        cfg.preemption = replace(p, qos=work.qos, partitions=work.partitions)


def _trace_warnings(cfg: SlurmConfig, work: Workload, args: argparse.Namespace) -> None:
    """Settings the chosen trace cannot honour, said rather than ignored."""
    w = cfg.priority
    if args.sacct and w.qos:
        cfg.warnings.append(
            f"PriorityWeightQOS={w.qos:g} has no effect on an sacct trace: its QOS column "
            "is not read (QOS priorities live in the accounting database), so every job's "
            "QOS factor is 0"
        )
    if w.usage_reset_period in {"DAILY", "WEEKLY"}:
        boundary = "midnight" if w.usage_reset_period == "DAILY" else "Sunday 00:00"
        if work.calendar_origin:
            work.notes.append(
                f"PriorityUsageResetPeriod={w.usage_reset_period}: resets at {boundary} on "
                f"the trace's Submit clock; t=0 is {work.calendar_origin}"
            )
        else:
            cfg.warnings.append(
                f"PriorityUsageResetPeriod={w.usage_reset_period}: this trace has no "
                f"calendar, so t=0 is taken as Sunday 00:00 and resets fall at {boundary} "
                "counted from it (an sacct trace places them from its Submit times)"
            )


def _simulate(
    jobs: list[Job],
    args: argparse.Namespace,
    cfg: SlurmConfig,
    weights: PriorityWeights,
    backfill: bool,
    make_cluster: Any,
    calendar_offset: float = 0.0,
) -> tuple[SimulationResult, Cluster]:
    # Each run needs its own job objects and cluster — simulation mutates both.
    run_jobs = copy.deepcopy(jobs)
    cluster: Cluster = make_cluster()
    mode: BackfillMode = args.backfill_mode
    uses_table = cfg.preemption.enabled or bool(cfg.preemption.partitions)
    result = simulate(
        run_jobs,
        cluster,
        weights=weights,
        fairshare=weights.make_fairshare(job.account for job in run_jobs),
        backfill=backfill,
        backfill_mode=mode,
        sched_params=cfg.scheduler,
        calendar_offset=calendar_offset,
        preemption=cfg.preemption if uses_table else None,
    )
    return result, cluster


def _run(
    jobs: list[Job],
    args: argparse.Namespace,
    cfg: SlurmConfig,
    weights: PriorityWeights,
    backfill: bool,
    make_cluster: Any,
    calendar_offset: float = 0.0,
) -> Metrics:
    result, cluster = _simulate(
        jobs, args, cfg, weights, backfill, make_cluster, calendar_offset
    )
    return compute(result, cluster)


def _backfill_disabled_by(cfg: SlurmConfig) -> list[str]:
    """The config settings that turn backfill off, as the header names them."""
    reasons = []
    if cfg.scheduler_type == "sched/builtin":
        reasons.append("SchedulerType=sched/builtin")
    if not cfg.scheduler.backfill_enabled:
        reasons.append("bf_interval=-1")
    return reasons


def _backfill_on(cfg: SlurmConfig, mode: str) -> tuple[SlurmConfig, str | None]:
    """The config for `--compare-backfill`'s ON leg, and a note if it overrides one.

    One rule for both ways a config can disable backfill: the ON leg runs
    backfill anyway, because that is the comparison asked for, and the
    header says what was overridden. (Before, `SchedulerType=sched/builtin`
    was overridden silently while `bf_interval=-1` was honoured, so the ON
    leg repeated the OFF leg.) The OFF leg is the config as given.
    """
    reasons = _backfill_disabled_by(cfg)
    if not reasons:
        return cfg, None
    scheduler = cfg.scheduler
    how = "runs backfill anyway"
    if not scheduler.backfill_enabled:
        scheduler = replace(scheduler, bf_interval=SchedulerParameters().bf_interval)
        if mode == "conservative":
            how += f", with bf_interval={scheduler.bf_interval:g} s (Slurm's default)"
    on = replace(cfg, scheduler_type="sched/backfill", scheduler=scheduler)
    note = (
        f"--compare-backfill: the config disables backfill ({' and '.join(reasons)}); "
        f"the ON leg {how}, and the OFF leg is the config as given"
    )
    return on, note


def _scheduler_label(cfg: SlurmConfig, mode: str, compare: bool = False) -> str:
    """What actually runs: the backfill mode, or why backfill is off."""
    model = "easy" if mode == "easy" else "conservative"
    reasons = _backfill_disabled_by(cfg)
    if compare and reasons:
        # The ON leg overrides the config (`_backfill_on`); naming only the
        # config here would contradict half the output.
        return f"{model} model, backfill OFF vs ON"
    if mode == "conservative" and reasons and reasons[0] == "SchedulerType=sched/builtin":
        return "sched/builtin (no backfill)"
    if reasons:
        return f"{model} model, backfill disabled ({' and '.join(reasons)})"
    return f"{model} backfill"


def _describe(cfg: SlurmConfig, mode: str, compare: bool = False) -> list[str]:
    w = cfg.priority
    period = "every event (idealised)" if w.calc_period == 0 else f"{w.calc_period / 60:g} min"
    lines = [
        f"model: {_scheduler_label(cfg, mode, compare)} · fairshare {w.fairshare_algorithm} · "
        f"PriorityCalcPeriod {period}"
    ]
    if mode == "conservative":
        lines.append(f"SchedulerParameters: {cfg.scheduler.describe()}")
    p = cfg.preemption
    if p.enabled:
        grace = sorted(
            {s.grace_time for s in p.qos.values()} | {s.grace_time for s in p.partitions.values()}
        )
        lines.append(
            f"preemption: {p.preempt_type} · PreemptMode={p.mode} · "
            f"GraceTime {'/'.join(f'{g:g}' for g in grace) or '0'} s · "
            f"requeue_delay {p.requeue_delay:g} s · checkpoint {p.checkpoint_fraction:g}"
        )
    elif p.partitions:
        tiers = ", ".join(f"{s.name}={s.priority_tier}" for s in p.partitions.values())
        lines.append(f"partitions (PriorityTier): {tiers}")
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="schedlab",
        description="Simulate Slurm priority and backfill policy against a job trace.",
    )
    parser.add_argument("--nodes", type=int, default=16, help="cluster node count")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per node")
    parser.add_argument("--gpus", type=int, default=2, help="GPUs per node")
    parser.add_argument(
        "--fleet",
        metavar="PATH",
        help="build a heterogeneous cluster from a k8s-gpu-scheduler-lab fleet YAML "
        "(replaces --nodes/--cpus/--gpus)",
    )
    parser.add_argument("--jobs", type=int, default=400, help="synthetic job count")
    parser.add_argument("--seed", type=int, default=0, help="trace seed")
    parser.add_argument("--sacct", help="replay a real trace from sacct output")
    parser.add_argument(
        "--partition-max-time",
        metavar="TIME",
        help="--sacct only: the limit to plan jobs with no finite Timelimit (UNLIMITED, "
        "Partition_Limit) at, as a Slurm time string (a bare number is minutes; seconds "
        "round up to a whole minute, as MaxTime does). Default UNLIMITED, which backfill "
        "plans as 365 days, as Slurm does",
    )
    parser.add_argument(
        "--k8s-trace",
        metavar="CSV",
        help="replay a k8s-gpu-scheduler-lab trace CSV (the S0 control); needs "
        "--time-limit-model",
    )
    parser.add_argument(
        "--time-limit-model",
        choices=k8strace.TIME_LIMIT_MODELS,
        help="requested time limits, each rounded up to a whole minute as Slurm stores "
        "them: exact (limit = runtime; not a best case, see README), padded (runtime × "
        "--time-limit-factor), synthetic (this repo's padding distribution). Required for "
        "--k8s-trace, which has none: the choice changes the answer. On a synthetic trace "
        "it replaces the generator's padding and nothing else",
    )
    parser.add_argument(
        "--time-limit-factor",
        type=float,
        metavar="F",
        help="padding factor for --time-limit-model padded (>= 1)",
    )
    parser.add_argument(
        "--k8s-priority",
        choices=k8strace.PRIORITY_MAPPINGS,
        default="qos",
        help="map k8s priority onto a QOS (additive PriorityWeightQOS factor; default), "
        "a partition PriorityTier (strict order), or ignore it",
    )
    parser.add_argument(
        "--k8s-cpus-per-pod",
        type=int,
        default=1,
        metavar="N",
        help="CPUs per node for each S0 job (default 1, the k8s lab pods' cpu request)",
    )
    parser.add_argument(
        "--slurm-conf",
        help="read Priority*, PriorityFlags, SchedulerType, SchedulerParameters and the "
        "cluster-wide Preempt* keys from a slurm.conf; omitted keys take Slurm's defaults",
    )
    parser.add_argument(
        "--backfill-mode",
        choices=BACKFILL_MODES,
        default="conservative",
        help="conservative: Slurm-like sched/backfill cycles on bf_interval, bounded by "
        "SchedulerParameters (default). easy: textbook EASY on every event, unbounded",
    )
    parser.add_argument(
        "--fairshare-algorithm",
        choices=FAIRSHARE_ALGORITHMS,
        help="override the fairshare algorithm (default: fair_tree, or classic when "
        "PriorityFlags has NO_FAIR_TREE)",
    )
    parser.add_argument(
        "--calc-period",
        type=float,
        metavar="MINUTES",
        help="override PriorityCalcPeriod, in whole minutes as in Slurm (a fraction is "
        "rounded up, with a warning); 0 refreshes priorities at every event, an "
        "idealisation Slurm does not allow",
    )
    parser.add_argument(
        "--sched-params",
        metavar="LIST",
        help="SchedulerParameters value to use instead of the config's, "
        "e.g. bf_max_job_test=50,bf_window=120",
    )
    parser.add_argument(
        "--preempt-type",
        choices=["none", "partition_prio", "qos"],
        help="PreemptType (default: the config's, i.e. none). Off by default",
    )
    parser.add_argument(
        "--preempt-mode",
        metavar="MODE",
        help="PreemptMode: CANCEL or REQUEUE, optionally with ,WITHIN or ,PRIORITY",
    )
    parser.add_argument(
        "--grace-time",
        type=float,
        metavar="SECONDS",
        help="GraceTime for every preemptable QOS / partition (default 0)",
    )
    parser.add_argument(
        "--checkpoint-fraction",
        type=float,
        metavar="F",
        help="share of a preempted run's progress a requeued job keeps (lab-only; default 0)",
    )
    parser.add_argument(
        "--preempt-exempt-time",
        metavar="TIME",
        help="PreemptExemptTime as a Slurm time string (a bare number is minutes)",
    )
    parser.add_argument(
        "--compare-backfill",
        action="store_true",
        help="run with and without backfill and show the delta",
    )
    parser.add_argument(
        "--sweep",
        metavar="FACTOR",
        choices=["age", "fairshare", "jobsize", "qos"],
        help="sweep one PriorityWeight across several magnitudes",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        metavar="N",
        help="repeat a single synthetic run over N consecutive seeds starting at --seed and "
        "print the spread (with work lost and grace-locked CPU-hours when preemption is "
        "on); one seed is an anecdote",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write the run's metrics as JSON, with the k8s lab's results.json field "
        "names where the quantities coincide (single run only)",
    )
    parser.add_argument(
        "--config-name",
        metavar="NAME",
        help="the `config` label in --json output (default S0 for a k8s trace, else slurm)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    for flag, value, low in (
        # The fleet loader refuses the same values (fleet.NodeClass); a
        # cluster of -2 GPUs per node used to run and print a report.
        ("--nodes", args.nodes, 1),
        ("--cpus", args.cpus, 1),
        ("--gpus", args.gpus, 0),
        ("--jobs", args.jobs, 1),
    ):
        if value < low:
            parser.error(f"{flag} must be at least {low}, got {value}")
    if args.seeds > 1 and (args.sacct or args.k8s_trace or args.sweep or args.compare_backfill):
        parser.error("--seeds applies to a single synthetic run only")
    if args.sacct and args.k8s_trace:
        parser.error("--sacct and --k8s-trace are alternative traces; pick one")
    if args.k8s_trace and not args.time_limit_model:
        parser.error(
            "--k8s-trace needs --time-limit-model {exact,padded,synthetic}: the k8s trace "
            "has no time limits, and which ones Slurm is given changes what backfill can do"
        )
    if args.time_limit_model and args.sacct:
        parser.error(
            "--time-limit-model does not apply to --sacct: the trace has its own limits "
            "(--partition-max-time sets the one for jobs without a finite limit)"
        )
    if args.partition_max_time is not None and not args.sacct:
        parser.error("--partition-max-time applies to --sacct only")
    if args.time_limit_model == "padded" and args.time_limit_factor is None:
        parser.error("--time-limit-model padded needs --time-limit-factor")
    if args.time_limit_factor is not None and args.time_limit_model != "padded":
        parser.error("--time-limit-factor applies to --time-limit-model padded only")
    if args.time_limit_factor is not None and not args.time_limit_factor >= 1:
        # For every trace, before anything runs: a limit below the runtime is
        # a TIMEOUT the simulator does not model, and it used to surface as
        # an error about one job instead of about this flag.
        parser.error(
            "--time-limit-factor must be >= 1: a limit below the runtime is a TIMEOUT, "
            "which is not modelled"
        )
    if args.json and (args.sweep or args.compare_backfill or args.seeds > 1):
        parser.error("--json applies to a single run only")
    if args.json and not Path(args.json).resolve().parent.is_dir():
        # Fail before a long run, not after it.
        parser.error(f"--json {args.json}: directory {Path(args.json).parent} does not exist")

    try:
        make_cluster, cluster_line = _cluster_factory(args)
        cfg = _config(args)
        work = _load_workload(args, args.grace_time or 0.0)
        _attach_tables(cfg, work, args)
        _trace_warnings(cfg, work, args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.compare_backfill:
        _, override = _backfill_on(cfg, args.backfill_mode)
        if override:
            cfg.warnings.append(override)

    jobs = work.jobs
    weights = cfg.priority
    if args.seeds > 1:
        source = f"synthetic (seeds {args.seed}-{args.seed + args.seeds - 1})"
    else:
        source = work.source
    print(f"\n{len(jobs)} jobs · {cluster_line}")
    print(f"trace: {source}")
    for note in work.notes:
        print(note)
    for line in _describe(cfg, args.backfill_mode, compare=args.compare_backfill):
        print(line)
    for warning in cfg.warnings:
        print(f"warning: {warning}")
    print()

    try:
        return _dispatch_runs(args, cfg, work, weights, make_cluster)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # Writing --json; every other input was read before the header.
        print(f"error: cannot write --json file: {exc}", file=sys.stderr)
        return 1


def _dispatch_runs(
    args: argparse.Namespace,
    cfg: SlurmConfig,
    work: Workload,
    weights: PriorityWeights,
    make_cluster: Any,
) -> int:
    jobs = work.jobs
    if args.sweep:
        print(f"Sweeping PriorityWeight{args.sweep.capitalize()}\n")
        for value in (0, 1_000, 10_000, 100_000):
            swept = replace(weights)
            setattr(swept, args.sweep, float(value))
            metrics = _run(
                jobs, args, cfg, swept, cfg.backfill_enabled, make_cluster, work.calendar_offset
            )
            print(
                f"  weight={value:<7}  "
                f"util {metrics.utilization * 100:5.1f}%  "
                f"mean wait {metrics.mean_wait / 60:7.1f} min  "
                f"p95 {metrics.p95_wait / 60:7.1f} min  "
                f"slowdown {metrics.mean_bounded_slowdown:5.2f}"
            )
        print()
        return 0

    if args.compare_backfill:
        on_cfg, _ = _backfill_on(cfg, args.backfill_mode)
        for label, leg, enabled in (("backfill OFF", cfg, False), ("backfill ON", on_cfg, True)):
            print(f"{label}")
            metrics = _run(jobs, args, leg, weights, enabled, make_cluster, work.calendar_offset)
            print(metrics.format())
            print()
        return 0

    if args.seeds > 1:
        _seed_spread(args, cfg, weights, make_cluster)
        return 0

    result, cluster = _simulate(
        jobs, args, cfg, weights, cfg.backfill_enabled, make_cluster, work.calendar_offset
    )
    metrics = compute(result, cluster)
    print(metrics.format())
    if result.unschedulable:
        print(f"  unschedulable on this cluster  {len(result.unschedulable)} job(s)")
    print()
    if args.json:
        name = args.config_name or ("S0" if args.k8s_trace else "slurm")
        doc = parity.to_json(
            parity.compute(result, cluster, config=name, trace_digest=work.digest),
            metrics,
            model=_json_model(args, cfg, work, weights, cluster, len(result.unschedulable)),
        )
        parity.write_json(doc, args.json)
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


def _json_model(
    args: argparse.Namespace,
    cfg: SlurmConfig,
    work: Workload,
    weights: PriorityWeights,
    cluster: Cluster,
    unschedulable: int,
) -> dict[str, Any]:
    """Every setting that changes the result, for `--json`'s `model` block.

    Two runs that differ in any input must not write the same provenance.
    It used to leave out the seed (the whole synthetic trace, and a k8s
    trace's synthetic limits), the job count, the cluster shape, the
    priority weights and every preemption setting but the type.
    """
    p = cfg.preemption
    grace = {f"qos {q.name}": q.grace_time for q in p.qos.values()} | {
        f"partition {s.name}": s.grace_time for s in p.partitions.values()
    }
    shape: dict[str, Any] = {
        "name": cluster.name,
        "nodes": len(cluster.nodes),
        "total_cpus": cluster.total_cpus,
        "total_gpus": cluster.total_gpus,
    }
    if not args.fleet:
        shape |= {"cpus_per_node": args.cpus, "gpus_per_node": args.gpus}
    return {
        **work.model,
        "backfill_mode": args.backfill_mode,
        "backfill": cfg.backfill_enabled,
        "fairshare_algorithm": weights.fairshare_algorithm,
        "priority": asdict(weights),
        "sched_params": cfg.scheduler.describe(),
        "preemption": str(p.preempt_type),
        "preemption_settings": {
            "mode": str(p.mode),
            "grace_time_seconds": grace,
            "exempt_time_seconds": p.exempt_time,
            "requeue_delay_seconds": p.requeue_delay,
            "checkpoint_fraction": p.checkpoint_fraction,
            "youngest_first": p.youngest_first,
            "reorder_count": p.reorder_count,
            "strict_order": p.strict_order,
            "job_requeue": p.job_requeue,
            "kill_wait_seconds": p.kill_wait,
            "message_timeout_seconds": p.message_timeout,
        },
        "fleet": cluster.name,
        "cluster": shape,
        "unschedulable_jobs": unschedulable,
    }


def _seed_spread(
    args: argparse.Namespace, cfg: SlurmConfig, weights: PriorityWeights, make_cluster: Any
) -> None:
    """One row per seed, then mean / min / max: how much is signal.

    With preemption on, two more columns: work lost and grace-locked
    CPU-hours (the report's `work lost` and `grace-locked` lines).
    """
    preempting = cfg.preemption.enabled
    rows: list[tuple[float, ...]] = []
    header = "  seed    util %   mean wait min   p95 wait min   slowdown"
    widths = [8, 14, 13, 9]
    formats = [".1f", ".1f", ".1f", ".2f"]
    if preempting:
        header += "   lost CPU-h   grace CPU-h"
        widths += [11, 12]
        formats += [".1f", ".1f"]
    print(header)

    def line(label: str, cols: tuple[float, ...] | list[float]) -> str:
        cells = "  ".join(f"{v:{w}{f}}" for v, w, f in zip(cols, widths, formats, strict=True))
        return f"  {label:>4}  {cells}"

    for seed in range(args.seed, args.seed + args.seeds):
        jobs = _synthetic_jobs(args, seed)
        if preempting:
            ladder = _synthetic_ladder(jobs, cfg.preemption.preempt_type, args.grace_time or 0.0)
            cfg.preemption = replace(cfg.preemption, qos=ladder.qos, partitions=ladder.partitions)
        m = _run(jobs, args, cfg, weights, cfg.backfill_enabled, make_cluster)
        row: tuple[float, ...] = (
            m.utilization * 100,
            m.mean_wait / 60,
            m.p95_wait / 60,
            m.mean_bounded_slowdown,
        )
        if preempting:
            row += (m.work_lost_cpu_hours, m.grace_locked_cpu_hours)
        rows.append(row)
        print(line(str(seed), row))
    for label, pick in (("mean", statistics.fmean), ("min", min), ("max", max)):
        print(line(label, [pick([r[i] for r in rows]) for i in range(len(widths))]))
    print()


if __name__ == "__main__":
    raise SystemExit(main())
