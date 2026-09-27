"""slurm.conf parsing: time strings, Priority*, SchedulerParameters, SchedulerType."""

from __future__ import annotations

import dataclasses
import math

import pytest

from schedlab.cli import main
from schedlab.config import SlurmConfig
from schedlab.params import SchedulerParameters
from schedlab.priority import PriorityWeights, _parse_duration
from schedlab.slurmconf import parse_minutes_str, parse_time_str, read_slurm_conf
from schedlab.trace import from_sacct

# ── Time strings: time_str2secs / time_str2mins ─────────────────────────────


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("30", 30 * 60),  # a bare number is MINUTES
        ("1:30", 90),  # minutes:seconds
        ("1:02:03", 3723),  # hours:minutes:seconds
        ("2-3", 2 * 86_400 + 3 * 3600),  # days-hours
        ("2-3:04", 2 * 86_400 + 3 * 3600 + 4 * 60),
        ("2-3:04:05", 2 * 86_400 + 3 * 3600 + 4 * 60 + 5),
        ("7-0", 7 * 86_400),
    ],
)
def test_time_strings_follow_slurm(text, seconds):
    assert parse_time_str(text) == seconds


@pytest.mark.parametrize("text", ["UNLIMITED", "INFINITE", "-1", "unlimited"])
def test_infinite_time_strings(text):
    assert parse_time_str(text) == math.inf


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "1-2-3", "1:", ":5", "1-2:3:4:5", "5:1-2"])
def test_invalid_time_strings_are_rejected(text):
    with pytest.raises(ValueError):
        parse_time_str(text)


def test_minute_settings_round_up_like_time_str2mins():
    assert parse_minutes_str("1:30") == 120
    assert parse_minutes_str("00:00:30") == 60
    assert parse_minutes_str("5") == 300


def test_priority_durations_read_bare_numbers_as_minutes():
    # 0.1.0 read "30" as thirty hours.
    assert _parse_duration("30") == 1800


def test_read_slurm_conf_lowercases_keys_and_keeps_full_values(tmp_path):
    conf = tmp_path / "slurm.conf"
    conf.write_text(
        "# comment\n"
        "  SchedulerParameters=bf_window=2880,bf_continue   # trailing\n"
        "\n"
        "PriorityType = priority/multifactor\n"
    )
    parsed = read_slurm_conf(str(conf))
    assert parsed["schedulerparameters"] == "bf_window=2880,bf_continue"
    assert parsed["prioritytype"] == "priority/multifactor"


# ── Priority settings ───────────────────────────────────────────────────────


def test_omitted_priority_keys_take_slurm_defaults(tmp_path):
    """0.1.0 filled gaps with the lab's demo weights; Slurm uses 0."""
    conf = tmp_path / "slurm.conf"
    conf.write_text("PriorityWeightFairshare=5000\n")
    w = PriorityWeights.from_slurm_conf(str(conf))
    assert w.fairshare == 5000
    assert (w.age, w.jobsize, w.partition, w.qos) == (0, 0, 0, 0)
    assert w.max_age == 7 * 86_400
    assert w.decay_half_life == 7 * 86_400
    assert w.calc_period == 300
    assert w.usage_reset_period == "NONE"
    assert w.fairshare_algorithm == "fair_tree"


def test_priority_calc_period_and_reset_period_parse(tmp_path):
    conf = tmp_path / "slurm.conf"
    conf.write_text("PriorityCalcPeriod=2\nPriorityUsageResetPeriod=weekly\n")
    w = PriorityWeights.from_slurm_conf(str(conf))
    assert w.calc_period == 120
    assert w.usage_reset_period == "WEEKLY"


@pytest.mark.parametrize("value", ["0", "00:00:00", "UNLIMITED"])
def test_calc_period_of_zero_is_refused(tmp_path, value):
    conf = tmp_path / "slurm.conf"
    conf.write_text(f"PriorityCalcPeriod={value}\n")
    with pytest.raises(ValueError, match="PriorityCalcPeriod"):
        PriorityWeights.from_slurm_conf(str(conf))


@pytest.mark.parametrize(("value", "seconds"), [("0:30", 60), ("0:01", 60), ("1:01", 120)])
def test_sub_minute_calc_periods_round_up_rather_than_fail(tmp_path, value, seconds):
    """time_str2mins() rounds up (parse_time.c L841-L847) and read_config.c
    rejects only a result below 1 (L4756-L4766): `0:30` runs as one minute."""
    conf = tmp_path / "slurm.conf"
    conf.write_text(f"PriorityCalcPeriod={value}\n")
    assert PriorityWeights.from_slurm_conf(str(conf)).calc_period == seconds


def test_bogus_reset_period_is_refused(tmp_path):
    conf = tmp_path / "slurm.conf"
    conf.write_text("PriorityUsageResetPeriod=FORTNIGHTLY\n")
    with pytest.raises(ValueError):
        PriorityWeights.from_slurm_conf(str(conf))


@pytest.mark.parametrize(
    ("flags", "algorithm"),
    [
        ("", "fair_tree"),
        ("NO_FAIR_TREE", "classic"),
        ("SMALL_RELATIVE_TO_TIME,DEPTH_OBLIVIOUS", "classic"),
    ],
)
def test_priority_flags_select_the_fairshare_algorithm(flags, algorithm):
    w, warnings = PriorityWeights.from_conf({"priorityflags": flags})
    assert w.fairshare_algorithm == algorithm
    if "SMALL_RELATIVE_TO_TIME" in flags:
        assert any("SMALL_RELATIVE_TO_TIME" in m for m in warnings)
        assert any("DEPTH_OBLIVIOUS" in m for m in warnings)


def test_priority_basic_is_fifo():
    w, _ = PriorityWeights.from_conf(
        {"prioritytype": "priority/basic", "priorityweightage": "1000"}
    )
    assert (w.age, w.fairshare, w.jobsize, w.partition, w.qos) == (0, 0, 0, 0, 0)


# ── SchedulerParameters ─────────────────────────────────────────────────────


def test_scheduler_parameter_defaults_match_slurm_conf_5():
    p = SchedulerParameters()
    assert p.bf_interval == 30
    assert p.bf_window == 1440 * 60
    assert p.bf_resolution == 60
    assert p.bf_max_job_test == 500  # sched_config.html's "100" is stale
    assert p.default_queue_depth == 100
    assert p.sched_interval == 60
    assert (p.bf_max_job_start, p.bf_max_job_user, p.bf_max_job_part) == (0, 0, 0)
    assert p.partition_job_depth == 0
    assert p.backfill_enabled and p.sched_interval_enabled


def test_scheduler_parameters_parse_values_flags_and_case():
    p = SchedulerParameters.parse(
        "bf_interval=60, BF_WINDOW=2880,bf_continue,bf_max_job_test=50,defer,"
        "bf_resolution=300,default_queue_depth=20,bf_max_job_user=4"
    )
    assert p.bf_interval == 60
    assert p.bf_window == 2880 * 60  # minutes in slurm.conf, seconds here
    assert p.bf_max_job_test == 50
    assert p.bf_resolution == 300
    assert p.default_queue_depth == 20
    assert p.bf_max_job_user == 4
    assert p.unmodelled == ("bf_continue", "defer")
    assert p.warnings == ()


@pytest.mark.parametrize(
    "item",
    [
        "bf_resolution=0",
        "bf_resolution=3601",
        "bf_window=43201",
        "bf_max_job_test=0",
        "bf_interval=0",
        "bf_max_job_start=10001",
        "bf_max_job_test=lots",
        "bf_window",
    ],
)
def test_invalid_scheduler_parameters_fall_back_to_default(item):
    p = SchedulerParameters.parse(item)
    key = item.split("=")[0]
    assert getattr(p, key) == getattr(SchedulerParameters(), key)
    assert len(p.warnings) == 1


def test_minus_one_disables_the_loops():
    p = SchedulerParameters.parse("bf_interval=-1,sched_interval=-1")
    assert not p.backfill_enabled
    assert not p.sched_interval_enabled
    # sched_interval=-1 is the whole main scheduler, not just its timer
    # (job_scheduler.c L1360-L1372).
    assert not p.main_scheduler_enabled


def test_zero_depth_and_zero_interval_are_valid_in_slurm():
    """job_scheduler.c rejects only negative default_queue_depth (L1273-L1284)
    and negative sched_interval other than -1 (L1373-L1377)."""
    p = SchedulerParameters.parse("default_queue_depth=0,sched_interval=0")
    assert p.warnings == ()
    assert p.default_queue_depth == 0
    assert p.sched_interval == 0
    assert p.main_scheduler_enabled and not p.sched_interval_enabled
    for bad in ("default_queue_depth=-1", "sched_interval=-2"):
        q = SchedulerParameters.parse(bad)
        key = bad.split("=")[0]
        assert getattr(q, key) == getattr(SchedulerParameters(), key)
        assert len(q.warnings) == 1


def test_bf_max_job_assoc_takes_precedence_over_bf_max_job_user():
    """backfill.c `_load_config()` L974-L979: "Both bf_max_job_user and
    bf_max_job_assoc are set: bf_max_job_assoc taking precedence"."""
    p = SchedulerParameters.parse("bf_max_job_user=1,bf_max_job_assoc=2")
    assert p.bf_max_job_assoc == 2
    assert p.bf_max_job_user == 0
    assert any("bf_max_job_assoc taking precedence" in w for w in p.warnings)
    alone = SchedulerParameters.parse("bf_max_job_user=1")
    assert alone.bf_max_job_user == 1 and alone.warnings == ()


def test_describe_names_every_modelled_value():
    text = SchedulerParameters().describe()
    for name in ("bf_interval=30s", "bf_window=1440min", "bf_max_job_test=500"):
        assert name in text


# ── The whole config ────────────────────────────────────────────────────────


def test_slurm_config_collects_everything_and_says_what_it_ignored(tmp_path):
    conf = tmp_path / "slurm.conf"
    conf.write_text(
        "SchedulerType=sched/builtin\n"
        "SchedulerParameters=bf_max_job_test=100,bf_continue\n"
        "PriorityFlags=NO_FAIR_TREE,MAX_TRES\n"
        "PriorityWeightJobSize=2000\n"
    )
    cfg = SlurmConfig.load(str(conf))
    assert cfg.scheduler_type == "sched/builtin"
    assert not cfg.backfill_enabled
    assert cfg.scheduler.bf_max_job_test == 100
    assert cfg.priority.fairshare_algorithm == "classic"
    assert cfg.priority.jobsize == 2000
    assert any("bf_continue" in w for w in cfg.warnings)
    assert any("MAX_TRES" in w for w in cfg.warnings)


def test_unknown_scheduler_type_falls_back_with_a_warning():
    cfg = SlurmConfig.from_conf({"schedulertype": "sched/magic"})
    assert cfg.scheduler_type == "sched/backfill"
    assert cfg.warnings


def test_sacct_user_column_is_optional(tmp_path):
    path = tmp_path / "sacct.txt"
    path.write_text(
        "JobID|User|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
        "1|alice|research|2026-07-26T09:00:00|00:10:00|01:00:00|1|8|cpu=8\n"
        "2||research|2026-07-26T09:01:00|00:10:00|01:00:00|1|8|cpu=8\n"
    )
    jobs = from_sacct(str(path))
    assert jobs[0].user == "alice" and jobs[0].owner == "alice"
    assert jobs[1].user is None and jobs[1].owner == "research"


def test_sacct_array_tasks_get_distinct_ids_and_the_trace_runs(tmp_path):
    """`sacct -X` lists each array task as its own row. Collapsing `5000_1`
    and `5000_2` onto 5000 gave two jobs one id, and `schedlab --sacct` hung."""
    path = tmp_path / "sacct.txt"
    path.write_text(
        "JobID|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
        "5000_1|research|2026-07-26T09:00:00|00:10:00|01:00:00|1|8|cpu=8\n"
        "5000_2|research|2026-07-26T09:00:00|00:10:00|01:00:00|1|8|cpu=8\n"
        "4999|trading|2026-07-26T09:01:00|00:05:00|00:30:00|1|8|cpu=8\n"
        "6000+0|trading|2026-07-26T09:02:00|00:05:00|00:30:00|1|8|cpu=8\n"
        "4999|trading|2026-07-26T09:03:00|00:05:00|00:30:00|1|8|cpu=8\n"
    )
    jobs = from_sacct(str(path))
    # A plain id keeps its number; everything else, and a repeat, gets a
    # fresh one above the largest plain id, in file order.
    assert [j.job_id for j in jobs] == [5000, 5001, 4999, 5002, 5003]
    assert main(["--sacct", str(path), "--nodes", "1", "--cpus", "8", "--gpus", "0"]) == 0


def test_scheduler_parameter_values_are_read_like_atoi():
    """backfill.c parses with atoi(): trailing junk and fractions are dropped."""
    p = SchedulerParameters.parse("bf_interval=45.9,bf_max_job_test=20jobs")
    assert p.bf_interval == 45
    assert p.bf_max_job_test == 20
    assert p.warnings == ()


# ── Preemption keys (PART B) ────────────────────────────────────────────────


def test_preemption_keys_are_read_from_slurm_conf():
    from schedlab.config import SlurmConfig

    cfg = SlurmConfig.from_conf(
        {
            "preempttype": "preempt/qos",
            "preemptmode": "REQUEUE,WITHIN",
            "preemptexempttime": "5",  # minutes, as time_str2secs reads it
            "preemptparameters": "youngest_first,reorder_count=3,send_user_signal",
            "jobrequeue": "0",
            "schedulerparameters": "requeue_delay=30",
            "killwait": "20",
        }
    )
    p = cfg.preemption
    assert p.enabled and p.preempt_type == "preempt/qos"
    assert (p.mode.base, p.mode.within) == ("REQUEUE", True)
    assert p.exempt_time == 300.0
    assert (p.youngest_first, p.reorder_count, p.strict_order) == (True, 3, False)
    assert p.job_requeue is False and p.requeue_delay == 30.0 and p.kill_wait == 20.0
    assert any("send_user_signal" in w for w in cfg.warnings)


def test_preemption_defaults_and_refusals():
    import pytest

    from schedlab.config import SlurmConfig

    assert not SlurmConfig.from_conf({}).preemption.enabled
    assert SlurmConfig.from_conf({}).preemption.requeue_delay == 120.0
    with pytest.raises(ValueError):
        SlurmConfig.from_conf({"preempttype": "preempt/qos"})  # PreemptMode OFF
    with pytest.raises(ValueError):
        SlurmConfig.from_conf({"preemptmode": "CANCEL"})  # no PreemptType
    with pytest.raises(ValueError, match="SUSPEND"):
        SlurmConfig.from_conf({"preempttype": "preempt/qos", "preemptmode": "SUSPEND,GANG"})
    with pytest.raises(ValueError, match="not modelled"):
        SlurmConfig.from_conf({"preempttype": "preempt/magic", "preemptmode": "CANCEL"})


def test_requeue_delay_is_a_modelled_scheduler_parameter():
    from schedlab.params import SchedulerParameters

    p = SchedulerParameters.parse("requeue_delay=45,bf_continue")
    assert p.requeue_delay == 45.0 and p.unmodelled == ("bf_continue",)
    assert "requeue_delay=45s" in p.describe()
    assert "requeue_delay" not in SchedulerParameters().describe()
    assert SchedulerParameters.parse("requeue_delay=-3").warnings


# ── Unmodelled keys that change scheduling are reported ─────────────────────


def test_scheduling_keys_that_are_not_modelled_are_reported():
    """They used to be dropped silently, although SlurmConfig promises a
    warning for every setting present and not modelled (slurm.conf(5) for
    what each one changes)."""
    cfg = SlurmConfig.from_conf(
        {
            "priorityflags": "NO_FAIR_TREE",
            "fairsharedampeningfactor": "5",
            "prioritysitefactorplugin": "site_factor/example",
            "priorityparameters": "x",
            "topologyplugin": "topology/tree",
            "selecttypeparameters": "CR_Core_Memory",
            "overtimelimit": "10",
            "partitionname": "gpu Nodes=ALL PriorityTier=10 PriorityJobFactor=5",
            "nodename": "n[0-3] CPUs=8",
        }
    )
    assert cfg.priority.fairshare_algorithm == "classic"
    text = "\n".join(cfg.warnings)
    for name in (
        "FairShareDampeningFactor=5",
        "PrioritySiteFactorPlugin=site_factor/example",
        "PriorityParameters=x",
        "TopologyPlugin=topology/tree",
        "SelectTypeParameters=CR_Core_Memory",
        "OverTimeLimit=10",
        "PartitionName lines are not parsed",
        "NodeName lines are not read",
    ):
        assert name in text, name
    # At their no-effect values these keys say nothing.
    quiet = SlurmConfig.from_conf(
        {"fairsharedampeningfactor": "1", "overtimelimit": "0", "topologyplugin": "Topology/Flat"}
    )
    assert quiet.warnings == []
    assert SlurmConfig.from_conf({"slurmctldhost": "ctl", "slurmctlddebug": "info"}).warnings == []


def test_a_partition_weight_without_a_partition_factor_is_reported():
    """PriorityJobFactor lives on PartitionName lines, which are not parsed,
    so from a slurm.conf every job's partition factor is 0."""
    cfg = SlurmConfig.from_conf({"priorityweightpartition": "100000"})
    assert cfg.priority.partition == 100_000
    assert any("PriorityWeightPartition=100000 has no effect" in w for w in cfg.warnings)
    assert not any("PriorityWeightPartition" in w for w in SlurmConfig.from_conf(
        {"priorityweightpartition": "0"}
    ).warnings)


def test_zero_half_life_needs_a_reset_period():
    """read_config.c (SchedMD/slurm@9f9da53 L4851-L4860) returns SLURM_ERROR
    for PriorityDecayHalfLife=0 with no PriorityUsageResetPeriod line, and
    slurm.conf(5): "If set to 0 PriorityUsageResetPeriod must be set"."""
    for zero in ("0", "0-0", "00:00:00"):
        with pytest.raises(ValueError, match="PriorityUsageResetPeriod"):
            SlurmConfig.from_conf({"prioritydecayhalflife": zero})
    # Any PriorityUsageResetPeriod line satisfies slurmctld, NONE included.
    for period in ("NONE", "WEEKLY"):
        w, _ = PriorityWeights.from_conf(
            {"prioritydecayhalflife": "0", "priorityusageresetperiod": period}
        )
        assert w.decay_half_life == 0.0
    assert PriorityWeights.from_conf({"prioritydecayhalflife": "1"})[0].decay_half_life == 60.0


def test_every_modelled_preempt_parameter_has_an_effect():
    """`_MODELLED_PREEMPT_PARAMS` alone decides which keys are reported; a key
    added there without a parser branch would be neither applied nor reported."""
    from schedlab.config import _MODELLED_PREEMPT_PARAMS, preemption_from_conf

    base = {"preempttype": "preempt/qos", "preemptmode": "REQUEUE"}
    default, _ = preemption_from_conf(base)
    for key in sorted(_MODELLED_PREEMPT_PARAMS):
        item = "reorder_count=5" if key == "reorder_count" else key
        cfg, warnings = preemption_from_conf({**base, "preemptparameters": item})
        assert warnings == [], key
        assert cfg != default, key
    _, warnings = preemption_from_conf({**base, "preemptparameters": "min_exempt_priority=5"})
    assert warnings == ["PreemptParameters min_exempt_priority is parsed but not modelled"]


# ── sacct time limits: never the true runtime ───────────────────────────────

_SACCT_HEADER = "JobID|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
_SACCT_ROWS = (
    "101|a|2026-07-29T23:50:00|00:20:00|UNLIMITED|1|1|cpu=1\n"
    "102|a|2026-07-29T23:55:00|00:10:00|Partition_Limit|1|1|cpu=1\n"
    "103|a|2026-07-29T23:56:00|01:00:31|01:00:00|1|1|cpu=1\n"
    "104|a|2026-07-29T23:57:00|00:30:00|02:00:00|1|1|cpu=1\n"
)


def _sacct(tmp_path, rows: str = _SACCT_ROWS, header: str = _SACCT_HEADER) -> str:
    path = tmp_path / "sacct.txt"
    path.write_text(header + rows)
    return str(path)


def test_sacct_limits_never_come_from_elapsed(tmp_path):
    """UNLIMITED and Partition_Limit used to get Elapsed, the true runtime, as
    their limit, and an overrun got its Elapsed too: a perfect estimate the
    scheduler must never have.

    * No finite limit: planned at 365 days, what backfill.c's
      `_set_backfill_timelimits()` plans with when MaxTime is UNLIMITED too.
    * Elapsed over the limit: the limit stays; the runtime is cut to it (the
      TIMEOUT kill; 31 s of kill latency here), and counted.
    """
    from schedlab.trace import UNLIMITED_PLANNING_LIMIT, load_sacct

    trace = load_sacct(_sacct(tmp_path))
    year = 365 * 86_400.0
    assert year == UNLIMITED_PLANNING_LIMIT
    assert [(j.duration, j.time_limit) for j in trace.jobs] == [
        (1200.0, year), (600.0, year), (3600.0, 3600.0), (1800.0, 7200.0),
    ]  # fmt: skip
    assert all(j.time_limit != j.duration for j in trace.jobs[:2])
    assert (trace.no_limit_jobs, trace.over_limit_jobs, trace.over_limit_seconds) == (2, 1, 31.0)
    assert "2 of 4 jobs have no finite Timelimit" in trace.notes[0]
    assert "planned at 365 days" in trace.notes[0]
    assert "cut to the limit, 31 s in all" in trace.notes[1]
    # A finite partition MaxTime plans them at that instead.
    short = load_sacct(_sacct(tmp_path), planning_limit=3600.0)
    assert [j.time_limit for j in short.jobs[:2]] == [3600.0, 3600.0]
    # A runtime longer than the planning limit contradicts it: refused.
    with pytest.raises(ValueError, match="longer than the planning limit"):
        load_sacct(_sacct(tmp_path), planning_limit=900.0)


def test_sacct_without_a_timelimit_column_is_refused(tmp_path):
    header = "JobID|Account|Submit|Elapsed|NNodes|ReqCPUS|ReqTRES\n"
    rows = "1|a|2026-07-29T09:00:00|00:10:00|1|1|cpu=1\n"
    with pytest.raises(ValueError, match="no Timelimit column"):
        from_sacct(_sacct(tmp_path, rows, header))


def test_sacct_places_t0_in_the_week_for_daily_and_weekly_resets(tmp_path):
    """2026-07-29 is a Wednesday: 3 days and 23 h 50 min after Sunday 00:00.

    With that offset a DAILY reset fires at the first tick past local
    midnight, t=600 (`_next_reset()`), and usage accrued before it is gone;
    at offset 0, as the CLI used to run, it would not fire until t=86400.
    """
    from schedlab.model import Cluster
    from schedlab.simulate import simulate
    from schedlab.trace import load_sacct

    trace = load_sacct(_sacct(tmp_path))
    assert trace.calendar_offset == 3 * 86_400 + 23 * 3600 + 50 * 60
    assert trace.origin == "2026-07-29T23:50:00"
    w = dataclasses.replace(PriorityWeights(), calc_period=300.0, usage_reset_period="DAILY")

    def usage_at(offset: float) -> dict[float, float]:
        work = load_sacct(_sacct(tmp_path)).jobs[:1]  # job 101 runs 0-1200
        result = simulate(
            work, Cluster.homogeneous(1, 1), weights=w, fairshare=w.make_fairshare(["a"]),
            calendar_offset=offset,
        )  # fmt: skip
        return {s.time: s.usage.get("a", 0.0) for s in result.fairshare_snapshots}

    local, flat = usage_at(trace.calendar_offset), usage_at(0.0)
    assert local[600.0] == pytest.approx(local[300.0])  # reset, then 300 s more
    assert flat[600.0] == pytest.approx(2 * flat[300.0], rel=1e-3)  # no reset


def test_cli_reports_the_sacct_limit_handling_and_the_calendar(tmp_path, capsys):
    conf = tmp_path / "slurm.conf"
    conf.write_text("PriorityUsageResetPeriod=DAILY\nPriorityWeightQOS=1000\n")
    path = _sacct(tmp_path)
    assert main(["--sacct", path, "--slurm-conf", str(conf), "--nodes", "1", "--cpus", "1"]) == 0
    out = capsys.readouterr().out
    assert "sacct: 2 of 4 jobs have no finite Timelimit" in out
    assert "PriorityUsageResetPeriod=DAILY: resets at midnight" in out
    assert "t=0 is 2026-07-29T23:50:00" in out
    assert "warning: PriorityWeightQOS=1000 has no effect on an sacct trace" in out
    argv = ["--sacct", path, "--partition-max-time", "1-0", "--nodes", "1", "--cpus", "1"]
    assert main(argv) == 0
    assert "planned at 1 day," in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--jobs", "5", "--partition-max-time", "60"])
    # A synthetic trace has no calendar: said, not silently assumed.
    assert main(["--jobs", "10", "--slurm-conf", str(conf)]) == 0
    out = capsys.readouterr().out
    assert "warning: PriorityUsageResetPeriod=DAILY: this trace has no calendar" in out
    assert "PriorityWeightQOS" not in out


def test_cli_hands_the_sacct_calendar_to_the_simulator(tmp_path, monkeypatch):
    """The CLI never passed `calendar_offset`, so DAILY and WEEKLY resets
    counted from the trace's first submit instead of local midnight."""
    from schedlab.simulate import simulate as real

    seen: list[float] = []

    def spy(*args, **kwargs):
        seen.append(kwargs["calendar_offset"])
        return real(*args, **kwargs)

    monkeypatch.setattr("schedlab.cli.simulate", spy)
    path = _sacct(tmp_path)
    assert main(["--sacct", path, "--nodes", "1", "--cpus", "1"]) == 0
    assert main(["--sacct", path, "--nodes", "1", "--cpus", "1", "--compare-backfill"]) == 0
    assert main(["--sacct", path, "--nodes", "1", "--cpus", "1", "--sweep", "age"]) == 0
    assert seen == [3 * 86_400 + 23 * 3600 + 50 * 60] * 7
    seen.clear()
    assert main(["--jobs", "5"]) == 0
    assert seen == [0.0]


def test_a_replacement_scheduler_string_replaces_its_warnings_and_requeue_delay():
    """`from_conf(sched_params=...)` reads the new string instead of the
    config's, so nothing derived from the discarded one survives."""
    parsed = {
        "schedulerparameters": "bf_continue,bf_window=99999,requeue_delay=5",
        "preempttype": "preempt/qos",
        "preemptmode": "REQUEUE",
    }
    own = SlurmConfig.from_conf(parsed)
    assert own.preemption.requeue_delay == 5.0
    assert any("bf_window=99999" in w for w in own.warnings)
    replaced = SlurmConfig.from_conf(parsed, sched_params="bf_window=120")
    assert replaced.scheduler.bf_window == 7200.0
    assert replaced.preemption.requeue_delay == 120.0
    assert not any("bf_window=99999" in w or "bf_continue" in w for w in replaced.warnings)
