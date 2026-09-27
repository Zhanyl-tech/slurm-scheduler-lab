"""The command line: headers, warnings, and the three run shapes."""

from __future__ import annotations

from schedlab.cli import main


def test_single_run_prints_the_model_in_force(capsys):
    assert main(["--jobs", "40"]) == 0
    out = capsys.readouterr().out
    assert "model: conservative backfill · fairshare fair_tree · PriorityCalcPeriod 5 min" in out
    assert "SchedulerParameters: bf_interval=30s" in out
    assert "backfill cycles" in out


def test_easy_mode_has_no_scheduler_parameters_line(capsys):
    assert main(["--jobs", "40", "--backfill-mode", "easy", "--calc-period", "0"]) == 0
    out = capsys.readouterr().out
    assert "model: easy backfill" in out
    assert "every event (idealised)" in out
    assert "SchedulerParameters:" not in out
    assert "backfill cycles" not in out


def test_compare_backfill_reports_cycles_only_when_backfill_runs(capsys):
    assert main(["--jobs", "40", "--compare-backfill"]) == 0
    out = capsys.readouterr().out
    off, on = out.split("backfill ON")
    assert "backfill cycles" not in off
    assert "backfill cycles" in on


def test_sweep_runs_every_weight(capsys):
    assert main(["--jobs", "30", "--sweep", "age", "--fairshare-algorithm", "classic"]) == 0
    out = capsys.readouterr().out
    assert out.count("weight=") == 4
    assert "fairshare classic" in out


def test_unmodelled_settings_are_announced_not_ignored(tmp_path, capsys):
    conf = tmp_path / "slurm.conf"
    conf.write_text(
        "PriorityUsageResetPeriod=MONTHLY\n"
        "SchedulerParameters=bf_continue,bf_max_job_test=20\n"
        "PriorityWeightJobSize=1000\n"
    )
    assert main(["--jobs", "30", "--slurm-conf", str(conf)]) == 0
    out = capsys.readouterr().out
    assert "warning: PriorityUsageResetPeriod=MONTHLY" in out
    assert "warning: SchedulerParameters not modelled: bf_continue" in out
    assert "bf_max_job_test=20" in out


def test_sched_params_flag_overrides_and_warns(capsys):
    assert main(["--jobs", "30", "--sched-params", "bf_max_job_test=5,bf_yield_sleep=1"]) == 0
    out = capsys.readouterr().out
    assert "bf_max_job_test=5" in out
    assert "not modelled: bf_yield_sleep" in out


def test_bad_inputs_fail_cleanly(tmp_path, capsys):
    assert main(["--sacct", str(tmp_path / "missing.txt")]) == 1
    conf = tmp_path / "slurm.conf"
    conf.write_text("PriorityCalcPeriod=0\n")
    assert main(["--jobs", "10", "--slurm-conf", str(conf)]) == 1
    assert "PriorityCalcPeriod" in capsys.readouterr().err


def test_seed_spread_prints_one_row_per_seed_and_a_summary(capsys):
    assert main(["--jobs", "30", "--seed", "3", "--seeds", "3"]) == 0
    out = capsys.readouterr().out
    assert "trace: synthetic (seeds 3-5)" in out
    for label in ("     3", "     4", "     5", "mean", " min", " max"):
        assert label in out


def test_seed_spread_refuses_combinations_it_cannot_honour(capsys):
    import pytest

    with pytest.raises(SystemExit):
        main(["--seeds", "3", "--compare-backfill"])
    with pytest.raises(SystemExit):
        main(["--seeds", "0"])


# ── PART B: fleets, S0, preemption, --json ──────────────────────────────────

_FLEET = (
    "name: tiny\ntopology:\n  racksPerSwitch: 1\nnodeClasses:\n"
    "  - name: big\n    count: 2\n    gpus: 8\n    cpus: 16\n    nodesPerRack: 1\n"
    "  - name: small\n    count: 2\n    gpus: 2\n    cpus: 8\n"
)
_TRACE = (
    "job_id,account,submit_time,duration,gpus,gang_size,priority\n"
    "1,a,5.0,600.0,8,1,0\n2,b,6.0,300.0,2,2,500\n3,a,7.0,100.0,1,1,0\n"
    "4,b,8.0,200.0,4,2,100\n"
)


def _s0_files(tmp_path) -> tuple[str, str]:
    fleet = tmp_path / "fleet.yaml"
    fleet.write_text(_FLEET)
    trace = tmp_path / "trace.csv"
    trace.write_text(_TRACE)
    return str(fleet), str(trace)


def test_k8s_trace_requires_an_explicit_time_limit_model(tmp_path, capsys):
    import pytest

    fleet, trace = _s0_files(tmp_path)
    with pytest.raises(SystemExit):
        main(["--k8s-trace", trace, "--fleet", fleet])
    assert "--time-limit-model" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--k8s-trace", trace, "--time-limit-model", "padded"])
    with pytest.raises(SystemExit):
        main(["--sacct", trace, "--time-limit-model", "exact"])
    with pytest.raises(SystemExit):
        main(["--k8s-trace", trace, "--sacct", trace, "--time-limit-model", "exact"])


def test_s0_run_prints_the_fleet_and_mapping_and_writes_json(tmp_path, capsys):
    import json

    fleet, trace = _s0_files(tmp_path)
    out = tmp_path / "s0.json"
    argv = ["--k8s-trace", trace, "--fleet", fleet, "--time-limit-model", "padded",
            "--time-limit-factor", "2", "--json", str(out)]  # fmt: skip
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "fleet tiny: 4 nodes, 20 GPUs" in text
    assert "time limits padded ×2 · priority→qos · 1 CPU per pod" in text
    assert "gpu utilization" in text and "frag C gpu node/rack/sw" in text
    doc = json.loads(out.read_text())
    assert doc["config"] == "S0" and doc["fleet"] == "tiny" and doc["jobs"] == 4
    assert doc["pods"] == 6 and doc["topology_declared"] is True
    assert len(doc["trace_digest"]) == 12
    assert doc["gang_stranded_gpu_hours"] == 0.0 and doc["gang_assembled"] == 2
    assert doc["model"]["time_limit_model"] == "padded"
    assert doc["model"]["time_limit_factor"] == 2.0


def test_tier_mapping_is_announced(tmp_path, capsys):
    fleet, trace = _s0_files(tmp_path)
    argv = ["--k8s-trace", trace, "--fleet", fleet, "--time-limit-model", "exact",
            "--k8s-priority", "tier"]  # fmt: skip
    assert main(argv) == 0
    assert "partitions (PriorityTier): p0=1, p100=2, p500=3" in capsys.readouterr().out


def test_preemption_flags(tmp_path, capsys):
    fleet, trace = _s0_files(tmp_path)
    base = ["--k8s-trace", trace, "--fleet", fleet, "--time-limit-model", "exact"]
    assert main([*base, "--preempt-type", "qos", "--preempt-mode", "REQUEUE",
                 "--grace-time", "30"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "preemption: preempt/qos · PreemptMode=REQUEUE · GraceTime 30 s" in out
    assert "preemptions" in out and "grace-locked" in out
    # A plugin with PreemptMode OFF is a config slurmctld refuses.
    assert main([*base, "--preempt-type", "qos"]) == 1
    assert "PreemptMode=OFF" in capsys.readouterr().err
    # partition_prio needs the tier mapping on a k8s trace.
    assert main([*base, "--preempt-type", "partition_prio", "--preempt-mode", "CANCEL"]) == 1
    assert "--k8s-priority tier" in capsys.readouterr().err
    assert main(["--jobs", "20", "--backfill-mode", "easy", "--preempt-type", "qos",
                 "--preempt-mode", "CANCEL"]) == 1  # fmt: skip
    assert "conservative" in capsys.readouterr().err


def test_synthetic_preemption_and_ignored_flags(capsys):
    assert main(["--jobs", "60", "--preempt-type", "partition_prio", "--preempt-mode",
                 "CANCEL", "--checkpoint-fraction", "0.2"]) == 0  # fmt: skip
    assert "preemption: preempt/partition_prio" in capsys.readouterr().out
    assert main(["--jobs", "20", "--grace-time", "30"]) == 0
    assert "warning: --grace-time ignored: preemption is off" in capsys.readouterr().out


def test_json_refuses_multi_run_shapes(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        main(["--jobs", "10", "--compare-backfill", "--json", str(tmp_path / "x.json")])


def test_bad_fleet_fails_cleanly(tmp_path, capsys):
    bad = tmp_path / "f.yaml"
    bad.write_text("nodeClasses:\n  - name: a\n    count: 1\n")
    assert main(["--fleet", str(bad), "--jobs", "5"]) == 1
    assert "missing" in capsys.readouterr().err


def test_config_preemption_with_sacct_is_announced_and_not_applied(tmp_path, capsys):
    conf = tmp_path / "slurm.conf"
    conf.write_text("PreemptType=preempt/qos\nPreemptMode=REQUEUE\n")
    sacct = tmp_path / "sacct.txt"
    sacct.write_text(
        "JobID|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
        "101|research|2026-07-26T09:00:00|00:10:00|01:00:00|1|8|cpu=8\n"
    )
    assert main(["--slurm-conf", str(conf), "--sacct", str(sacct)]) == 0
    out = capsys.readouterr().out
    assert "warning: preempt/qos in the config is not applied" in out
    assert "preemptions" not in out
    assert main(["--sacct", str(sacct), "--preempt-type", "qos", "--preempt-mode", "CANCEL"]) == 1


def _line_value(out: str, label: str) -> str:
    """The value column of the report line starting with `label`."""
    for line in out.splitlines():
        if line.strip().startswith(label):
            return line.strip()[len(label) :].split()[0]
    raise AssertionError(f"no {label!r} line in:\n{out}")


def test_preempt_exempt_time_minus_one_and_unlimited_mean_none(capsys):
    """slurm.conf(5): "A time of -1 disables the option, equivalent to 0".
    `-1`, `INFINITE` and `UNLIMITED` all parse to INFINITE, which
    `acct_policy_get_preemptable_time()` treats as none. They used to make
    every job exempt forever on the command line."""
    base = ["--jobs", "150", "--seed", "5", "--preempt-type", "qos", "--preempt-mode", "REQUEUE"]
    counts = {}
    for value in ("0", "-1", "UNLIMITED", "INFINITE", "10"):
        assert main([*base, f"--preempt-exempt-time={value}"]) == 0
        counts[value] = int(_line_value(capsys.readouterr().out, "preemptions"))
    assert counts["0"] > 0
    assert counts["-1"] == counts["UNLIMITED"] == counts["INFINITE"] == counts["0"]
    assert counts["10"] != counts["0"]  # a real exempt time still does something


def test_sweep_honours_sched_builtin_from_the_config(tmp_path, capsys):
    """The sweep used to run backfill whatever SchedulerType said."""
    conf = tmp_path / "slurm.conf"
    conf.write_text("SchedulerType=sched/builtin\nPriorityWeightAge=1000\n")
    argv = ["--jobs", "80", "--seed", "5", "--slurm-conf", str(conf)]
    assert main(argv) == 0
    single = capsys.readouterr().out
    assert "model: sched/builtin (no backfill)" in single
    assert _line_value(single, "backfilled jobs") == "0"
    assert main([*argv, "--sweep", "jobsize"]) == 0
    sweep = capsys.readouterr().out
    assert "model: sched/builtin (no backfill)" in sweep
    row = next(line for line in sweep.splitlines() if "weight=0 " in line)
    # The weight=0 row is the config's own weights: the same run as above.
    assert f"util {float(_line_value(single, 'cpu utilization')):5.1f}%" in row
    assert f"mean wait {float(_line_value(single, 'mean wait')):7.1f} min" in row


def test_bf_interval_minus_one_is_named_in_the_header(capsys):
    assert main(["--jobs", "20", "--sched-params", "bf_interval=-1"]) == 0
    assert "backfill disabled (bf_interval=-1)" in capsys.readouterr().out
    # EASY honours it too (no reservation jumping), and the header now says
    # so instead of "easy backfill".
    argv = ["--jobs", "20", "--backfill-mode", "easy", "--sched-params", "bf_interval=-1"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "model: easy model, backfill disabled (bf_interval=-1)" in out
    assert _line_value(out, "backfilled jobs") == "0"


def test_fractional_calc_period_rounds_up_to_whole_minutes(capsys):
    """time_str2mins() rounds up; a Slurm period is always whole minutes."""
    assert main(["--jobs", "20", "--calc-period", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "PriorityCalcPeriod 1 min" in out
    assert "warning: --calc-period 0.5: PriorityCalcPeriod is whole minutes" in out
    assert main(["--jobs", "20", "--calc-period", "2"]) == 0
    out = capsys.readouterr().out
    assert "PriorityCalcPeriod 2 min" in out and "whole minutes" not in out


def test_time_limit_models_apply_to_synthetic_traces(capsys):
    """Same runtimes, different requests: only `time_limit` changes.

    Every limit is rounded up to a whole minute, as Slurm stores it, so
    "exact" uses a little under 100% of its request and ×2 under 50%."""
    assert main(["--jobs", "30", "--time-limit-model", "exact"]) == 0
    out = capsys.readouterr().out
    assert "time limits: exact (limit = runtime, rounded up to a whole minute)" in out
    assert 95.0 < float(_line_value(out, "time-limit accuracy")) < 100.0
    assert main(["--jobs", "30", "--time-limit-model", "padded", "--time-limit-factor", "2"]) == 0
    out = capsys.readouterr().out
    assert "time limits: padded ×2" in out
    assert 48.0 < float(_line_value(out, "time-limit accuracy")) < 50.0
    # `synthetic` is the generator's own padding: the same run as no flag.
    assert main(["--jobs", "30", "--time-limit-model", "synthetic"]) == 0
    modelled = capsys.readouterr().out
    assert main(["--jobs", "30"]) == 0
    plain = capsys.readouterr().out
    assert modelled.split("\n\n", 1)[1] == plain.split("\n\n", 1)[1]
    assert main(["--jobs", "30", "--seeds", "2", "--time-limit-model", "exact"]) == 0


def test_seed_spread_shows_preemption_cost_columns(capsys):
    argv = ["--jobs", "60", "--seeds", "2", "--preempt-type", "qos", "--preempt-mode", "REQUEUE"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "lost CPU-h   grace CPU-h" in out
    row = next(line for line in out.splitlines() if line.strip().startswith("mean"))
    assert len(row.split()) == 7  # label + four metrics + two preemption columns
    assert main(["--jobs", "60", "--seeds", "2"]) == 0
    assert "lost CPU-h" not in capsys.readouterr().out


def test_backfilled_jobs_are_distinct_and_backfill_starts_are_reported_beside_them(capsys):
    """A requeued job that backfill starts again is one backfilled job and two
    backfill starts. The report counted starts under "backfilled jobs", so
    with preemption on it read as more jobs backfilled when fewer were."""
    argv = ["--jobs", "100", "--seed", "5", "--preempt-type", "qos", "--preempt-mode",
            "REQUEUE", "--grace-time", "300"]  # fmt: skip
    assert main(argv) == 0
    line = next(
        line for line in capsys.readouterr().out.splitlines() if "backfilled jobs" in line
    )
    distinct = int(line.split()[2])
    starts = int(line.split("(")[1].split()[0])
    assert starts > distinct
    assert main(["--jobs", "100", "--seed", "5"]) == 0
    assert "backfill starts" not in capsys.readouterr().out  # equal without requeues


def test_time_limit_factor_below_one_is_refused_for_every_trace(tmp_path, capsys):
    """It was checked for k8s traces only; a synthetic run reached the
    simulator and failed on one job's limit instead of on the flag."""
    import pytest

    fleet, trace = _s0_files(tmp_path)
    for argv in (
        ["--jobs", "20"],
        ["--jobs", "20", "--seeds", "2"],
        ["--k8s-trace", trace, "--fleet", fleet],
    ):
        with pytest.raises(SystemExit):
            main([*argv, "--time-limit-model", "padded", "--time-limit-factor", "0.5"])
        assert "--time-limit-factor must be >= 1" in capsys.readouterr().err
    assert main(["--jobs", "20", "--time-limit-model", "padded", "--time-limit-factor", "1"]) == 0


def test_json_write_failures_are_clean_errors(tmp_path, capsys):
    import pytest

    with pytest.raises(SystemExit):  # before the run, not after it
        main(["--jobs", "10", "--json", str(tmp_path / "missing" / "x.json")])
    assert "does not exist" in capsys.readouterr().err
    # A directory in the way: caught when writing, reported without a traceback.
    blocked = tmp_path / "out.json"
    blocked.mkdir()
    assert main(["--jobs", "10", "--json", str(blocked)]) == 1
    assert "error: cannot write --json file" in capsys.readouterr().err


def test_compare_backfill_overrides_either_way_of_disabling_backfill(tmp_path, capsys):
    """One rule for both: the ON leg backfills, and the header says what it
    overrode. `sched/builtin` used to be overridden silently under a header
    naming it, and `bf_interval=-1` honoured, so the ON leg repeated OFF."""
    conf = tmp_path / "slurm.conf"
    conf.write_text("SchedulerType=sched/builtin\n")
    for extra in (["--slurm-conf", str(conf)], ["--sched-params", "bf_interval=-1"]):
        assert main(["--jobs", "80", "--seed", "5", "--compare-backfill", *extra]) == 0
        out = capsys.readouterr().out
        assert "model: conservative model, backfill OFF vs ON" in out
        assert "warning: --compare-backfill: the config disables backfill" in out
        off, on = out.split("backfill ON\n")
        assert _line_value(off, "backfilled jobs") == "0"
        assert int(_line_value(on, "backfilled jobs")) > 0
    # With backfill enabled nothing is overridden and the header is unchanged.
    assert main(["--jobs", "40", "--compare-backfill"]) == 0
    out = capsys.readouterr().out
    assert "model: conservative backfill" in out and "--compare-backfill:" not in out


# ── Review round 4: inputs, provenance and overrides ────────────────────────


def test_cluster_shape_and_job_count_are_range_checked(capsys):
    """The fleet loader refuses these values; `--gpus -2` used to run and
    print a report of a cluster with -32 GPUs."""
    import pytest

    for argv, message in (
        (["--gpus", "-2"], "--gpus must be at least 0"),
        (["--nodes", "0"], "--nodes must be at least 1"),
        (["--cpus", "0"], "--cpus must be at least 1"),
        (["--jobs", "-3"], "--jobs must be at least 1"),
    ):
        with pytest.raises(SystemExit):
            main([*argv, "--jobs", "5"] if argv[0] != "--jobs" else argv)
        assert message in capsys.readouterr().err
    assert main(["--jobs", "5", "--gpus", "0"]) == 0


def test_preempt_mode_with_preempt_type_none_is_announced(capsys):
    """`--preempt-type none` discards the mode; it used to do so silently."""
    argv = ["--jobs", "5", "--preempt-type", "none", "--preempt-mode", "REQUEUE,PRIORITY"]
    assert main(argv) == 0
    assert "warning: --preempt-mode ignored: preemption is off" in capsys.readouterr().out


def test_sched_params_replace_the_configs_string_whole(tmp_path, capsys):
    """Warnings about the config's own SchedulerParameters, and its
    requeue_delay, used to survive `--sched-params`."""
    conf = tmp_path / "slurm.conf"
    conf.write_text("SchedulerParameters=bf_continue,bf_window=99999,requeue_delay=5\n")
    base = ["--jobs", "30", "--slurm-conf", str(conf), "--preempt-type", "qos",
            "--preempt-mode", "REQUEUE"]  # fmt: skip
    assert main(base) == 0
    out = capsys.readouterr().out
    assert "requeue_delay 5 s" in out and "bf_window=99999 is out of range" in out
    assert main([*base, "--sched-params", "bf_window=120"]) == 0
    out = capsys.readouterr().out
    assert "bf_window=120min" in out
    assert "requeue_delay 120 s" in out  # Slurm's default: the new string sets none
    assert "bf_window=99999" not in out and "bf_continue" not in out
    assert main([*base, "--sched-params", "requeue_delay=7,bf_yield_sleep=1"]) == 0
    out = capsys.readouterr().out
    assert "requeue_delay 7 s" in out and "not modelled: bf_yield_sleep" in out


def test_json_model_records_every_setting_that_changes_the_result(tmp_path):
    """Two runs that differ only in --seed wrote identical `model` blocks."""
    import json

    docs = []
    for seed in ("0", "1"):
        out = tmp_path / f"s{seed}.json"
        assert main(["--jobs", "20", "--seed", seed, "--json", str(out)]) == 0
        docs.append(json.loads(out.read_text()))
    a, b = (d["model"] for d in docs)
    assert a != b and (a["seed"], b["seed"]) == (0, 1) and a["jobs"] == 20
    assert a["cluster"] == {
        "name": "homogeneous", "nodes": 16, "total_cpus": 128, "total_gpus": 32,
        "cpus_per_node": 8, "gpus_per_node": 2,
    }  # fmt: skip
    assert a["priority"]["fairshare"] == 10_000.0 and a["priority"]["calc_period"] == 300.0
    out = tmp_path / "p.json"
    argv = ["--jobs", "20", "--preempt-type", "qos", "--preempt-mode", "CANCEL",
            "--grace-time", "30", "--checkpoint-fraction", "0.25", "--json", str(out)]  # fmt: skip
    assert main(argv) == 0
    settings = json.loads(out.read_text())["model"]["preemption_settings"]
    assert settings["mode"] == "CANCEL" and settings["checkpoint_fraction"] == 0.25
    assert set(settings["grace_time_seconds"].values()) == {30.0}
    assert settings["requeue_delay_seconds"] == 120.0


def test_json_model_records_the_seed_of_synthetic_k8s_limits(tmp_path):
    import json

    fleet, trace = _s0_files(tmp_path)
    models = []
    for seed in ("0", "1"):
        out = tmp_path / f"k{seed}.json"
        argv = ["--k8s-trace", trace, "--fleet", fleet, "--time-limit-model", "synthetic",
                "--seed", seed, "--json", str(out)]  # fmt: skip
        assert main(argv) == 0
        models.append(json.loads(out.read_text())["model"])
    assert [m["seed"] for m in models] == [0, 1]
    out = tmp_path / "exact.json"
    argv = ["--k8s-trace", trace, "--fleet", fleet, "--time-limit-model", "exact",
            "--json", str(out)]  # fmt: skip
    assert main(argv) == 0
    exact = json.loads(out.read_text())["model"]
    assert exact["seed"] is None  # the seed changes nothing there
    assert "cpus_per_node" not in exact["cluster"] and exact["cluster"]["total_gpus"] == 20


def test_truncated_trace_rows_fail_cleanly(tmp_path, capsys):
    """A short row raised TypeError / AttributeError out of main()."""
    fleet, _ = _s0_files(tmp_path)
    short = tmp_path / "short.csv"
    short.write_text("job_id,account,submit_time,duration,gpus,gang_size,priority\n"
                     "1,a,5.0,600.0,8,1,0\n2,b,6.0,300.0,2,2\n")  # fmt: skip
    assert main(["--k8s-trace", str(short), "--fleet", fleet, "--time-limit-model", "exact"]) == 1
    assert "line 3: fewer fields than the header" in capsys.readouterr().err
    sacct = tmp_path / "sacct.txt"
    sacct.write_text(
        "JobID|User|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
        "101|u|a|2026-07-26T09:32:04\n"
    )
    assert main(["--sacct", str(sacct)]) == 1
    assert "line 2: fewer fields than the header" in capsys.readouterr().err
    sacct.write_text(
        "JobID|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
        "101|a|2026-07-26T09:32:04|00:10:00|01:00:00|1|8|cpu=8|extra\n"
    )
    assert main(["--sacct", str(sacct)]) == 1
    assert "line 2: more fields than the header" in capsys.readouterr().err


def test_bf_max_job_start_stops_are_reported(capsys):
    """They were counted but never printed, so a run where every cycle
    stopped early read `cycles at bf_max_job_test 0`, as if the whole queue
    had been considered."""
    assert main(["--jobs", "100", "--seed", "5", "--sched-params", "bf_max_job_start=1"]) == 0
    out = capsys.readouterr().out
    assert int(_line_value(out, "cycles at bf_max_job_start")) > 0
    assert main(["--jobs", "100", "--seed", "5"]) == 0
    assert "cycles at bf_max_job_start" not in capsys.readouterr().out  # 0: not printed


def test_partition_max_time_is_whole_minutes_like_slurm_conf_maxtime(tmp_path):
    """MaxTime is read with time_str2mins(): `0:30` (30 s) is one minute."""
    import json

    sacct = tmp_path / "sacct.txt"
    sacct.write_text(
        "JobID|Account|Submit|Elapsed|Timelimit|NNodes|ReqCPUS|ReqTRES\n"
        "101|a|2026-07-26T09:00:00|00:00:20|UNLIMITED|1|8|cpu=8\n"
    )
    out = tmp_path / "m.json"
    argv = ["--sacct", str(sacct), "--partition-max-time", "0:30", "--json", str(out)]
    assert main(argv) == 0
    assert json.loads(out.read_text())["model"]["planning_limit_seconds"] == 60.0


def test_a_gang_flag_on_the_command_line_is_announced(capsys):
    """A slurm.conf GANG was reported; the same flag on --preempt-mode was not."""
    assert main(["--jobs", "10", "--preempt-type", "qos", "--preempt-mode", "REQUEUE,GANG"]) == 0
    assert "warning: PreemptMode=GANG (time-slicing) is not modelled" in capsys.readouterr().out
