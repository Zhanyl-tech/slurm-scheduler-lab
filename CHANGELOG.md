# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Fidelity release. Every entry below is relative to 0.1.0. The simulator now
accrues fairshare usage the way Slurm does, plans backfill the way
`sched/backfill` does, and uses Slurm's default fairshare algorithm and job
size formula. Upstream behaviour was checked against slurm.conf(5) and the
scheduling, multifactor and Fair Tree docs for Slurm 26.05, and against source
at SchedMD/slurm@9f9da53b4a7bc5b56062bc357f4f94e0d19c71cc. Citations are in
the code. Every number in the README was regenerated from the commands it
shows or the scripts it names (`scripts/`); none come from a live cluster.

A note on the plan this work started from: it assumed a wall-clock harness
with a `--speedup` factor, like the sibling k8s lab. This simulator has none.
It is event-driven, so `bf_window`, `bf_interval`, `PriorityDecayHalfLife`
and `PriorityCalcPeriod` were already in simulated seconds, and no time
scaling was added. The README section "No time scaling, and why" explains.

It also adds preemption, heterogeneous fleets, and the S0 cross-substrate
control for k8s-gpu-scheduler-lab. Preemption is off by default, as in Slurm
(`PreemptType=preempt/none`), and none of its code runs then. Its semantics
were checked against slurm.conf(5), sacctmgr(1) and preempt.html, and against
source at the same commit: src/interfaces/preempt.c, the partition_prio and
qos plugins, cons_tres `_run_now()`, node_scheduler.c, and job_mgr.c
`batch_requeue_fini()`.

### Fixed

- **Invented time limits are whole minutes.** The synthetic generator and
  every S0 time-limit model handed the scheduler fractional seconds, a limit
  no Slurm job can have: `sbatch --time` goes through `time_str2mins()`,
  which rounds seconds up, and backfill plans with `time_limit * 60`
  (sbatch(1): "Time resolution is one minute and second values are rounded
  up to the next minute"). Every invented limit, `exact` included, is now
  rounded up to a whole minute (`trace.whole_minutes`), and
  `--partition-max-time` is read in whole minutes like slurm.conf's
  `MaxTime`. Every synthetic and S0 number in the README was regenerated.
  Most conclusions held; some moved. The seed-5 headline went from 89.9% /
  442.5 min to 83.0% / 396.1 min with backfill on, and seed 5 now favours
  EASY over the conservative mode by 2.9 points of utilization where it
  favoured conservative by 6.0. Over ten seeds, exact limits raise
  utilization on 7 of 10 seeds, not 9. Preemption lowers it on 6 of 10, not
  8, so the README no longer says preemption costs throughput. S0 with exact
  limits went from 75.5% / 159.2 min to 74.8% / 155.8 min, and ×3 padding
  now has the higher GPU utilization. Exact limits still gave the highest S0
  mean wait and the lowest starvation ratio on 5 of 5 seeds.
- **`--json` wait under preemption is the k8s lab's.** `parity.compute`
  took a requeued job's wait from submit to its last start, so the time its
  earlier runs ran counted as waiting, and four places in the docs called
  that the k8s lab's rule. The k8s lab counts pending time only (its
  docs/metrics.md, "Under eviction: pending time only"). `parity.pending_wait`
  ports `_pending_seconds`, with the k8s `evict_time` read as Slurm's
  selection, and every wait-derived key follows it. On five S0
  default-profile traces with preemption on, `mean_wait` came out 1.3–2.9
  min lower than under the old rule. Without preemption nothing changes. The
  text report keeps submit to last start, and the docs now say the two
  differ.
- **PreemptMode values slurmctld refuses are refused** (`preempt_mode_num()`,
  read_config.c). That covers two base modes, `GANG` with `WITHIN` or
  `PRIORITY`, and `WITHIN` or `PRIORITY` under `preempt/none`. Before, two
  base modes such as `CANCEL,REQUEUE` ran as whichever came last, and
  `WITHIN` or `PRIORITY` under `preempt/none` loaded as preemption off.
  `CLUSTER` now reads as OFF, and `ON` as SUSPEND, which is refused. A `GANG`
  flag on `--preempt-mode` is reported as not modelled, as a slurm.conf one
  already was.
- **Per-QOS preemption settings resolve as Slurm resolves them.** A QOS
  `PreemptExemptTime` now applies under `preempt/partition_prio` too, since
  `acct_policy_get_preemptable_time()` never reads PreemptType. A QOS
  PreemptMode of flags only, such as a bare `WITHIN`, resolves to OFF and
  protects that QOS's jobs; it used to fall back to the cluster's mode. A
  bare OFF still defers, as CLUSTER does. Both are reachable from the
  library only: the CLI's ladders set neither.
- **Jobs that share an id are preempted.** `select_preemptees` kept its
  bookkeeping by job id, so two candidates sharing one overwrote each
  other's entry and nothing was preempted. It is keyed by the job now. Queue
  order and victim order break a final tie on the original submit time after
  the id, so jobs sharing an id no longer depend on list order. The test
  meant to cover this never turned preemption on; it does now.
- **`--json`'s `model` block records every setting that changes the
  result.** It now includes the seed and job count of a synthetic trace, the
  seed of a k8s trace's synthetic limits, the cluster's shape, the priority
  weights and every preemption setting. Before, two runs differing only in
  `--seed` wrote the same provenance.
- **`--sched-params` replaces the config's `SchedulerParameters` whole.**
  Warnings about the discarded string, and its `requeue_delay`, used to stay
  in force.
- **Malformed trace rows are clean errors.** A `--k8s-trace` or `--sacct`
  row with fewer or more fields than the header is refused with the file and
  line. A short row used to be a raw `TypeError` or `AttributeError`
  traceback, and extra fields were read silently.
- **`--nodes`, `--cpus`, `--gpus` and `--jobs` are range-checked**, as the
  fleet loader checks the same values. `--gpus -2` used to run.
- `--preempt-type none` with `--preempt-mode` warns that the mode is
  ignored, as it already did for the other preemption flags.
- **`cycles at bf_max_job_start` is printed** when non-zero. The count
  existed (and was in `--json`), and this changelog said it was in the
  report, but it was not printed. With that limit set, the report read
  `cycles at bf_max_job_test 0` while every cycle stopped early.
- **docs/backfill.svg** showed 0.1.0's numbers, which the command in its
  caption did not produce. It now shows `schedlab --compare-backfill --jobs
  300 --seed 5 --backfill-mode easy` (it is the EASY picture) and prints
  that command.
- `Job.planned_start` and the README called the planned start what `squeue
  --start` shows. It is narrower: a reservation's start only. Slurm also
  keeps the estimate of a job beyond `bf_window`, which the model does not
  compute.
- README statements "checked in this session" were replaced by tests or by
  dates. The tests: the fleet derivation against `k8slab.topology.derive()`,
  both shipped fleet files against PyYAML, and the trace digest pinned to a
  value the k8s lab's own code computed. The S0 k8s reference rows cited a
  working-tree fingerprint nobody can recover. They now cite the tree they
  were regenerated against.
- **Fairshare no longer charges a job's whole future at dispatch, or reads its
  true runtime.** 0.1.0 charged `total_cpus * duration` the moment a job
  started. Usage now accrues for elapsed time only, on `PriorityCalcPeriod`
  ticks, plus a final partial period at job end (`accounting.PriorityEngine`).
  A test runs whole simulations with jobs whose `duration` raises if read by
  anything but the completion event.
- **The job size factor is Slurm's.** 0.1.0 used the job's share of the
  cluster's CPUs. `set_priority_factors()` averages the share of nodes and the
  share of CPUs (favor-small inverts each). They are equal for whole-node jobs
  on identical nodes, so no synthetic-workload number moved. They differ for
  sub-node jobs: an S0 gang of G one-CPU pods on the k8s lab's 26-node,
  1,680-CPU default fleet gets about G/52 instead of G/1680, and the S0 table
  was regenerated.
- **Priorities are Slurm's integers.** `_get_priority_internal()` stores the
  weighted sum as a `uint32_t` of at least 1. 0.1.0 kept floats, so of two
  jobs whose sums differ by less than one the higher sum went first; now they
  tie and the earlier submit goes first. A sum below 1 (a large `nice`)
  becomes 1.
- **EASY shadow time mis-projected a job that started at exactly t=0.**
  `start_time or now` treated a 0.0 start as missing, so that job's end slid
  forward with `now`, the reservation came late, and backfill could delay
  the job it was protecting. Only runs where a job starts at exactly t=0 were
  affected: sacct traces, which are rebased so the earliest submit is 0, and
  hand-built workloads. Synthetic traces and the k8s lab's generated CSVs
  draw an arrival gap before the first job, so their first submit is above 0
  and their EASY numbers were not affected.
- **The sacct loader handed backfill the true runtime as a time limit.**
  `UNLIMITED`, `Partition_Limit` and blank limits fell back to `Elapsed`, and a
  limit shorter than `Elapsed` was raised to it, so exactly the jobs that gave
  backfill no usable estimate were planned with perfect foresight. Limits
  now come from `Timelimit` only, and the column is required. A job without
  a finite limit is planned at 365 days, what `sched/backfill` uses when the
  partition MaxTime is UNLIMITED too (`_set_backfill_timelimits()`,
  `YEAR_MINUTES`), or at `--partition-max-time`. A job that overran its limit
  keeps it, and its runtime is cut to it (the TIMEOUT kill). The run header
  counts both (`trace.load_sacct`, `SacctTrace`).
- **`from_slurm_conf` filled omitted keys with the lab's demo weights.** They
  now take Slurm's defaults: every `PriorityWeight*` is 0.
- **Time strings are parsed as Slurm parses them.** A bare number is minutes,
  and `a:b` is minutes:seconds (`time_str2secs`); it was read as hours.
  Priority durations round up to whole minutes (`time_str2mins`).
- `end_time` is no longer set at dispatch, where it exposed the true runtime.
- `Job` uses identity equality. `list.remove(job)` was comparing every field,
  `duration` included.
- The README's headline and sweep commands omitted `--jobs 300 --seed 5`, the
  flags their numbers came from.

### Added

- **Conservative, Slurm-like backfill** (`--backfill-mode conservative`, now
  the default). It has an event-triggered main scheduler and a periodic full
  pass every `sched_interval`, and backfill cycles every `bf_interval`. The
  main pass stops at its first blocked job, because Slurm removes that job's
  partition nodes from the pass and every partition here spans every node.
  An event pass tests up to `default_queue_depth + 1` jobs (`job_depth++ >
  def_job_limit`). `sched_interval=0` makes every pass a full one, and
  `sched_interval=-1` disables the main scheduler outright, event passes
  included, so only backfill starts jobs. Each backfill cycle plans every
  tested job into a per-node timeline with whole-node reservations, bounded
  by `bf_max_job_test`, `bf_window`, `bf_resolution`, `bf_max_job_start` and
  the per-user, per-partition and per-association caps (with both
  `bf_max_job_assoc` and `bf_max_job_user` set, the association cap wins and
  a warning says so, as in `backfill.c`). `backfill=False` in this mode is
  `sched/builtin`. The previous model remains as `--backfill-mode easy`.
- Cycles that nothing could change are replayed instead of re-planned: same
  queue, no start, end or preemption since, and no job that was beyond
  `bf_window` has come inside it. Jobs at least as large as one already
  beyond the window skip the planner. Both are exact: a test compares start
  times, first planned starts and every cycle's statistics with a run that
  plans every job of every cycle, on whole-node and sub-node workloads with
  default and short windows. The README has a measured cost table.
- `SchedulerParameters` dataclass with slurm.conf(5) defaults and the range
  checks of `backfill.c` and `job_scheduler.c` (0 is valid for
  `default_queue_depth` and `sched_interval`). Values are read like `atoi`.
  Unmodelled keys are reported.
- `SlurmConfig`, which reads `Priority*`, `PriorityFlags`, `SchedulerType`
  and `SchedulerParameters` together. It warns about every setting present
  that would change what Slurm schedules and is not modelled: unmodelled
  flags and parameters, `FairShareDampeningFactor`,
  `PrioritySiteFactorPlugin`, `PriorityParameters`, `OverTimeLimit`,
  `TopologyPlugin`, `SelectTypeParameters`, `PartitionName` and `NodeName`
  lines, and a `PriorityWeightPartition` that has no partition factor to
  weigh. `PriorityWeightQOS` on an sacct trace gets a warning too. Keys with
  no scheduling effect are ignored silently.
  `PriorityDecayHalfLife=0` without `PriorityUsageResetPeriod` is refused,
  as `read_config.c` refuses it. The run header names `sched/builtin` or
  `bf_interval=-1` when backfill is off, and `--sweep` honours both.
  `--compare-backfill` overrides either one for its ON leg and says so.
- Backfill-cycle statistics (cycles, jobs tested per cycle, cycles stopped at
  `bf_max_job_test` or, printed when non-zero, `bf_max_job_start`) in
  `SimulationResult` and the report, plus each job's first planned start
  (`planned_starts`, on `SimulationResult` only).
  `backfilled jobs` counts distinct jobs (`RunRecord.backfilled`). When
  requeued jobs were backfilled again, the report adds `backfill starts`,
  the per-start count `sdiag` reports.
- **Fair Tree fairshare**, now the default as in Slurm since 19.05. Classic is
  selected by `PriorityFlags=NO_FAIR_TREE` (or `DEPTH_OBLIVIOUS`) or by
  `--fairshare-algorithm classic`.
- `PriorityCalcPeriod`, `PriorityUsageResetPeriod`, and the per-algorithm
  usage-visibility lag: Fair Tree priorities see usage up to the last tick,
  classic ones usage as it stood at the start of that tick, one period
  behind. `PriorityUsageResetPeriod` NONE, NOW, DAILY and WEEKLY are
  implemented. DAILY and WEEKLY fall at local midnight and Sunday 00:00; an
  sacct trace places its t=0 in the week from its earliest `Submit`, and
  other traces take t=0 as Sunday 00:00 with a warning. The calendar-month
  values are refused by the library (`PriorityEngine`); the CLI, reading
  them from a slurm.conf, warns and simulates NONE. `--calc-period 0` gives
  an idealised no-lag mode, which refreshes priorities at every event with a
  job pending: timer-driven main passes, backfill cycles and requeue begin
  times included, not only submits and completions. A fractional
  `--calc-period` is rounded up to whole minutes, as Slurm reads it, with a
  warning.
- Per-tick fairshare snapshots on `SimulationResult`.
- `Job.user` (read from an optional sacct `User` column), `Job.priority` (the
  controller's current snapshot) and `Job.planned_start`.
- CLI: `--backfill-mode`, `--fairshare-algorithm`, `--calc-period`,
  `--sched-params`, and `--seeds N` to print the spread over seeds (with work
  lost and grace-locked CPU-hours when preemption is on).
  `--time-limit-model exact|padded|synthetic` also applies to synthetic
  traces, replacing the generator's padding and nothing else, and
  `--time-limit-factor` below 1 is refused for every trace.
  `--partition-max-time` sets the planning limit for sacct jobs without a
  finite one. A `--json` path that cannot be written is a clean error, and
  a missing directory is caught before the run.
- `scripts/cost_table.py`, which regenerates the README's cost table through
  the CLI's own configuration. `scripts/planned_start_slips.py` gives the
  planned-start slips and backfill-restart counts, which no report prints.
  `scripts/s0_table.py` gives the S0 table; with `--reference` it adds the
  k8s reference rows and the k8s lab's working-tree fingerprint.
- PyYAML in the `dev` extra, never a runtime dependency, so the YAML-subset
  cross-checks run in CI.
- ruff and strict mypy configuration, run in CI.
- **Preemption** (`preempt.py`; `--preempt-type`, `--preempt-mode`,
  `--grace-time`, `--checkpoint-fraction`, `--preempt-exempt-time`; the
  `Preempt*`, `JobRequeue`, `KillWait` and `MessageTimeout` keys and
  `SchedulerParameters=requeue_delay`). Covers `preempt/partition_prio` and
  `preempt/qos`, CANCEL and REQUEUE with WITHIN and PRIORITY, GraceTime (the
  victim keeps its nodes until its reset end time), PreemptExemptTime (`-1`,
  `INFINITE` and `UNLIMITED` mean none, "equivalent to 0", from any source),
  youngest_first, reorder_count and strict_order. On requeue, submit and
  begin times are reset as in `batch_requeue_fini()`. QOS preempt lists that
  form a loop are refused, as slurmdbd refuses them; acyclic lists that are
  not a ladder are accepted, and a job can then be preempted in the same
  main pass that started it. The completion heap has cancellable per-run
  entries, and a preemption bumps the backfill replay key. New metrics:
  preemptions, work lost and grace-locked CPU/GPU-hours (disjoint, as in the
  k8s lab), and thrash. SUSPEND is refused; the GANG flag is parsed and
  reported as not modelled.
- **Partition `PriorityTier`** in queue order ("jobs that can preempt", then
  tier, then priority), in both modes.
- **Heterogeneous fleets** from the k8s lab's fleet YAML (`fleet.py`,
  `--fleet`). Nodes get the k8s lab's names and derived rack/switch. The
  loader reads a strict YAML subset; no PyYAML dependency. It refuses
  exactly the words PyYAML's resolver reads as booleans besides true/false
  (`yes`/`no`/`on`/`off` in three casings). `y`, `n` and other casings stay
  strings, as in PyYAML.
- **The S0 adapter** (`k8strace.py`, `--k8s-trace`): the k8s lab's trace CSV
  as Slurm jobs, with a mandatory `--time-limit-model exact|padded|synthetic`,
  a documented priority mapping (`--k8s-priority qos|tier|none`), and the k8s
  lab's trace digest.
- **Definition C** structural fragmentation (`fragmentation.py`), the same
  pure function as the k8s lab's, with the shared golden vector as a test.
  It validates like the k8s function too: every sample is checked before
  anything is integrated, the final one included, and unknown nodes are
  refused. Also a CPU version and an any-resource variant. The simulator now records
  per-node free GPUs and CPUs from trace t=0 at every change
  (`SimulationResult.node_samples`, `free_gpu_samples()`).
- **GPU utilization** under the k8s lab's definition, in the report. It and
  the work-lost and grace-locked GPU-hours are computed once for the report
  and the JSON (`metrics.gpu_use`, `metrics.preemption_cost`).
- **Metric parity** (`parity.py`, `--json PATH`): the k8s lab's results.json
  field names and definitions wherever the quantities coincide. That covers
  wait (pending time only under eviction, as there), wait by footprint bucket (p95 by true nearest-rank, as there; `p95_wait`
  keeps the rounded rank), the large-job starvation ratio, size–wait
  Spearman, placement tier shares (null when no multi-node job ran, as
  there) and fairness, and the k8s lab's
  execution-layer keys: `gpu_hours_demanded`, `preemptions` (pod attempts),
  `preempted_gpu_hours_lost` and `grace_locked_gpu_hours`. The k8s lab added
  those keys while this work was in progress, and they were re-read before
  release. The gang-stranding, startup-overhead and topology-extension
  metrics are reported as 0, each with its reason. Fragmentation A/B, the
  placement penalty and the harness fields are listed as not reported, each
  with its reason.
- `Job.qos`, `first_start_time`, preemption state and per-run `RunRecord`s;
  `Node.name`, `rack`, `switch`, `node_class`, `nvlink`; `Cluster.name`,
  `topology_declared`, `domains()`, `max_node_gpus`.

### Removed

- `trace._timestamp`, which had no callers, and the `replay_unchanged`
  argument of `SlurmScheduler`, which nothing passed. The equivalence tests
  flip `REPLAY_UNCHANGED_BACKFILL_CYCLES`, the one switch.

### Changed

- `PriorityDecayHalfLife` defaults to 7 days (Slurm's default), not 5.
- The default backfill mode is `conservative`, and the default fairshare
  algorithm is `fair_tree`. Every README number was re-run. Two existing
  tests now state their mode explicitly: the EASY timing cases, and the
  classic-formula case.
- `Scheduler` is now `EasyScheduler` (the old name is an alias). It reads
  `job.priority` instead of computing priorities itself. Its constructor
  takes `(cluster, backfill)`, plus an optional `PreemptionConfig` whose
  partition table sets the `PriorityTier` order.
- A job whose runtime exceeds its own `time_limit` is refused with a
  `ValueError` naming it, in both modes. Slurm would kill it at the limit
  (TIMEOUT), which is not modelled; 0.1.0's EASY mode ran it past its limit.
  The trace loaders never produce such a job.
- `from_sacct` gives every job a distinct id. Array tasks (`5000_1`), het
  components (`6000+0`) and repeated ids get fresh ones above the largest
  plain id; 0.1.0 collapsed `5000_1` and `5000_2` onto 5000. Jobs that do
  share an id are still simulated correctly: internal state is keyed by job,
  not id.
- With preemption, utilization counts delivered work only, the k8s lab's
  rule. The text report's wait runs to the start of a job's last run.
  `--json`'s wait keys count pending time only, the k8s lab's rule under
  eviction. Without preemption all three are unchanged.
- The text report has new lines: GPU utilization, Definition C (node level,
  plus rack and switch for a fleet with declared topology), the backfill-cycle
  statistics in conservative mode, and the preemption block when it is on.
- A cancelled job (`State.PREEMPTED`) is left out of bounded slowdown.
- The README no longer claims that accurate wall-clock requests are "worth
  more than any weight you can tune". Measured over ten synthetic seeds,
  exact limits raised utilization and also raised mean wait, p95 wait and
  bounded slowdown (README, "Time limits: what accurate requests buy,
  measured"). The uncited "published trace studies" remark on the synthetic
  padding is gone too.

### Known limits (measured, and pinned in tests)

- A job's first planned start is a prediction, not a bound. "No job starts
  later than its first planned start" holds exactly only for whole-node jobs
  under FIFO priorities, exact limits, a `bf_window` covering every plan, and
  1 s `bf_resolution` and `bf_interval`: the conditions of the test that pins
  it. Outside them, higher-priority arrivals, priority refreshes and jobs
  entering `bf_window` move planned starts later, and for sub-node jobs so
  do whole-node reservations under from-scratch re-planning. On the
  synthetic workload (every job whole-node) with the CLI defaults, `--jobs
  300`, seeds 0-9, 591 of 2,708 planned jobs started after their first
  planned start, 497 by more than an hour (`scripts/planned_start_slips.py`). An earlier version of this entry
  said the guarantee held for whole-node jobs without stating those
  conditions.
- On the synthetic workload, the utilization difference between EASY and
  conservative changes sign across seeds. See "One seed is an anecdote" in
  the README.
- S0's time-limit model moves results in both directions. On five k8s
  default-profile traces, exact limits gave the highest mean wait (5 of 5
  seeds) and the lowest large-job starvation ratio (5 of 5), compared with
  ×3 padding. See "Running S0" in the README.
- Preemption is triggered by the main scheduler only, and the victim search
  uses this simulator's first-fit in place of cons_tres's node selection.
- The conservative planner's cost grows with queue depth: on the README's
  contended synthetic traces, 10.2 s for 1,200 jobs and 109.4 s for 4,000 on
  the machine it was measured on, through the CLI's configuration
  (`scripts/cost_table.py`). The table first published here came from runs
  without a fairshare tree and did not say so.

## [0.1.0]

First public release.

Test Slurm priority and backfill policy against a real job trace.

- Core tool implemented and covered by tests.
- `make demo` (or equivalent) runs against a synthetic backend, no special
  hardware required.
- CI runs the test suite on every push.

This is a `0.x` release: the behaviour is tested and the safety properties are
asserted, but flags and metric names may still change before `1.0.0`.

[Unreleased]: https://github.com/Zhanyl-tech/slurm-scheduler-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zhanyl-tech/slurm-scheduler-lab/releases/tag/v0.1.0
