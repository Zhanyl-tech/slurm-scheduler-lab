# slurm-scheduler-lab

Test Slurm priority and backfill policy against a job trace **before** you put it
on a live controller.

Changing `PriorityWeightFairshare` on a production cluster is a slow, expensive
experiment: the feedback loop is days long, the blast radius is everyone's
queue, and the only rollback signal is people complaining. This runs the same
policy against a workload in about a second.

```bash
pip install -e .
schedlab --compare-backfill
```

```
300 jobs · 16 nodes × 8 CPU / 2 GPU

backfill OFF                          backfill ON
  cpu utilization      72.2 %           cpu utilization      83.6 %
  mean wait          1913.0 min         mean wait           373.7 min
  median wait        2303.3 min         median wait         131.9 min
  bounded slowdown    373.18            bounded slowdown     55.66
```

---

## What it models

**Multifactor priority** — the real formula from `priority/multifactor`:

```
priority = weight_age       × age_factor
         + weight_fairshare × fairshare_factor      # F = 2^(-U_norm / S_norm)
         + weight_jobsize   × jobsize_factor
         + weight_partition × partition_factor
         + weight_qos       × qos_factor
         - nice
```

Every factor is normalised to `[0, 1]`, so the weights alone decide what the
queue optimises for. Fairshare usage decays on a configurable half-life, exactly
as `PriorityDecayHalfLife` does.

**EASY backfill** (Lifka, 1995) — sort by priority, start what fits, and when a
job does not fit, reserve for it at its *shadow time*: the earliest moment
enough resources free up assuming every running job uses its full time limit.
Lower-priority jobs may still start, but only if they provably cannot delay
that reservation.

The scheduler is **never allowed to read a job's true runtime** — only its
requested `time_limit`. That asymmetry is the point of the whole exercise (see
below).

## Point it at your own cluster

Read the weights straight out of the config you are about to deploy:

```bash
schedlab --slurm-conf /etc/slurm/slurm.conf --nodes 64 --cpus 32 --gpus 4
```

Replay a real trace instead of a synthetic one:

```bash
sacct -a -X --parsable2 --starttime=now-30days \
      --format=JobID,Account,Submit,Elapsed,Timelimit,NNodes,ReqCPUS,ReqTRES \
      > trace.txt

schedlab --sacct trace.txt --nodes 64 --cpus 32 --gpus 4
```

Sweep one weight and watch the tradeoff move:

```bash
$ schedlab --sweep jobsize --jobs 300

  weight=0        util  87.3%  mean wait  352.9 min  p95 1978.4 min  slowdown 52.08
  weight=1000     util  90.6%  mean wait  348.2 min  p95 1879.4 min  slowdown 58.08
  weight=10000    util  83.6%  mean wait  373.7 min  p95 1817.7 min  slowdown 55.66
  weight=100000   util  93.4%  mean wait  496.7 min  p95 1904.7 min  slowdown 60.79
```

Cranking `PriorityWeightJobsize` buys the highest utilization in the sweep
(93.4%) and the worst mean wait (497 min). Big jobs go first, the machine stays
full, and everything small waits behind them. Whether that is the right call
depends on what the cluster is for — which is precisely the decision the weights
encode, and precisely what is hard to reason about without running it.

## Why time limits are the real lever

Backfill plans against what users *request*, not what their jobs actually need.
The default synthetic workload pads requests ~3x, which matches what published
trace studies keep finding — and what `--compare-backfill` reports as
`time-limit accuracy 32.2%`.

The consequence is measurable, and it's the case pinned in
`test_backfill_plans_against_time_limit_not_true_runtime`: two identical
workloads, same real runtimes, differing only in requested wall-clock. The
honest one backfills into the gap immediately. The padded one does not, because
the scheduler cannot prove the job would finish before the reservation needs its
nodes.

**Getting users to request accurate wall-clock is often worth more than any
weight you can tune.** This gives you the number to make that argument with.

## Metrics

| Metric | Reads as |
| --- | --- |
| `cpu utilization` | Delivered CPU-seconds ÷ cluster capacity over the makespan |
| `mean / median / p95 wait` | Submit → start. p95 is where users form their opinion |
| `bounded slowdown` | Turnaround ÷ max(runtime, 60s) — the standard fairness metric, floored so 2-second jobs don't dominate |
| `backfilled jobs` | How many started only because backfill let them |
| `time-limit accuracy` | Runtime ÷ requested. Low is the biggest cause of bad backfill decisions |
| `mean wait by account` | Whether fairshare is doing anything |

## Scope and simplifications

Deliberately not a Slurm reimplementation. What it leaves out:

- **Topology.** Node selection is first-fit over an ordered list; no switch or
  NUMA awareness. Reproduces fragmentation behaviour, not network placement.
- **Reservation accounting is aggregate.** Shadow time is computed on total
  CPU/GPU counts rather than per node, so a reservation is slightly optimistic
  in heavily fragmented states.
- **Flat fairshare.** One level of accounts, not the full hierarchical tree.
- **No preemption, gang scheduling, job arrays, or licences.**

Jobs larger than the cluster are reported on `result.unschedulable` rather than
raising — on a real trace that usually means the trace and the cluster
definition disagree, and you want to see which.

## Development

```bash
pip install -e ".[dev]"
pytest -q          # 23 tests
```

The suite pins the behaviours worth trusting: the EASY reservation refuses jobs
that would delay it, fairshare halves at exactly one share of usage, backfill
beats no-backfill on a generated workload, capacity is never exceeded, and runs
are deterministic across identical seeds.

## License

MIT
