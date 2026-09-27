"""Conservative, Slurm-like backfill: a per-node plan rebuilt every cycle.

EASY backfill (Lifka, 1995; `simulate.EasyScheduler`) reserves for exactly one
job — the first that does not fit — and lets anything else jump the queue if
it cannot delay *that* job. Slurm's `sched/backfill` does more. From the
Scheduling Configuration Guide (https://slurm.schedmd.com/sched_config.html):

    "Backfill scheduling will start lower priority jobs if doing so does not
    delay the expected start time of any higher priority jobs. [...] If the
    job under consideration can start immediately without impacting the
    expected start time of any higher priority job, then it does so.
    Otherwise the resources required by the job will be reserved during the
    job's expected execution time."

and: "For performance reasons, the backfill scheduler reserves whole nodes for
jobs, even if jobs don't require whole nodes."

That is conservative backfill (Mu'alem & Feitelson, 2001), bounded:

* only the first `bf_max_job_test` jobs a cycle *tries* are planned;
* a job whose earliest start lies beyond `bf_window` gets no reservation, so it
  protects nothing and can itself be pushed back;
* reservations are quantised outward to `bf_resolution` — start floored, end
  rounded up (`_set_slot_time()`, backfill.c L1973-L1981) — so a planned job
  blocks a little more time than it asked for, and a job being tested must
  clear reservations with its own quantised interval. Running jobs are *not*
  rounded: backfill.c only reserves for running jobs (`_bf_reserve_running`,
  rounding up) under `bf_running_job_reserve` or `bf_licenses`, both off by
  default (L1075-L1084, L2410). Their exact `start + time_limit` is used,
  which assumes the select plugin's will-run test also sees exact end times
  (not verified in select/cons_tres).

Every cycle rebuilds the plan from scratch, as `_attempt_backfill()` in
backfill.c does (SchedMD/slurm@9f9da53).

Unlike EASY mode, accounting is **per node**: a job needs `nodes` distinct
nodes that each have `cpus_per_node`/`gpus_per_node` free for its whole
planned interval. The EASY mode's aggregate shadow time can be optimistic when
free CPUs are scattered across nodes; this plan cannot.

What is not modelled: topology, `bf_continue`/lock yielding, `bf_max_time`
(a wall-clock compute budget with no simulated analogue), licences, advanced
reservations, and heterogeneous jobs. Preemption is triggered by the main
scheduler only (see preempt.py); a backfill cycle plans a preemptor around
the running jobs like any other job.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from .model import Cluster, Job


def floor_to(t: float, resolution: float) -> float:
    return math.floor(t / resolution) * resolution


def ceil_to(t: float, resolution: float) -> float:
    return math.ceil(t / resolution) * resolution


@dataclass
class _NodeTimeline:
    cpus: int
    gpus: int
    #: Running (or started-this-cycle) allocations: (planned_end, cpus, gpus),
    #: sorted by planned end. Capacity only ever returns over time, because
    #: every *future* start in the plan is a whole-node reservation below.
    allocs: list[tuple[float, int, int]] = field(default_factory=list)
    #: Whole-node reservations [start, end), sorted. They never overlap on one
    #: node, so `res_ends` is sorted too and can be bisected.
    res_starts: list[float] = field(default_factory=list)
    res_ends: list[float] = field(default_factory=list)
    free_cpus: int = -1
    free_gpus: int = -1

    def __post_init__(self) -> None:
        self.free_cpus = self.cpus
        self.free_gpus = self.gpus

    def free_from(self, cpus: int, gpus: int, earliest: float) -> float:
        """Earliest time >= `earliest` with `cpus`/`gpus` free for good."""
        if cpus > self.cpus or gpus > self.gpus:
            return math.inf
        free_c, free_g = self.free_cpus, self.free_gpus
        if free_c >= cpus and free_g >= gpus:
            return earliest
        for end, c, g in self.allocs:
            free_c += c
            free_g += g
            if free_c >= cpus and free_g >= gpus:
                return end if end > earliest else earliest
        return math.inf

    def feasible_starts(
        self, cpus: int, gpus: int, length: float, lo: float, hi: float
    ) -> list[tuple[float, float]]:
        """Closed intervals of start times t in [lo, hi] this node can take.

        A start t is feasible if the node has the resources from t on and the
        job's quantised interval [floor(t), ceil(t + length)) meets no
        reservation [s, e). Both reservation bounds sit on the resolution
        grid, so that reduces exactly to: t + length <= s, or t >= e.
        """
        cur = self.free_from(cpus, gpus, lo)
        if cur > hi:
            return []
        out: list[tuple[float, float]] = []
        starts, ends = self.res_starts, self.res_ends
        for i in range(bisect.bisect_right(ends, cur), len(starts)):
            latest = starts[i] - length
            if latest >= cur:
                out.append((cur, latest if latest < hi else hi))
            if ends[i] > cur:
                cur = ends[i]
            if cur > hi:
                return out
        out.append((cur, hi))
        return out

    def first_start(
        self, cpus: int, gpus: int, length: float, lo: float, hi: float
    ) -> float | None:
        """The start of the first interval `feasible_starts` would return,
        without building the list; None if there is none."""
        cur = self.free_from(cpus, gpus, lo)
        if cur > hi:
            return None
        starts, ends = self.res_starts, self.res_ends
        for i in range(bisect.bisect_right(ends, cur), len(starts)):
            if starts[i] - length >= cur:
                return cur
            if ends[i] > cur:
                cur = ends[i]
                if cur > hi:
                    return None
        return cur

    def add_alloc(self, end: float, cpus: int, gpus: int) -> None:
        bisect.insort(self.allocs, (end, cpus, gpus))
        self.free_cpus -= cpus
        self.free_gpus -= gpus

    def add_reservation(self, start: float, end: float) -> None:
        i = bisect.bisect_left(self.res_starts, start)
        self.res_starts.insert(i, start)
        self.res_ends.insert(i, end)


class NodeSpace:
    """The plan for one backfill cycle, seeded with the running jobs.

    Running jobs hold their resources until `start + time_limit` — the
    requested limit, never the true runtime — or, once selected for
    preemption, until their grace deadline (`Job.expected_end`, which is the
    end time Slurm itself resets).
    """

    def __init__(self, cluster: Cluster, running: list[Job], resolution: float) -> None:
        self.resolution = resolution
        self.nodes = [_NodeTimeline(n.cpus, n.gpus) for n in cluster.nodes]
        for job in running:
            end = job.expected_end
            for nid in job.allocated:
                self.nodes[nid].add_alloc(end, job.cpus_per_node, job.gpus_per_node)

    def earliest_start(
        self, job: Job, now: float, horizon: float
    ) -> tuple[float, list[int]] | None:
        """Earliest t in [now, horizon] where `job` fits, and the nodes to use.

        A sweep over every node's feasible-start intervals: the answer is the
        first instant at which `job.nodes` intervals overlap. Nodes are chosen
        in id order, so the plan is deterministic and agrees with the
        simulator's first-fit allocation.

        Shortcuts that return exactly what the sweep would (pinned against
        a plain sweep by `test_planner_shortcuts_pick_the_same_time_and_nodes_as_the_sweep`,
        and against brute force by `test_planner_matches_brute_force`):

        * nodes too small ever to hold the job are skipped;
        * a one-node job needs only each node's *first* feasible start, and
          the search on later nodes can stop at the best found so far; the
          earliest wins, ties to the lowest node id;
        * a multi-node job that `job.nodes` nodes can take at `now` — every
          backfill start — returns as soon as that many are found, in id
          order, without sorting anything.
        """
        length = job.time_limit
        cpus, gpus, want = job.cpus_per_node, job.gpus_per_node, job.nodes
        if now > horizon:
            return None

        if want == 1:
            best: float | None = None
            best_nid = -1
            bound = horizon
            for nid, node in enumerate(self.nodes):
                if node.cpus < cpus or node.gpus < gpus:
                    continue
                t = node.first_start(cpus, gpus, length, now, bound)
                if t is not None and (best is None or t < best):
                    best, best_nid, bound = t, nid, t
                    if t <= now:
                        break  # nothing is earlier, and this is the lowest id
            return None if best is None else (best, [best_nid])

        events: list[tuple[float, int, int]] = []
        ready: list[int] = []
        spans_found = 0
        for nid, node in enumerate(self.nodes):
            if node.cpus < cpus or node.gpus < gpus:
                continue
            spans = node.feasible_starts(cpus, gpus, length, now, horizon)
            if not spans:
                continue
            spans_found += 1
            if spans[0][0] <= now:
                ready.append(nid)
                if len(ready) == want:
                    return now, ready
            for a, b in spans:
                events.append((a, 0, nid))  # 0 sorts opens before closes
                events.append((b, 1, nid))
        if spans_found < want:
            return None
        events.sort()
        # Nodes with an interval open at the sweep's current time. When the
        # count reaches `want` at time t, every node containing t is either
        # open or opens at t later in the order (closes at t sort after
        # opens), and the lowest ids among them are chosen.
        open_nodes: set[int] = set()
        for i, (t, kind, nid) in enumerate(events):
            if kind == 1:
                open_nodes.discard(nid)
                continue
            open_nodes.add(nid)
            if len(open_nodes) >= want:
                chosen = set(open_nodes)
                for t2, kind2, nid2 in events[i + 1 :]:
                    if t2 != t or kind2 != 0:
                        break
                    chosen.add(nid2)
                return t, sorted(chosen)[:want]
        return None

    def start(self, job: Job, node_ids: list[int], now: float) -> None:
        """Record a job started now: from here on it is a running job."""
        end = now + job.time_limit
        for nid in node_ids:
            self.nodes[nid].add_alloc(end, job.cpus_per_node, job.gpus_per_node)

    def reserve(self, job: Job, node_ids: list[int], start: float) -> float:
        """Reserve whole nodes for a planned job; returns the planned end."""
        begin = floor_to(start, self.resolution)
        end = ceil_to(start + job.time_limit, self.resolution)
        for nid in node_ids:
            self.nodes[nid].add_reservation(begin, end)
        return end


@dataclass
class BackfillCycle:
    """What one backfill cycle did."""

    time: float
    #: Pending jobs when the cycle began.
    queue_length: int
    #: Jobs that reached the placement test (what `bf_max_job_test` bounds).
    jobs_tested: int
    jobs_started: int
    reservations: int
    #: Tested jobs whose earliest start lay beyond `bf_window`.
    beyond_window: int
    #: Jobs skipped by per-user / per-partition / per-association limits.
    skipped_by_limits: int
    #: The cycle stopped at `bf_max_job_test` with jobs still untested.
    hit_max_job_test: bool
    #: The cycle stopped at `bf_max_job_start`.
    hit_max_job_start: bool
    #: True when the cycle was not re-evaluated because nothing it depends on
    #: had changed since the last evaluated cycle (see `SlurmScheduler`); its
    #: numbers are copied from that cycle.
    replayed: bool = False


@dataclass
class BackfillStats:
    """All cycles of one run. Only cycles with a non-empty queue are counted."""

    cycles: list[BackfillCycle] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.cycles)

    @property
    def evaluated(self) -> int:
        return sum(1 for c in self.cycles if not c.replayed)

    @property
    def mean_tested(self) -> float:
        return sum(c.jobs_tested for c in self.cycles) / len(self.cycles) if self.cycles else 0.0

    @property
    def max_tested(self) -> int:
        return max((c.jobs_tested for c in self.cycles), default=0)

    @property
    def hit_max_job_test(self) -> int:
        return sum(1 for c in self.cycles if c.hit_max_job_test)

    @property
    def hit_max_job_start(self) -> int:
        return sum(1 for c in self.cycles if c.hit_max_job_start)

    @property
    def jobs_started(self) -> int:
        return sum(c.jobs_started for c in self.cycles if not c.replayed)
