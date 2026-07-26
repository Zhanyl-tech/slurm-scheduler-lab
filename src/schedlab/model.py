"""Core scheduling types: jobs, partitions, and the cluster's resource state.

Resources are modelled as CPUs and GPUs per node. That is enough to reproduce
the two effects that actually drive Slurm queue behaviour — node-level
fragmentation and GPU scarcity — without simulating topology or NUMA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"


@dataclass
class Job:
    job_id: int
    account: str
    submit_time: float
    #: Wall-clock the job actually needs. In a replay this comes from the trace;
    #: the scheduler is never allowed to read it (see `time_limit`).
    duration: float
    #: The user's requested limit. Backfill plans against *this*, not `duration`,
    #: which is why over-requesting hurts everyone — the central lesson the
    #: simulator exists to make measurable.
    time_limit: float
    nodes: int = 1
    cpus_per_node: int = 1
    gpus_per_node: int = 0
    qos_factor: float = 0.0
    partition: str = "normal"
    nice: int = 0

    state: State = State.PENDING
    start_time: float | None = None
    end_time: float | None = None
    allocated: list[int] = field(default_factory=list)
    #: Priority at the moment of dispatch, kept for post-hoc analysis.
    dispatch_priority: float = 0.0

    @property
    def total_cpus(self) -> int:
        return self.nodes * self.cpus_per_node

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def wait_time(self) -> float:
        if self.start_time is None:
            raise ValueError(f"job {self.job_id} never started")
        return self.start_time - self.submit_time

    @property
    def turnaround(self) -> float:
        if self.end_time is None:
            raise ValueError(f"job {self.job_id} never finished")
        return self.end_time - self.submit_time

    def bounded_slowdown(self, threshold: float = 60.0) -> float:
        """Turnaround relative to runtime, floored so short jobs don't dominate.

        The standard metric from the parallel-workload literature: without the
        threshold, a 2-second job waiting 20 seconds reports a slowdown of 11
        and swamps the mean.
        """
        return self.turnaround / max(self.duration, threshold)


@dataclass
class Node:
    node_id: int
    cpus: int
    gpus: int = 0
    free_cpus: int = 0
    free_gpus: int = 0

    def __post_init__(self) -> None:
        self.free_cpus = self.cpus
        self.free_gpus = self.gpus

    def fits(self, cpus: int, gpus: int) -> bool:
        return self.free_cpus >= cpus and self.free_gpus >= gpus

    def allocate(self, cpus: int, gpus: int) -> None:
        if not self.fits(cpus, gpus):
            raise ValueError(f"node {self.node_id} cannot fit {cpus}c/{gpus}g")
        self.free_cpus -= cpus
        self.free_gpus -= gpus

    def release(self, cpus: int, gpus: int) -> None:
        self.free_cpus = min(self.cpus, self.free_cpus + cpus)
        self.free_gpus = min(self.gpus, self.free_gpus + gpus)


@dataclass
class Cluster:
    nodes: list[Node]

    @classmethod
    def homogeneous(cls, count: int, cpus: int, gpus: int = 0) -> Cluster:
        return cls([Node(i, cpus, gpus) for i in range(count)])

    @property
    def total_cpus(self) -> int:
        return sum(n.cpus for n in self.nodes)

    @property
    def total_gpus(self) -> int:
        return sum(n.gpus for n in self.nodes)

    @property
    def busy_cpus(self) -> int:
        return sum(n.cpus - n.free_cpus for n in self.nodes)

    def find_nodes(self, job: Job) -> list[int] | None:
        """First-fit node selection. Returns node ids, or None if it won't fit.

        Slurm's real selection is topology-aware; first-fit over a sorted node
        list reproduces the fragmentation behaviour that matters here while
        staying deterministic.
        """
        chosen: list[int] = []
        for node in self.nodes:
            if node.fits(job.cpus_per_node, job.gpus_per_node):
                chosen.append(node.node_id)
                if len(chosen) == job.nodes:
                    return chosen
        return None

    def allocate(self, job: Job, node_ids: list[int]) -> None:
        for nid in node_ids:
            self.nodes[nid].allocate(job.cpus_per_node, job.gpus_per_node)
        job.allocated = list(node_ids)

    def release(self, job: Job) -> None:
        for nid in job.allocated:
            self.nodes[nid].release(job.cpus_per_node, job.gpus_per_node)
        job.allocated = []
