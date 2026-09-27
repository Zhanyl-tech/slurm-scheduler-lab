"""Shared test helpers. Imported by the test modules, not collected itself."""

from __future__ import annotations

import copy
from typing import Any

from schedlab.model import Job
from schedlab.priority import PriorityWeights


def job(job_id: int, submit: float, duration: float, nodes: int = 1, **kw: Any) -> Job:
    """A job whose time limit defaults to its true runtime (a perfect estimate)."""
    kw.setdefault("time_limit", duration)
    return Job(
        job_id=job_id,
        account=kw.pop("account", "acct"),
        submit_time=submit,
        duration=duration,
        nodes=nodes,
        cpus_per_node=kw.pop("cpus_per_node", 1),
        **kw,
    )


def started(j: Job) -> float:
    """The job's start time, asserting it started (narrows the Optional)."""
    assert j.start_time is not None, f"job {j.job_id} never started"
    return j.start_time


def by_id(jobs: list[Job]) -> dict[int, Job]:
    return {j.job_id: j for j in jobs}


def clone(jobs: list[Job]) -> list[Job]:
    return copy.deepcopy(jobs)


def weights_only(**kw: float) -> PriorityWeights:
    """Every weight 0 except the ones given — isolates a single factor."""
    base: dict[str, Any] = dict.fromkeys(("age", "fairshare", "jobsize", "partition", "qos"), 0.0)
    base.update(kw)
    return PriorityWeights(**base)
