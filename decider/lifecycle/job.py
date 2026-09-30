from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class JobStatus(str, Enum):
    """The lifecycle of a finite job: queued, running, then one terminal status."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT})


class PartialResult(BaseModel):
    """What a job finished before it stopped.

    A cancelled or failed experiment run can report how many of its scenarios
    finished (`completed`) so the caller can resume or show what was lost.

    Example::

        PartialResult(description="3 of 5 scenarios", completed=3)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = ""
    completed: int = 0


class Job(BaseModel):
    """An immutable snapshot of one finite job: the wire form VS Code and MCP read.

    A job is finite long-running work — data loading, a check run, revision
    comparison, experiment execution, trace export — not a live debug session
    (a paused `Session` is deliberately not a job). `JobHandle` mutates one and
    hands out these snapshots.

    Example::

        handle = JobHandle(job_id="j1", kind="experiment_run")
        handle.start()
        handle.progress(2, total=5)
        handle.succeed()
        handle.snapshot().status  # JobStatus.SUCCEEDED
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    message: str = ""
    done: int = 0
    total: int | None = None
    logs: tuple[str, ...] = ()
    error: str | None = None
    partial: PartialResult | None = None
    result: Any = None
    timeout_seconds: float | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def fraction(self) -> float | None:
        """Progress as `done / total`, or `None` when the total is unknown."""
        if not self.total:
            return None
        return self.done / self.total

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


class JobHandle:
    """The mutable controller for one job; `snapshot()` returns its immutable `Job`.

    `cancel()` may be called from any thread; the running operation checks
    `cancelled` (and `timed_out`) and stops. Terminal transitions are one-way:
    once a job has ended, later calls are ignored.

    Example::

        handle = JobHandle(job_id="j1", kind="check_run", timeout=30.0)
        handle.start()
        for i, row in enumerate(rows, 1):
            if handle.cancelled:
                break
            if handle.timed_out:
                handle.timeout()
                break
            handle.progress(i, total=len(rows))
        else:
            handle.succeed(result={"rows": len(rows)})
    """

    def __init__(self, job_id: str, kind: str, timeout: float | None = None):
        self._snapshot = Job(job_id=job_id, kind=kind, timeout_seconds=timeout)
        self._cancel = threading.Event()
        self._deadline: float | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def timed_out(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    @property
    def terminal(self) -> bool:
        return self._snapshot.terminal

    def snapshot(self) -> Job:
        return self._snapshot

    def start(self) -> None:
        if self._snapshot.status is not JobStatus.QUEUED:
            return
        if self._snapshot.timeout_seconds is not None:
            self._deadline = time.monotonic() + self._snapshot.timeout_seconds
        self._update(status=JobStatus.RUNNING, started_at=time.time())

    def progress(self, done: int, total: int | None = None, message: str = "") -> None:
        if self.terminal:
            return
        fields: dict[str, Any] = {"done": done, "message": message}
        if total is not None:
            fields["total"] = total
        self._update(**fields)

    def log(self, message: str) -> None:
        if self.terminal:
            return
        self._update(logs=self._snapshot.logs + (message,))

    def cancel(self) -> None:
        self._cancel.set()
        self._finish(JobStatus.CANCELLED)

    def timeout(self) -> None:
        self._finish(JobStatus.TIMED_OUT)

    def succeed(self, result: Any = None, partial: PartialResult | None = None) -> None:
        self._finish(JobStatus.SUCCEEDED, result=result, partial=partial)

    def fail(self, error: str, partial: PartialResult | None = None) -> None:
        self._finish(JobStatus.FAILED, error=error, partial=partial)

    def _finish(self, status: JobStatus, **fields: Any) -> None:
        if self.terminal:
            return
        self._update(status=status, finished_at=time.time(), **fields)

    def _update(self, **fields: Any) -> None:
        self._snapshot = self._snapshot.model_copy(update=fields)
