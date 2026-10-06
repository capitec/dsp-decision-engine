"""The handle `start()` returns: one experiment run in a background thread."""
from __future__ import annotations

import threading
import uuid
from typing import Callable

from decider.lab.experiments._jobs import JobHandle, JobStatus


class RunHandle:
    """A running experiment: poll `status` and `progress`, `cancel()` it, wait for `result()`."""

    def __init__(self, work: Callable[[JobHandle], object]):
        self._job = JobHandle(job_id=uuid.uuid4().hex, kind="experiment_run")
        self._result: object = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._main, args=(work,), daemon=True)
        self._thread.start()

    def _main(self, work) -> None:
        try:
            self._result = work(self._job)
        except BaseException as e:  # noqa: BLE001  re-raised from result()
            self._error = e
            self._job.fail(f"{type(e).__name__}: {e}")

    @property
    def status(self) -> JobStatus:
        """`QUEUED`, `RUNNING`, then `SUCCEEDED`, `FAILED` or `CANCELLED`."""
        return self._job.snapshot().status

    @property
    def progress(self) -> float | None:
        """Fraction of records done, or `None` before the total is known."""
        return self._job.snapshot().fraction

    def cancel(self) -> None:
        """Stop after the batch in progress; `result()` then returns the partial run."""
        self._job.cancel()

    def result(self, timeout: float | None = None):
        """Wait for the run and return its `ExperimentResult`; raise what the run raised."""
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("the experiment is still running")
        if self._error is not None:
            raise self._error
        return self._result
