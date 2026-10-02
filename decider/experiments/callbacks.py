"""The hooks the experiment runner calls, what they receive, and the built-in callbacks.

The runner feeds every scenario the same records one batch at a time, the
baseline first, so each hook sees one scenario's batch next to the baseline's
output for the same records. A `Callback` sees every scenario; a `Metric` (see
`decider.experiments.metrics`) is a callback copied per scenario.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from decider.lifecycle import JobHandle


@dataclass
class Run:
    """The run every hook shares. Set `stop` to `"cancelled"` or `"timed_out"` to end it after this batch.

    `total` is the record count; `status` is set just before `on_run_end`.
    """

    name: str
    step: Any
    scenarios: tuple[str, ...]
    total: int
    batch_size: int
    stop: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class Batch:
    """One scenario's output for one batch, and the baseline's output for the same records.

    `offset` is the index of the batch's first record in the whole input.
    """

    scenario: str
    index: int
    offset: int
    output: pl.DataFrame
    baseline: pl.DataFrame


@dataclass(frozen=True)
class NodeBatch:
    """What one node wrote in one batch, on the records it ran on.

    `values` has a column per name the node writes (none for a branch, loop or
    sequence); `rows` are the batch positions it ran on, `None` meaning all of
    them. `baseline` is the baseline's `NodeBatch` for the same node and
    iteration, `None` when the baseline never ran it on this batch.
    """

    scenario: str
    path: str
    index: int
    offset: int
    values: pl.DataFrame
    rows: np.ndarray | None
    arm: int | None
    iteration: int | None
    baseline: NodeBatch | None = None


class Callback:
    """Hooks the runner calls; override the ones you need.

    `on_node_end` fires only for a callback with a `nodes` attribute, a glob on
    node paths (`"*"`, `"term/*"`); asking for nodes runs the flow one kernel per
    node instead of fused, which is still compiled.

    Example::

        class Rows(Callback):
            def on_batch_end(self, batch):
                print(batch.scenario, batch.offset + batch.output.height)
    """

    def on_run_start(self, run: Run) -> None:
        pass

    def on_node_end(self, node: NodeBatch) -> None:
        pass

    def on_batch_end(self, batch: Batch) -> None:
        pass

    def on_progress(self, run: Run, done: int) -> None:
        """After every scenario has run the batch ending at record `done`."""

    def on_run_end(self, run: Run) -> None:
        pass


def overrides(callback: Any, hook: str) -> bool:
    """Whether `callback`'s class overrides `hook`, so the runner skips work nobody listens to."""
    return getattr(type(callback), hook, None) is not getattr(Callback, hook)


class Progress(Callback):
    """Prints records done, the total and the elapsed time after each batch."""

    def on_run_start(self, run: Run) -> None:
        self._start = time.perf_counter()

    def on_progress(self, run: Run, done: int) -> None:
        print(f"{run.name}: {done:,}/{run.total:,} records, {time.perf_counter() - self._start:.1f}s")


class JobProgress(Callback):
    """Reports progress to a `JobHandle` and stops the run when the job is cancelled or times out."""

    def __init__(self, job: JobHandle):
        self.job = job

    def on_run_start(self, run: Run) -> None:
        self.job.start()
        self.job.progress(0, total=run.total)

    def on_progress(self, run: Run, done: int) -> None:
        self.job.progress(done, total=run.total)
        if self.job.timed_out:
            self.job.timeout()
            run.stop = "timed_out"
        elif self.job.cancelled:
            run.stop = "cancelled"

    def on_run_end(self, run: Run) -> None:
        if not self.job.terminal:
            self.job.succeed(result={"status": run.status})


class WriteOutputs(Callback):
    """Writes each scenario's output to `<out_dir>/<scenario>/<batch>.parquet`.

    Read one back with `pl.scan_parquet(f"{out_dir}/baseline/*.parquet")`.
    """

    def __init__(self, out_dir: str | Path):
        self.root = Path(out_dir)

    def on_batch_end(self, batch: Batch) -> None:
        folder = self.root / batch.scenario
        folder.mkdir(parents=True, exist_ok=True)
        batch.output.write_parquet(folder / f"{batch.index:06d}.parquet")
