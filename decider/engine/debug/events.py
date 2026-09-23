from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union

import polars as pl

from decider.engine.ir.origin import Origin

PREVIEW_ROWS = 5


@dataclass(frozen=True, slots=True)
class Summary:
    """What an event says about a column: never the full values, which may be personal data.

    Example::

        summarize(pl.Series([1.0, None, 3.0]))
        # Summary(dtype="Float64", rows=3, nulls=1, preview=(1.0, None, 3.0))
    """

    dtype: str
    rows: int
    nulls: int
    preview: tuple[Any, ...]


def summarize(series: pl.Series) -> Summary:
    """A `Summary` of `series`: dtype, row and null counts, the first few values."""
    return Summary(str(series.dtype), series.len(), series.null_count(), tuple(series.head(PREVIEW_ROWS).to_list()))


@dataclass(frozen=True, slots=True)
class RunStarted:
    """A session opened over `rows` rows; nothing has run yet."""

    rows: int
    kind: Literal["run_started"] = "run_started"


@dataclass(frozen=True, slots=True)
class NodeStarted:
    """A node is about to run."""

    origin: Origin
    kind: Literal["node_started"] = "node_started"


@dataclass(frozen=True, slots=True)
class NodeFinished:
    """A node ran; `outputs` summarises every value it produced, by name."""

    origin: Origin
    outputs: dict[str, Summary]
    kind: Literal["node_finished"] = "node_finished"


@dataclass(frozen=True, slots=True)
class NodeVisited:
    """A position inside a row node (`origin.locator`) was reached on `rows` rows."""

    origin: Origin
    rows: int
    kind: Literal["node_visited"] = "node_visited"


@dataclass(frozen=True, slots=True)
class Paused:
    """The session stopped at a checkpoint; `reason` is `"breakpoint"`, `"step"`, `"pause"` or `"rewind"`.

    `kernel` lists the steps a fused kernel runs when the checkpoint is one
    (the first is `origin`); a breakpoint on any of them pauses here.
    """

    origin: Origin
    when: Literal["before", "after"]
    reason: str
    kernel: tuple[str, ...] = ()
    kind: Literal["paused"] = "paused"


@dataclass(frozen=True, slots=True)
class Overridden:
    """`name` was set; `producer` is the recorded version's producer, `override@<path>`."""

    name: str
    producer: str
    value: Summary
    previous: Summary
    kind: Literal["overridden"] = "overridden"


@dataclass(frozen=True, slots=True)
class ParamsValidated:
    """Params of the nodes at `paths` were validated; `invalid` lists those that failed."""

    paths: tuple[str, ...]
    invalid: tuple[str, ...]
    kind: Literal["params_validated"] = "params_validated"


@dataclass(frozen=True, slots=True)
class Warning:
    """A params value was replaced by its default."""

    message: str
    kind: Literal["warning"] = "warning"


@dataclass(frozen=True, slots=True)
class Error:
    """The run raised while in the node at `path`; the session can `rewind` but not go on."""

    message: str
    path: str | None
    kind: Literal["error"] = "error"


@dataclass(frozen=True, slots=True)
class RunFinished:
    """The run reached the end; `output` summarises every column of `session.output()`."""

    output: dict[str, Summary]
    kind: Literal["run_finished"] = "run_finished"


Event = Union[RunStarted, NodeStarted, NodeFinished, NodeVisited, Paused, Overridden,
              ParamsValidated, Warning, Error, RunFinished]
