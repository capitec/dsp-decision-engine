from __future__ import annotations

import typing as t
from abc import abstractmethod
from collections import Counter

import numpy as np
import polars as pl

from decider.lab.experiments.callbacks import Batch, NodeBatch
from decider.lab.experiments.metrics._base import Metric, Summary


class Counts(Summary):
    column: str
    counts: dict[str, int]


class Stats(Summary):
    column: str
    count: int
    nulls: int
    mean: float | None
    min: float | None
    max: float | None


class _ColumnMetric(Metric):
    # Reads `column` from the flow's output, or from what the selected nodes wrote when `nodes` is set.
    column: str

    def on_batch_end(self, batch: Batch) -> None:
        if not self.on_steps:
            self._add(batch.output[self.column])

    def on_node_end(self, node: NodeBatch) -> None:
        if self.column in node.values.columns:
            self._add(node.values[self.column])

    @abstractmethod
    def _add(self, values: pl.Series) -> None: ...


class ValueCounts(_ColumnMetric):
    """How many records landed on each value of `column`.

    Example::

        ValueCounts(column="approved")
    """

    type: t.Literal["value_counts"] = "value_counts"
    summary: t.ClassVar = Counts

    def reset(self) -> None:
        self._counts: Counter[str] = Counter()

    def _add(self, values: pl.Series) -> None:
        for value, n in values.value_counts().iter_rows():
            self._counts[str(value)] += n

    def summarise(self) -> Counts:
        return Counts(column=self.column, counts=dict(self._counts))


class Describe(_ColumnMetric):
    """Count, nulls, mean, min and max of a numeric `column`, accumulated batch by batch.

    Example::

        Describe(column="tier")
    """

    type: t.Literal["describe"] = "describe"
    summary: t.ClassVar = Stats

    def reset(self) -> None:
        self._count = self._nulls = 0
        self._total = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def _add(self, values: pl.Series) -> None:
        nulls = values.null_count()
        self._nulls += nulls
        if values.len() == nulls:
            return
        self._count += values.len() - nulls
        self._total += float(values.sum())
        lo, hi = float(values.min()), float(values.max())
        self._min = lo if self._min is None else min(self._min, lo)
        self._max = hi if self._max is None else max(self._max, hi)

    def summarise(self) -> Stats:
        mean = self._total / self._count if self._count else None
        return Stats(column=self.column, count=self._count, nulls=self._nulls, mean=mean,
                     min=self._min, max=self._max)


class Bins(Summary):
    """Records per bin of `column`: `counts[i]` lies in `[edges[i], edges[i + 1])`, the last bin closed.

    `below`, `above` and `nulls` count what fell outside the bins.
    """

    column: str
    edges: list[float]
    counts: list[int]
    below: int
    above: int
    nulls: int


class Histogram(_ColumnMetric):
    """How `column`'s values spread over bins, accumulated batch by batch.

    `edges` fixes the bins, or `range` spans `bins` equal-width ones; without
    either they span the baseline's first batch, so every scenario counts into
    the same bins. Give a `range` when a scenario moves values past the
    baseline's, or they only count as `above`/`below`.

    Example::

        Histogram(column="rate", nodes="loans/pricing/*", bins=30, range=(0.0, 0.4))
    """

    type: t.Literal["histogram"] = "histogram"
    summary: t.ClassVar = Bins
    bins: int = 20
    range: tuple[float, float] | None = None
    edges: list[float] | None = None

    def reset(self) -> None:
        if self.edges is not None:
            self._edges = np.asarray(self.edges, float)
        else:
            self._edges = None if self.range is None else np.linspace(*self.range, self.bins + 1)
        self._counts = None
        self._below = self._above = self._nulls = 0

    def on_batch_end(self, batch: Batch) -> None:
        if not self.on_steps:
            self._ensure_edges(batch.baseline[self.column])
            self._add(batch.output[self.column])

    def on_node_end(self, node: NodeBatch) -> None:
        if self.column in node.values.columns:
            # ponytail: a scenario reaching a node the baseline's first batch skipped bins on its own values.
            reference = node.baseline.values if node.baseline is not None else node.values
            self._ensure_edges(reference[self.column])
            self._add(node.values[self.column])

    def _ensure_edges(self, reference: pl.Series) -> None:
        if self._edges is not None:
            return
        values = _numbers(reference)
        lo, hi = (float(values.min()), float(values.max())) if values.size else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        self._edges = np.linspace(lo, hi, self.bins + 1)

    def _add(self, values: pl.Series) -> None:
        numbers = _numbers(values)
        self._nulls += values.len() - numbers.size
        lo, hi = self._edges[0], self._edges[-1]
        self._below += int((numbers < lo).sum())
        self._above += int((numbers > hi).sum())
        counts = np.histogram(numbers, self._edges)[0]
        self._counts = counts if self._counts is None else self._counts + counts

    def summarise(self) -> Bins:
        edges = [] if self._edges is None else self._edges.tolist()
        counts = [0] * max(len(edges) - 1, 0) if self._counts is None else self._counts.tolist()
        return Bins(column=self.column, edges=edges, counts=counts, below=self._below, above=self._above,
                    nulls=self._nulls)


def _numbers(values: pl.Series) -> np.ndarray:
    # Nulls and NaN have no bin.
    numbers = values.drop_nulls().cast(pl.Float64).to_numpy()
    return numbers[~np.isnan(numbers)]
