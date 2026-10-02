from __future__ import annotations

import typing as t
from collections import Counter

import polars as pl

from decider.experiments.callbacks import Batch, NodeBatch
from decider.experiments.metrics.base import Metric, Summary


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
        if self.nodes is None:
            self._add(batch.output[self.column])

    def on_node_end(self, node: NodeBatch) -> None:
        if self.column in node.values.columns:
            self._add(node.values[self.column])

    def _add(self, values: pl.Series) -> None:
        raise NotImplementedError


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
