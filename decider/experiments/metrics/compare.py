from __future__ import annotations

import typing as t

import numpy as np
import polars as pl

from decider.experiments.callbacks import Batch, NodeBatch
from decider.experiments.metrics.base import Metric, Summary

_ROW = "__row"


class Divergences(Summary):
    """`first` is the index of the first changed record; `samples` hold its row, expected and actual values."""

    column: str
    compared: int
    changed: int
    first: int | None
    samples: list[dict[str, t.Any]]


class Divergence(Metric):
    """How many records' `column` differs from the baseline's.

    Floats compare with polars `is_close` (`rtol`/`atol`, NaN equals NaN),
    everything else exactly; null equals only null. With `nodes`, it compares
    what the matching nodes wrote, on the records both runs sent through them.

    Example::

        Divergence(column="tier", samples=10)
    """

    type: t.Literal["divergence"] = "divergence"
    summary: t.ClassVar = Divergences
    column: str
    rtol: float = 1e-6
    atol: float = 1e-9
    samples: int = 5

    def reset(self) -> None:
        self._compared = self._changed = 0
        self._first: int | None = None
        self._samples: list[dict[str, t.Any]] = []

    def on_batch_end(self, batch: Batch) -> None:
        if self.nodes is None:
            rows = pl.Series(_ROW, np.arange(batch.offset, batch.offset + batch.output.height))
            self._compare(batch.output[self.column], batch.baseline[self.column], rows)

    def on_node_end(self, node: NodeBatch) -> None:
        if node.baseline is None or self.column not in node.values.columns:
            return
        both = self._positioned(node).join(self._positioned(node.baseline), on=_ROW, suffix="_baseline")
        self._compare(both[self.column], both[f"{self.column}_baseline"], both[_ROW])

    def summarise(self) -> Divergences:
        return Divergences(column=self.column, compared=self._compared, changed=self._changed,
                           first=self._first, samples=self._samples)

    def _positioned(self, node: NodeBatch) -> pl.DataFrame:
        rows = np.arange(node.values.height) if node.rows is None else node.rows
        return node.values.select(self.column).with_columns(pl.Series(_ROW, rows + node.offset))

    def _compare(self, actual: pl.Series, expected: pl.Series, rows: pl.Series) -> None:
        changed = _changed(actual, expected, self.rtol, self.atol)
        n = int(changed.sum())
        self._compared += changed.len()
        self._changed += n
        if not n:
            return
        if self._first is None:
            self._first = int(rows.filter(changed)[0])
        room = self.samples - len(self._samples)
        if room > 0:
            hits = pl.DataFrame({"row": rows, "expected": expected, "actual": actual}).filter(changed)
            self._samples += hits.head(room).to_dicts()


def _changed(actual: pl.Series, expected: pl.Series, rtol: float, atol: float) -> pl.Series:
    if actual.dtype.is_float() and expected.dtype.is_float():
        same = actual.is_close(expected, abs_tol=atol, rel_tol=rtol, nans_equal=True)
    else:
        same = actual.eq(expected)
    return ~same.fill_null(actual.is_null() & expected.is_null())
