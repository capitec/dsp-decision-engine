"""Getting a compiled kernel's results back into polars — doc 05 §3.1.

**Dtype-grouped 2D arrays, never a record array.** Record write-back is the
dominant batch cost (64.0% of total at 633 outputs, EXPERIMENTS.md §J) and is
beaten on both axes by three grouped arrays — one `float64`, one `int64`, one
`bool` — so arity stays at three regardless of how many columns a pipeline
produces:

    | convention                | write-back @100k | compile  |
    |----------------------------|-------------------|----------|
    | record -> record           | 775.6 ms           | 11.31 s  |
    | record -> column-major 2D  | **5.7 ms**          | 0.27 s   |
    | record -> row-major 2D     | 505.8 ms            | 0.25 s   |

**Layout is chosen by entry point** (doc 05 §3.1's callout), not globally:

    * `apply()` (batch) -> **column-major**, shape `(n_cols, n_rows)`. Row `i`
      of the array (fixed column, varying row) is one whole output column,
      contiguous in memory — handed to polars directly, which is where the
      measured zero-copy comes from (EXPERIMENTS.md §J: "all True").
    * `score()` (single record) -> **row-major**, shape `(n_rows, n_cols)`.
      Row `i` (fixed record, varying column) is one whole record, contiguous
      — there is no bulk write-back to amortise here (doc 05 §3.1b), so
      kernel locality while *writing* the row is what matters, and row-major
      has the fastest per-record kernel of the three (800.8 ns vs 3497.1 ns
      for column-major at N=1).

The output frame's own contract (doc 03 §7) is additive: inputs, plus
terminals, plus whatever `.emit()`s, minus whatever `.drop()`s. Deciding
*which names* belong in that set is the graph layer's job (doc 03 §5.1's
inferred interface); this module's job is only to assemble, efficiently,
whatever set it is handed — via `hstack`, never chained `with_columns`
(~24us fixed + ~10us/col, independent of row count — doc 05 §3.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import polars as pl

__all__ = [
    "Layout",
    "DtypeGroup",
    "KernelOutputs",
    "to_series",
    "write_back",
    "row_to_dict",
    "resolve_kept_input_columns",
]


class Layout(Enum):
    COLUMN_MAJOR = "column_major"   # shape (n_cols, n_rows) — apply()
    ROW_MAJOR = "row_major"          # shape (n_rows, n_cols) — score()


@dataclass(frozen=True)
class DtypeGroup:
    """One of the (at most three) arrays a kernel writes its outputs into."""

    names: tuple[str, ...]
    array: np.ndarray
    layout: Layout

    def __post_init__(self) -> None:
        if self.array.ndim != 2:
            raise ValueError(f"DtypeGroup array must be 2D, got shape {self.array.shape}")
        cols_axis = 0 if self.layout is Layout.COLUMN_MAJOR else 1
        if self.array.shape[cols_axis] != len(self.names):
            raise ValueError(
                f"DtypeGroup has {len(self.names)} names but array shape "
                f"{self.array.shape} does not match on axis {cols_axis} ({self.layout.value})"
            )


@dataclass(frozen=True)
class KernelOutputs:
    """The three-argument arity doc 05 §3.1 fixes regardless of output
    count: at most one group per dtype family.
    """

    float64: DtypeGroup | None = None
    int64: DtypeGroup | None = None
    bool_: DtypeGroup | None = None

    def groups(self) -> tuple[DtypeGroup, ...]:
        return tuple(g for g in (self.float64, self.int64, self.bool_) if g is not None)


def to_series(group: DtypeGroup) -> list[pl.Series]:
    """One `pl.Series` per named column.

    Column-major: `array[i, :]` is already a contiguous 1D slice — handed to
    `pl.Series` directly, which is the zero-copy path EXPERIMENTS.md §J
    measured (5.7 ms vs 775.6 ms for the record equivalent at 100k rows).
    Row-major has no batch write-back to amortise (§3.1b's whole point); this
    still produces correct series for it, just without that property, via an
    explicit `ascontiguousarray` on what would otherwise be a strided view.
    """
    series: list[pl.Series] = []
    if group.layout is Layout.COLUMN_MAJOR:
        for i, name in enumerate(group.names):
            series.append(pl.Series(name, group.array[i, :]))
    else:
        for i, name in enumerate(group.names):
            series.append(pl.Series(name, np.ascontiguousarray(group.array[:, i])))
    return series


def resolve_kept_input_columns(
    frame_columns: Sequence[str],
    *,
    overwritten: Sequence[str] = (),
    dropped: Sequence[str] = (),
) -> tuple[str, ...]:
    """The frame-column half of doc 03 §7's additive rule: an input column
    rides through untouched unless it is `.drop()`-ed, or a kernel output
    overwrites it by name (doc 03 §7, "a value that gets rewritten" — the
    waterfall means the *kernel's* value wins, not the frame's original one).
    Which names are overwritten is a graph-layer fact (doc 03 §3.2's
    waterfall); this only applies the two exclusions once they're known.
    """
    excluded = set(dropped) | set(overwritten)
    return tuple(c for c in frame_columns if c not in excluded)


def write_back(
    frame: pl.DataFrame,
    outputs: KernelOutputs,
    *,
    keep: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Assemble the additive output frame (doc 03 §7).

    `keep` is the already-resolved list of original `frame` columns to carry
    through (e.g. from `resolve_kept_input_columns`); `None` keeps every
    column in `frame` as-is. Every `outputs` group is appended via `hstack`
    in one call — never chained `with_columns` (doc 05 §3.2).

    An intermediate nothing reads and nothing emits was never given to this
    function in the first place (doc 03 §7: "never materialised") — that
    decision happens upstream, in the graph/compile layers that decide what
    a kernel computes at all.
    """
    base = frame.select(keep) if keep is not None else frame
    computed: list[pl.Series] = []
    for group in outputs.groups():
        computed.extend(to_series(group))
    if not computed:
        return base
    return base.hstack(computed)


def row_to_dict(outputs: KernelOutputs) -> dict[str, object]:
    """The single-record path (doc 05 §3.1b): each group holds exactly one
    row. One `tolist()` per group (at most three calls total, never one per
    field) rather than a Python loop assigning hundreds of dict entries one
    at a time — the per-field loop is what §N1 measured as 92% of a
    single-record request's cost.
    """
    result: dict[str, object] = {}
    for group in outputs.groups():
        if group.layout is not Layout.ROW_MAJOR:
            raise ValueError("row_to_dict expects row-major groups (score() path, doc 05 §3.1b)")
        if group.array.shape[0] != 1:
            raise ValueError(f"row_to_dict expects exactly one row, got {group.array.shape[0]}")
        row = group.array[0, :]
        result.update(zip(group.names, row.tolist()))
    return result
