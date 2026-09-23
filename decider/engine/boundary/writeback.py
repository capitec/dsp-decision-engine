"""Kernel outputs back into polars: at most three dtype-grouped 2D arrays, never a record array."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import polars as pl


class Layout(Enum):
    """How a `DtypeGroup` array is laid out.

    Batch calls write column-major `(n_cols, n_rows)` so each output column is
    contiguous and reaches polars without a copy. Single-record calls write
    row-major `(n_rows, n_cols)`, which is faster for the kernel to fill.
    """

    COLUMN_MAJOR = "column_major"
    ROW_MAJOR = "row_major"


@dataclass(frozen=True)
class DtypeGroup:
    """One 2D output array and the names of its columns.

    Example::

        DtypeGroup(("score",), np.array([[0.5, 0.9]]), Layout.COLUMN_MAJOR)
    """

    names: tuple[str, ...]
    array: np.ndarray
    layout: Layout

    def __post_init__(self) -> None:
        if self.array.ndim != 2:
            raise ValueError(f"DtypeGroup array must be 2D, got shape {self.array.shape}")
        axis = 0 if self.layout is Layout.COLUMN_MAJOR else 1
        if self.array.shape[axis] != len(self.names):
            raise ValueError(
                f"DtypeGroup has {len(self.names)} names but array shape "
                f"{self.array.shape} does not match on axis {axis} ({self.layout.value})"
            )


@dataclass(frozen=True)
class KernelOutputs:
    """A kernel's outputs: at most one group per dtype family, whatever the output count."""

    float64: DtypeGroup | None = None
    int64: DtypeGroup | None = None
    bool_: DtypeGroup | None = None

    def groups(self) -> tuple[DtypeGroup, ...]:
        return tuple(g for g in (self.float64, self.int64, self.bool_) if g is not None)


def to_series(group: DtypeGroup) -> list[pl.Series]:
    """One `pl.Series` per named column; zero-copy for a column-major group."""
    if group.layout is Layout.COLUMN_MAJOR:
        return [pl.Series(name, group.array[i, :]) for i, name in enumerate(group.names)]
    return [pl.Series(name, np.ascontiguousarray(group.array[:, i])) for i, name in enumerate(group.names)]


def resolve_kept_input_columns(
    frame_columns: Sequence[str], *, overwritten: Sequence[str] = (), dropped: Sequence[str] = (),
) -> tuple[str, ...]:
    """The input columns that ride through to the output: all but the dropped and the overwritten.

    >>> resolve_kept_input_columns(["id", "term", "tmp"], overwritten=["term"], dropped=["tmp"])
    ('id',)
    """
    excluded = set(dropped) | set(overwritten)
    return tuple(c for c in frame_columns if c not in excluded)


def write_back(frame: pl.DataFrame, outputs: KernelOutputs, *, keep: Sequence[str] | None = None) -> pl.DataFrame:
    """The output frame: `frame`'s `keep` columns (all by default) plus every output column, in one `hstack`.

    Example::

        write_back(frame, KernelOutputs(float64=DtypeGroup(("score",), out.reshape(1, -1), Layout.COLUMN_MAJOR)))
    """
    base = frame.select(keep) if keep is not None else frame
    # One hstack: chained with_columns costs ~24 us fixed plus ~10 us a column.
    computed = [s for group in outputs.groups() for s in to_series(group)]
    return base.hstack(computed) if computed else base


def row_to_dict(outputs: KernelOutputs) -> dict[str, object]:
    """A single record's outputs as a dict of Python values, from row-major one-row groups.

    One `tolist()` per group rather than one per field, which dominates a single-record call.
    """
    result: dict[str, object] = {}
    for group in outputs.groups():
        if group.layout is not Layout.ROW_MAJOR:
            raise ValueError("row_to_dict expects row-major groups")
        if group.array.shape[0] != 1:
            raise ValueError(f"row_to_dict expects exactly one row, got {group.array.shape[0]}")
        result.update(zip(group.names, group.array[0, :].tolist()))
    return result
