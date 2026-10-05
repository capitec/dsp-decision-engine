from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, PrivateAttr

_SAMPLES = 5


class ColumnDiff(BaseModel):
    """How one column differs. Deltas are `new - old` over the changed rows, left out where non-finite or null.

    `max_abs` is an exact int for integer columns (money in cents).
    """

    changed: int
    max_abs: int | float | None = None
    max_rel: float | None = None
    mean_delta: float | None = None
    dtype_change: tuple[str, str] | None = None
    samples: list[dict[str, Any]] = []


class OutputDiff(BaseModel):
    """Which rows and columns differ between two versions' outputs over one frame.

    Example::

        o = diff.outputs(old, new, frame, key="application_id")
        o.columns["offer"].max_abs
        o.table()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows_compared: int
    rows_changed: int
    columns: dict[str, ColumnDiff]
    added_columns: list[str]
    removed_columns: list[str]
    _table: pl.DataFrame = PrivateAttr()

    def table(self) -> pl.DataFrame:
        """One row per changed value: the key (or row number), column, old, new and numeric delta."""
        return self._table


def changed(old: pl.Series, new: pl.Series, rtol: float = 1e-6, atol: float = 1e-9) -> pl.Series:
    """A bool series, True where `new` differs from `old`.

    Floats compare with `is_close`, relative to `old` (NaN equals NaN); everything else exactly; null equals only null.
    """
    if old.dtype.is_float() and new.dtype.is_float():
        same = new.is_close(old, abs_tol=atol, rel_tol=rtol, nans_equal=True)
    else:
        same = old.eq(new)
    return ~same.fill_null(old.is_null() & new.is_null())


def outputs(old: Any, new: Any, frame: Any, params: Any = None, key: str | None = None,
            rtol: float = 1e-6, atol: float = 1e-9) -> OutputDiff:
    """Run both pipelines over `frame` and report which rows and columns change.

    `params` goes to both; leave it out and each runs with its own defaults.
    `key` names a column that labels rows in samples and `table()`.

    Example::

        diff.outputs(old, new, frame, key="application_id").rows_changed
    """
    frame = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    runs = [step.compile("interpreted").run(frame, params) for step in (old, new)]
    return compare_frames(*runs, key=key, rtol=rtol, atol=atol)


def compare_frames(old: pl.DataFrame, new: pl.DataFrame, key: str | None = None,
                   rtol: float = 1e-6, atol: float = 1e-9) -> OutputDiff:
    if old.height != new.height:
        raise ValueError(f"the versions produced {old.height} and {new.height} rows from one frame; can't align them")
    label = key or "row"
    if key is not None and key not in old.columns:
        raise ValueError(f"key {key!r} is not a column of the output; columns: {old.columns}")
    labels = old[key] if key else pl.Series(label, range(old.height))
    shared = [c for c in old.columns if c in new.columns]
    columns: dict[str, ColumnDiff] = {}
    any_changed = pl.Series([False] * old.height)
    tables = []
    for name in shared:
        mask = _mask(old[name], new[name], rtol, atol)
        any_changed = any_changed | mask
        dtype_change = (str(old[name].dtype), str(new[name].dtype)) if old[name].dtype != new[name].dtype else None
        if not mask.any() and dtype_change is None:
            continue
        o, n = old[name].filter(mask), new[name].filter(mask)
        delta = _delta(o, n)
        columns[name] = ColumnDiff(
            changed=int(mask.sum()), dtype_change=dtype_change, samples=_samples(labels.filter(mask), o, n),
            **_stats(o, delta))
        tables.append(pl.DataFrame({label: labels.filter(mask), "column": name, "old": o.cast(pl.String),
                                    "new": n.cast(pl.String), "delta": _as_float(delta, o.len())}))
    out = OutputDiff(rows_compared=old.height, rows_changed=int(any_changed.sum()), columns=columns,
                     added_columns=[c for c in new.columns if c not in old.columns],
                     removed_columns=[c for c in old.columns if c not in new.columns])
    out._table = pl.concat(tables) if tables else pl.DataFrame(
        schema={label: labels.dtype, "column": pl.String, "old": pl.String, "new": pl.String, "delta": pl.Float64})
    return out


def _as_float(delta: pl.Series | None, n: int) -> pl.Series:
    return pl.Series("delta", [None] * n, dtype=pl.Float64) if delta is None else delta.cast(pl.Float64)


def _mask(old: pl.Series, new: pl.Series, rtol: float, atol: float) -> pl.Series:
    try:
        return changed(old, new, rtol, atol)
    except pl.exceptions.PolarsError:  # types that can't compare, such as a number against a string
        return pl.Series([True] * old.len())


def _delta(old: pl.Series, new: pl.Series) -> pl.Series | None:
    return new - old if old.dtype.is_numeric() and new.dtype.is_numeric() else None


def _stats(old: pl.Series, delta: pl.Series | None) -> dict[str, Any]:
    if delta is None:
        return {}
    ok = delta.is_not_null() & (delta.is_finite() if delta.dtype.is_float() else True)
    d, base = delta.filter(ok), old.filter(ok)
    if d.is_empty():
        return {}
    rel = (d.abs() / base.abs()).filter(base != 0)
    return {"max_abs": d.abs().max(), "mean_delta": float(d.mean()),
            "max_rel": float(rel.max()) if not rel.is_empty() else None}


def _samples(labels: pl.Series, old: pl.Series, new: pl.Series) -> list[dict[str, Any]]:
    return [{labels.name: k, "old": a, "new": b}
            for k, a, b in zip(labels.head(_SAMPLES), old.head(_SAMPLES), new.head(_SAMPLES))]
