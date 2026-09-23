from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import polars as pl

from decider.engine.ir.decls import TYPED, Input, NullPolicy, base_annotation
from decider.engine.wiring.plan import Plan, Version


class State:
    """Every value of one run: a numpy array per `Version.id`, a validity mask for nullable ones.

    Args:
        values: `Version.id` -> array of `n` values.
        valid: `Version.id` -> bool mask of `n` rows; absent means every row is valid.
        frame: the frame unread columns pass through from: the input frame,
            replaced by what an unknown-lineage frame step returns.

    Example::

        state = executable.prepare(df)[0]
        state.column("term_cap@term/cap_by_income")   # a polars Series
        state.versions("term_cap@*")                  # every version, in production order
    """

    def __init__(self, plan: Plan, frame: pl.DataFrame, n: int | None = None):
        self.plan = plan
        self.frame = frame
        self.n = frame.height if n is None else n
        self.values: dict[int, np.ndarray] = {}
        self.valid: dict[int, np.ndarray] = {}
        self.chains: dict[str, list[Version]] = {k: list(v) for k, v in plan.chains.items()}
        self._extra = 0
        self._sources: dict[int, tuple[np.ndarray, pl.Series]] = {}
        # Input columns read in place from Arrow: `Version.id` -> `(validity bitmap, bit offset, null count)`
        # or None. Their `values` may be a `(bitmap, bit offset)` pair (a bool column). `read` unpacks both.
        self.packed: dict[int, tuple | None] = {}

    @classmethod
    def from_frame(cls, plan: Plan, frame: pl.DataFrame, n: int | None = None) -> State:
        """A state holding the plan's input columns read from `frame`; absent columns are all null."""
        from decider.engine.boundary.extract import extract_frame

        state = cls(plan, frame, n)
        names = frame.columns
        typed = [i for i in _typed(plan) if i.name in names]
        extracted = extract_frame(frame, typed).columns if typed else {}
        for v in plan.versions:
            if v.producer is not None:
                continue
            col = extracted.get(v.name)
            if col is not None and (col.bits is not None or isinstance(col.values, tuple)):
                state.values[v.id] = col.values
                state.packed[v.id] = col.bits
            elif col is not None:
                state.write(v, col.values, valid=col.validity if col.has_nulls else None)
            elif v.name in names:
                # ponytail: a string column becomes Python objects even when only kernels read it
                # (as spans, from `source`); convert on first Python read if big string batches matter.
                series = frame.get_column(v.name)
                values, valid = from_series(series)
                state.write(v, values, valid=valid)
                state._sources[v.id] = (values, series)
        return state

    def write(self, version: Version, values: np.ndarray, rows: np.ndarray | None = None,
              valid: np.ndarray | None = None) -> None:
        """Store `values` for `version` on `rows`; `valid` marks which of them are non-null.

        With `rows=None` the array is stored as is (not copied). Rows a write
        never touches stay null.
        """
        vid = version.id
        if vid in self.packed:
            if rows is None:
                del self.packed[vid]
            else:
                self.unpack(vid)
        if rows is None:
            self.values[vid] = values
            if valid is not None and not valid.all():
                self.valid[vid] = valid
            else:
                self.valid.pop(vid, None)
            return
        target = self.values.get(vid)
        if target is None:
            target = self.values[vid] = np.full(self.n, None, object) if values.dtype == object else np.zeros(self.n, values.dtype)
            self.valid[vid] = np.zeros(self.n, bool)
        elif values.dtype == object and target.dtype != object:
            target = self.values[vid] = target.astype(object)
        target[rows] = values
        mask = self.valid.get(vid)
        if mask is None:
            if valid is None or valid.all():
                return
            mask = self.valid[vid] = np.ones(self.n, bool)
        mask[rows] = True if valid is None else valid

    def read(self, version: Version, rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """`version`'s values and validity mask (`None` = all valid) on `rows`; never written means all null."""
        if version.id in self.packed:
            self.unpack(version.id)
        values, valid = self.values.get(version.id), self.valid.get(version.id)
        if values is None:
            values, valid = np.full(self.n, None, object), np.zeros(self.n, bool)
        if rows is None:
            return values, valid
        return values[rows], None if valid is None else valid[rows]

    def unpack(self, vid: int) -> None:
        """Turn an in-place Arrow column's bitmaps into the bool arrays everything but a kernel reads."""
        bits = self.packed.pop(vid)
        x = self.values[vid]
        if isinstance(x, tuple):
            self.values[vid] = _unbits(*x, self.n)
        if bits is not None:
            self.valid[vid] = _unbits(bits[0], bits[1], self.n)

    def source(self, version: Version) -> pl.Series | None:
        """The input frame column `version` was read from, while its values are still that column's.

        Lets a compiled step read an object column's Arrow buffers in place.

        Example::

            state.source(state.versions("channel")[0])   # the frame's "channel" Series
        """
        values, series = self._sources.get(version.id, (None, None))
        return series if values is not None and self.values.get(version.id) is values else None

    def record(self, name: str, producer: str, values: np.ndarray, valid: np.ndarray | None = None) -> Version:
        """Store `values` as a new version of `name`, appended to its chain; for overrides.

        Example::

            state.record("disposable_income", "override@term/cap_by_income", np.full(state.n, 1200.0))
        """
        self._extra += 1
        v = Version(len(self.plan.versions) + self._extra - 1, name, producer)
        self.chains.setdefault(name, []).append(v)
        self.write(v, values, valid=valid)
        return v

    def restore(self, old: State, keep: set[int]) -> None:
        """Take back from `old` the values of the versions in `keep`, every override it recorded, and its chains.

        For re-running part of a run: a fresh state replays up to a node, then
        restores what the earlier run had upstream of it.

        Example::

            fresh.restore(state, {v.id for v in plan.versions if v.producer is None})
        """
        for vid, values in old.values.items():
            if vid in keep or vid >= len(self.plan.versions):
                if vid in old.packed:
                    old.unpack(vid)
                    values = old.values[vid]
                self.values[vid] = values
                if vid in old.valid:
                    self.valid[vid] = old.valid[vid]
                else:
                    self.valid.pop(vid, None)
        self.chains, self._extra = old.chains, old._extra

    def versions(self, spec: str) -> list[Version]:
        """The versions `spec` names: `name` (the latest), `name@path` (by producer) or `name@*` (all).

        Example::

            [v.producer for v in state.versions("term_cap@*")]
        """
        name, at, producer = spec.partition("@")
        chain = self.chains.get(name, [])
        if producer == "*":
            return list(chain)
        if at:
            return [v for v in chain if v.producer == producer][-1:]
        if chain:
            return chain[-1:]
        return [v for v in self.plan.versions if v.producer is None and v.name == name]

    def column(self, spec: str, version: Version | None = None) -> pl.Series:
        """One version as a polars Series named `spec`, null where invalid."""
        (v,) = [version] if version is not None else self.versions(spec)
        return _series(spec, *self.read(v))

    def frame_of(self, base: pl.DataFrame, names: dict[str, Version], rows: np.ndarray | None) -> pl.DataFrame:
        """`base` with the current value of every name in `names` on `rows`, for a frame step."""
        cols = [_series(name, *self.read(v, rows)) for name, v in names.items()]
        return base.with_columns(cols) if cols else base


# Keyed by the plan's identity; a fill value may not be hashable.
@lru_cache(maxsize=64)
def _typed(plan: Plan) -> tuple[Input, ...]:
    # The inputs the boundary reads into typed arrays, nulls kept as a mask.
    return tuple(Input(i.name, t, NullPolicy.OPTIONAL) for i in plan.inputs if (t := base_annotation(i.annotation)) in TYPED)


def _unbits(bitmap: np.ndarray, offset: int, n: int) -> np.ndarray:
    return np.unpackbits(bitmap, count=offset + n, bitorder="little")[offset:].view(np.bool_)


def _series(name: str, values: np.ndarray, valid: np.ndarray | None) -> pl.Series:
    if values.dtype == object:
        return pl.Series(name, (values if valid is None else np.where(valid, values, None)).tolist())
    s = pl.Series(name, values)
    return s if valid is None else s.scatter(np.flatnonzero(~valid), None)


def from_series(s: pl.Series) -> tuple[np.ndarray, np.ndarray | None]:
    """A polars Series as `(values, valid)`; nulls hold a zero (or `None` for objects) and are masked."""
    if not s.null_count():
        return s.to_numpy(), None
    valid = s.is_not_null().to_numpy()
    if s.dtype.is_numeric():
        s = s.fill_null(0)
    elif s.dtype == pl.Boolean:
        s = s.fill_null(False)
    return s.to_numpy(), valid


def dtype_of(annotation: Any) -> np.dtype:
    """The array dtype a value of `annotation` is stored in: float64, int64, bool, else object."""
    return np.dtype({float: np.float64, int: np.int64, bool: np.bool_}.get(annotation, object))
