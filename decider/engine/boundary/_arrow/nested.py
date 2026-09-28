"""A `list<struct<...>>` column read as Arrow already stores it: flat field arrays plus the offsets.

Arrow's layout is the one `Columnar[Item]` wants, so nothing is rebuilt: the
offsets are the per-row slices and each struct field is already one flat
array. Only a field whose Arrow width differs from the kernel's, a bit-packed
bool and a nulled field cost a vectorised pass.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import polars as pl

from decider.engine.boundary._arrow._shim import GET_STRING_ADDR, lib
from decider.engine.boundary._arrow.kernels import string_spans
from decider.engine.boundary._arrow.plan import ArrowImportError, FramePlan
from decider.engine.boundary._arrow.view import FrameView, _Borrowed
from decider.engine.compile.span import SPAN_DTYPE, span_array

# nanoarrow storage types (enum values are API): the list kinds, by offset width.
_OFFSET_DTYPE = {26: np.int32, 37: np.int64}
_STRUCT = 27
# Field storage types this can read, as the numpy dtype of the buffer.
_FIELD_DTYPE = {13: np.dtype(np.float64), 12: np.dtype(np.float32),
                10: np.dtype(np.int64), 8: np.dtype(np.int32), 2: None}  # None: bit-packed bool
# utf8, large_utf8, utf8_view: all three read through nanoarrow's own accessor.
_STRING = (14, 35, 41)


class Nested:
    """One `list<struct<...>>` column's flat field arrays and per-row bounds.

    `fields[k]` is field `k` of the whole column; row `i` owns
    `fields[k][starts[i]:stops[i]]`. Holds the bound Arrow import the arrays
    view, so it must outlive them.
    """

    __slots__ = ("fields", "starts", "stops", "view")

    def __init__(self, fields: list[np.ndarray], starts: list[int], stops: list[int], view: FrameView):
        self.fields = fields
        self.starts = starts
        self.stops = stops
        self.view = view


@lru_cache(maxsize=64)
def _plan(name: str) -> FramePlan:
    # Declares no column: nothing to resolve, the nested child is walked by hand.
    return FramePlan([], (name,))


def _struct_fields(series: pl.Series) -> dict[str, int] | None:
    # polars keeps the struct's field order on export, so the declared names map to child indices.
    dtype = series.dtype
    if not isinstance(dtype, pl.List) or not isinstance(dtype.inner, pl.Struct):
        return None
    return {f.name: k for k, f in enumerate(dtype.inner.fields)}


def read_nested(series: pl.Series, schema, dtypes, optional) -> Nested | None:
    """`series` as flat field arrays plus per-row bounds, or `None` when its layout isn't readable.

    `schema` names the `Item` fields to read, `dtypes` the numpy dtype each one
    enters the kernel as, and `optional[k]` whether field `k` may be null.

    Example::

        read_nested(df["items"], (("price", float),), [np.dtype(np.float64)], [False])
    """
    index = _struct_fields(series)
    if index is None or any(name not in index for name, _ in schema):
        return None
    view = FrameView(_plan(series.name))
    try:
        # An all-null struct field is Arrow `null`, which polars exports with a buffer nanoarrow
        # refuses; the Python values read it either way.
        view.bind(series.to_frame(series.name))
    except ArrowImportError:
        return None
    try:
        return _read(view, series.name, schema, dtypes, optional, index)
    except Exception:
        view.release()
        raise


def _read(view, name, schema, dtypes, optional, index) -> Nested | None:
    top = view.child_view(name)
    offset_dtype = _OFFSET_DTYPE.get(lib.sm_view_storage_type(top))
    struct = lib.sm_view_child(top, 0)
    if offset_dtype is None or not struct or lib.sm_view_storage_type(struct) != _STRUCT:
        view.release()
        return None
    n, top_offset = lib.sm_view_length(top), lib.sm_view_offset(top)
    offsets = _buffer(top, 1, top_offset, n + 1, offset_dtype, view)
    # A sliced frame keeps the whole child, so each field is windowed to the items these rows own
    # and the offsets are rebased onto that window: a null in a row outside the slice is not ours.
    base = int(offsets[0]) if n else 0
    starts, stops = (offsets[:-1] - base).tolist(), (offsets[1:] - base).tolist()
    count = stops[-1] if stops else 0
    # A whole item being null is not a value `Columnar[Item]` can hold; check only items in this slice.
    if count and lib.sm_view_null_count(struct):
        valid = _validity(struct, lib.sm_view_offset(struct) + base, count, view)
        if not valid.all():
            view.release()
            return None
    fields = []
    for (field, _), dtype, opt in zip(schema, dtypes, optional):
        x = _field(lib.sm_view_child(struct, index[field]), dtype, opt, field, base, count, starts, view)
        if x is None:
            view.release()
            return None
        fields.append(x)
    if lib.sm_view_null_count(top):
        # A null list is a row with no items, so it reads nothing rather than its neighbour's items.
        stops = np.where(_validity(top, top_offset, n, view), stops, starts).tolist()
    return Nested(fields, starts, stops, view)


def _field(child, dtype: np.dtype, optional: bool, name: str, base: int, count: int, starts,
           view) -> np.ndarray | None:
    if not child:
        return None
    storage = lib.sm_view_storage_type(child)
    if dtype == SPAN_DTYPE:
        return _spans(child, storage, optional, name, base, count, starts, view)
    if storage not in _FIELD_DTYPE:
        return None
    at = lib.sm_view_offset(child) + base
    valid = _validity(child, at, count, view) if lib.sm_view_null_count(child) else None
    if valid is not None and valid.all():
        valid = None   # every null the column has is in a row outside this slice
    if valid is not None and not optional:
        from decider.engine.compile.rows import null_item

        raise null_item(name, int(np.argmin(valid)), starts)
    if storage == 2:
        x = _bits(child, 1, at, count, view).astype(dtype)
    else:
        x = _buffer(child, 1, at, count, _FIELD_DTYPE[storage], view)
        if x.dtype != dtype:
            x = x.astype(dtype)
    return x if valid is None else np.where(valid, x, np.nan)


def _spans(child, storage: int, optional: bool, name: str, base: int, count: int, starts,
           view) -> np.ndarray | None:
    if storage not in _STRING:
        return None
    if not optional and lib.sm_view_null_count(child):
        valid = _validity(child, lib.sm_view_offset(child) + base, count, view)
        if not valid.all():
            from decider.engine.compile.rows import NULL_STR, null_item

            raise null_item(name, int(np.argmin(valid)), starts, NULL_STR)
    pairs = np.empty((count, 2), np.int64)
    # The strings stay where Arrow put them; only the addresses are gathered. nanoarrow's
    # accessor adds the child's own offset, so the index is the window start.
    string_spans(GET_STRING_ADDR, child, base, count, pairs)
    return span_array(pairs, kept=view)


def _buffer(v, k: int, at: int, n: int, dtype, view: FrameView) -> np.ndarray:
    addr = lib.sm_view_buffer(v, k)
    if not addr:
        raise ValueError(f"the Arrow column is missing buffer {k}")
    return np.asarray(_Borrowed(addr + at * np.dtype(dtype).itemsize, n, np.dtype(dtype), view))


def _bits(v, k: int, at: int, n: int, view: FrameView) -> np.ndarray:
    # A bit-packed buffer has to be unpacked whole bytes at a time, then trimmed to [at, at + n).
    byte, shift = divmod(at, 8)
    bits = _buffer(v, k, byte, (shift + n + 7) // 8, np.uint8, view)
    return np.unpackbits(bits, bitorder="little")[shift:shift + n]


def _validity(v, at: int, n: int, view: FrameView) -> np.ndarray:
    return _bits(v, 0, at, n, view).view(bool)
