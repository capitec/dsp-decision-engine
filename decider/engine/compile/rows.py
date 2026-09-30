"""`Columnar[Item]`: a list column's items as one array per field, sliced per parent row.

Physical shape: one flat array per `Item` field (like `param_table`'s bundle),
sliced `offsets[r]:offsets[r + 1]` for parent row `r`. Two schemas with the
same field name/type sequence share one generated namedtuple type. A column
straight from the input frame is read out of its Arrow buffers, which already
hold exactly that shape; anything else is built from the Python values.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import namedtuple
from itertools import accumulate
from typing import Any, NamedTuple

import numpy as np
import polars as pl

from decider.engine.compile.span import SPAN_DTYPE, span_array, spans_of
from decider.engine.ir.decls import NullPolicy, base_annotation

Schema = tuple[tuple[str, Any], ...]

# Above this, reading the Arrow buffers beats iterating the Python values; below it the export
# costs more than it saves.
ARROW_ROWS = 256

_DTYPES = {float: np.dtype(np.float64), int: np.dtype(np.int64), bool: np.dtype(np.bool_),
           str: SPAN_DTYPE}


class _Layout(NamedTuple):
    nt: type
    dtypes: tuple[np.dtype, ...]
    optional: tuple[bool, ...]


_LAYOUTS: dict[Schema, _Layout] = {}


def _layout(schema: Schema) -> _Layout:
    layout = _LAYOUTS.get(schema)
    if layout is None:
        if not schema:
            raise TypeError("Columnar[Item] needs an Item with at least one field")
        fields = [_field(name, t) for name, t in schema]
        nt = namedtuple("Item_" + "_".join(n for n, _ in schema), [n for n, _ in schema])
        layout = _LAYOUTS[schema] = _Layout(nt, tuple(d for d, _ in fields), tuple(o for _, o in fields))
    return layout


def _field(name: str, annotation: Any) -> tuple[np.dtype, bool]:
    base = base_annotation(annotation)
    dtype = _DTYPES.get(base)
    optional = base is not annotation
    if dtype is None:
        raise TypeError(
            f"Columnar[Item] field '{name}' is {annotation}; a field must be float, int, bool, str, "
            "`float | None` or `str | None`. A field that is itself a list or struct is nested data, "
            "which Columnar[Item] cannot hold: annotate the input as list[dict] instead.")
    if optional and base not in (float, str):
        raise TypeError(f"Columnar[Item] field '{name}' is {annotation}, which has no null value a kernel "
                        "can hold; only `float | None` and `str | None` do")
    return dtype, optional


def rows_class(schema: Schema) -> type:
    """The namedtuple class a `Columnar[Item]` value of `schema` is an instance of."""
    return _layout(schema).nt


def rows_probe(schema: Schema) -> Any:
    """A zero-length instance of `schema`'s namedtuple: enough for numba to type the argument.

    Example::

        typeof(rows_probe((("price", float),)))
    """
    nt, dtypes, _ = _layout(schema)
    return nt(*(span_array(np.empty((0, 2), np.int64)) if dtype == SPAN_DTYPE else np.empty(0, dtype)
                for dtype in dtypes))


def build_rows(values: np.ndarray, schema: Schema, source: pl.Series | None = None,
               alive: list | None = None) -> np.ndarray:
    """One `schema` namedtuple of array views per row, sliced from flat per-field arrays.

    `values` holds one `list[Item]` (or `None`) per row; a null list is a row
    with no items. A null in a field raises unless the field is declared
    `float | None`, which reads it as NaN. `source` is the input frame's own
    column while it still holds those values, and is then read straight out of
    its Arrow buffers; `alive` keeps that import alive for as long as the
    arrays are used.

    Example::

        build_rows(np.array([[{"price": 2.0}], None], object), (("price", float),))
    """
    nt, dtypes, optional = _layout(schema)
    nested = _from_arrow(values, schema, dtypes, optional, source, alive)
    if nested is not None:
        fields, starts, stops = nested.fields, nested.starts, nested.stops
    else:
        fields = [_flat(values, name, dtype, opt)
                  for (name, _), dtype, opt in zip(schema, dtypes, optional)]
        if len(values) == 1:
            # score()'s shape: the one row owns every item, so there is nothing to slice.
            out = np.empty(1, object)
            out[0] = nt(*fields)
            return out
        starts = item_starts(values)
        stops = starts[1:]
    # One C-level pass per step rather than a Python loop over the rows: a third cheaper.
    # `ndarray.__getitem__` unbound, so a `SpanArray`'s Python one (which a step's own `[j]`
    # needs) doesn't cost a Python call per row here.
    windows = list(map(slice, starts, stops))
    columns = [list(map(np.ndarray.__getitem__.__get__(f), windows)) for f in fields]
    return np.fromiter(map(nt, *columns), object, len(values))


def item_starts(values: np.ndarray) -> list[int]:
    """The flat index each row's items start at, plus the total: `len(values) + 1` entries."""
    return [0, *accumulate(0 if row is None else len(row) for row in values)]


class Ragged:
    """The same flat arrays, unsliced: row `i` owns `fields[k][lo[i]:hi[i]]`.

    What a shared array kernel takes for a `Columnar[Item]` input, in place of one
    namedtuple of views per row. `lo` and `hi` are per-row, so a branch or loop
    row subset is `rag[rows]` and never touches the child arrays.
    """

    __slots__ = ("lo", "hi", "fields", "nt")

    def __init__(self, lo: np.ndarray, hi: np.ndarray, fields: tuple[np.ndarray, ...], nt: type):
        self.lo, self.hi, self.fields, self.nt = lo, hi, fields, nt

    def __getitem__(self, rows: np.ndarray) -> Ragged:
        return Ragged(self.lo[rows], self.hi[rows], self.fields, self.nt)

    def row(self, i: int) -> Any:
        # `build_rows`'s shape for one row: what a kernel that fails at run time falls back to,
        # calling the per-row dispatcher directly.
        return self.nt(*(f[self.lo[i]:self.hi[i]] for f in self.fields))

    @property
    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.lo, self.hi, *self.fields)


def build_ragged(values: np.ndarray, schema: Schema, source: pl.Series | None = None,
                 alive: list | None = None) -> Ragged:
    """`values` as flat per-field arrays plus each row's `[lo, hi)` into them.

    Same nulls and same Arrow read as `build_rows`; it just stops before
    assembling one namedtuple per row.
    """
    nt, dtypes, optional = _layout(schema)
    nested = _from_arrow(values, schema, dtypes, optional, source, alive)
    if nested is not None:
        return Ragged(np.asarray(nested.starts, np.int64), np.asarray(nested.stops, np.int64),
                      tuple(nested.fields), nt)
    fields = tuple(_flat(values, name, dtype, opt)
                   for (name, _), dtype, opt in zip(schema, dtypes, optional))
    starts = np.asarray(item_starts(values), np.int64)
    return Ragged(starts[:-1], starts[1:], fields, nt)


def build_exploded(values: np.ndarray, schema: Schema, source: pl.Series | None,
                   alive: list | None) -> tuple[np.ndarray, np.ndarray, tuple, tuple] | None:
    """`values` as flat per-field arrays, per-row `[lo, hi)` offsets and per-field validity masks.

    Reads the Arrow buffers zero-copy, so no item dict is touched. `None` when
    the column can't be read that way: not Arrow-backed, too few rows, a field
    that isn't float/int/bool, or a struct with fields beyond `schema`.
    """
    dtypes = []
    for _, annotation in schema:
        base = base_annotation(annotation)
        if base not in (float, int, bool):
            return None
        dtypes.append(_DTYPES[base])
    if source is None or alive is None or len(values) < ARROW_ROWS:
        return None
    from decider.engine.boundary._arrow.nested import read_exploded

    exploded = read_exploded(source, schema, tuple(dtypes))
    if exploded is None:
        return None
    alive.append(exploded)
    return (np.asarray(exploded.starts, np.int64), np.asarray(exploded.stops, np.int64),
            tuple(exploded.fields), tuple(exploded.validity))


def _from_arrow(values, schema, dtypes, optional, source, alive):
    if source is None or alive is None or len(values) < ARROW_ROWS:
        return None
    from decider.engine.boundary._arrow.nested import read_nested

    nested = read_nested(source, schema, dtypes, optional)
    if nested is not None:
        alive.append(nested)
    return nested


def rows_needs_no_fill(decl, is_rows: bool) -> bool:
    """True when `decl`'s MISSING_AS fill is already what `Columnar[...]` gives a null row: no items.

    Raises when the fill is a non-empty list, which no `Columnar[...]` input can hold.
    """
    if not is_rows or decl.null_policy is not NullPolicy.MISSING_AS:
        return False
    if decl.fill:
        raise TypeError(f"input '{decl.name}': missing_as({decl.fill!r}) on a Columnar[...] input must be an "
                        "empty list; a null or absent list already reads as a row with no items")
    return True


NULL_FLOAT = "`float | None` to read a null as NaN"
NULL_STR = "`str | None` to read a null as a null span"


def null_item(name: str, at: int, starts, optional: str = NULL_FLOAT) -> ValueError:
    """The error for a null in `Item` field `name`, at flat item index `at`."""
    row = bisect_right(starts, at) - 1
    return ValueError(
        f"'{name}' is null in item {at - starts[row]} of row {row}; a Columnar[Item] field cannot hold a "
        f"null. Declare it {optional}, or fill the column in the frame.")


def _flat(values: np.ndarray, name: str, dtype: np.dtype, optional: bool) -> np.ndarray:
    try:
        items = [item[name] for row in values if row for item in row]
    except TypeError:
        raise ValueError(f"a Columnar[Item] row holds a null item; every item must have a "
                         f"'{name}' field") from None
    except KeyError:
        raise ValueError(f"a Columnar[Item] item has no '{name}' field") from None
    if dtype == SPAN_DTYPE:
        starts = item_starts(values)
        return spans_of(items, optional, lambda k: null_item(name, k, starts, NULL_STR))
    # numpy turns a None into NaN for a float field, which would sum silently. `in` costs a
    # tenth of a microsecond on one row, where `np.isnan(...).any()` costs four.
    if not optional and None in items:
        raise null_item(name, items.index(None), item_starts(values))
    return np.fromiter(items, dtype, len(items))
