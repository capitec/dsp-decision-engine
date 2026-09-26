"""`Rows[Item]`: a list column's items as one array per field, sliced per parent row.

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

from decider.engine.ir.decls import NullPolicy, base_annotation

Schema = tuple[tuple[str, Any], ...]

# Above this, reading the Arrow buffers beats iterating the Python values; below it the export
# costs more than it saves.
ARROW_ROWS = 256

_DTYPES = {float: np.dtype(np.float64), int: np.dtype(np.int64), bool: np.dtype(np.bool_)}


class _Layout(NamedTuple):
    nt: type
    dtypes: tuple[np.dtype, ...]
    optional: tuple[bool, ...]


_LAYOUTS: dict[Schema, _Layout] = {}


def _layout(schema: Schema) -> _Layout:
    layout = _LAYOUTS.get(schema)
    if layout is None:
        if not schema:
            raise TypeError("Rows[Item] needs an Item with at least one field")
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
            f"Rows[Item] field '{name}' is {annotation}; a field must be float, int, bool or "
            "`float | None`" + (". No kernel holds a variable-length string, so drop the field from "
                                "Item and read the column in a python_only step." if base is str else "."))
    if optional and base is not float:
        raise TypeError(f"Rows[Item] field '{name}' is {annotation}, which has no null value a kernel "
                        "can hold; only `float | None` does, as NaN")
    return dtype, optional


def rows_probe(schema: Schema) -> Any:
    """A zero-length instance of `schema`'s namedtuple: enough for numba to type the argument.

    Example::

        typeof(rows_probe((("price", float),)))
    """
    nt, dtypes, _ = _layout(schema)
    return nt(*(np.empty(0, dtype) for dtype in dtypes))


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
    windows = list(map(slice, starts, stops))
    columns = [list(map(f.__getitem__, windows)) for f in fields]
    return np.fromiter(map(nt, *columns), object, len(values))


def item_starts(values: np.ndarray) -> list[int]:
    """The flat index each row's items start at, plus the total: `len(values) + 1` entries."""
    return [0, *accumulate(0 if row is None else len(row) for row in values)]


def _from_arrow(values, schema, dtypes, optional, source, alive):
    if source is None or alive is None or len(values) < ARROW_ROWS:
        return None
    from decider.engine.boundary._arrow.nested import read_nested

    nested = read_nested(source, schema, dtypes, optional)
    if nested is not None:
        alive.append(nested)
    return nested


def rows_needs_no_fill(decl, is_rows: bool) -> bool:
    """True when `decl`'s MISSING_AS fill is already what `Rows[...]` gives a null row: no items.

    Raises when the fill is a non-empty list, which no `Rows[...]` input can hold.
    """
    if not is_rows or decl.null_policy is not NullPolicy.MISSING_AS:
        return False
    if decl.fill:
        raise TypeError(f"input '{decl.name}': missing_as({decl.fill!r}) on a Rows[...] input must be an "
                        "empty list; a null or absent list already reads as a row with no items")
    return True


def null_item(name: str, at: int, starts) -> ValueError:
    """The error for a null in `Item` field `name`, at flat item index `at`."""
    row = bisect_right(starts, at) - 1
    return ValueError(
        f"'{name}' is null in item {at - starts[row]} of row {row}; a Rows[Item] field cannot hold a "
        "null. Declare it `float | None` to read a null as NaN, or fill the column in the frame.")


def _flat(values: np.ndarray, name: str, dtype: np.dtype, optional: bool) -> np.ndarray:
    try:
        items = [item[name] for row in values if row for item in row]
    except TypeError:
        raise ValueError(f"a Rows[Item] row holds a null item; every item must have a "
                         f"'{name}' field") from None
    except KeyError:
        raise ValueError(f"a Rows[Item] item has no '{name}' field") from None
    # numpy turns a None into NaN for a float field, which would sum silently. `in` costs a
    # tenth of a microsecond on one row, where `np.isnan(...).any()` costs four.
    if not optional and None in items:
        raise null_item(name, items.index(None), item_starts(values))
    return np.fromiter(items, dtype, len(items))
