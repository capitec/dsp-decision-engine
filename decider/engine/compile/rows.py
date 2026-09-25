"""`Rows[Item]`: a list column's items as one array per field, sliced per parent row.

Physical shape: one flat array per `Item` field (like `param_table`'s bundle),
sliced `offsets[r]:offsets[r + 1]` for parent row `r`. Two schemas with the
same field name/type sequence share one generated namedtuple type.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np

from decider.engine.ir.decls import KIND_DTYPES, FeatureKind, feature_kind

Schema = tuple[tuple[str, Any], ...]

_NAMEDTUPLES: dict[Schema, type] = {}
_SUPPORTED = (float, int, bool)


def rows_dtype(annotation: Any) -> np.dtype:
    kind = feature_kind(annotation)
    return KIND_DTYPES[FeatureKind.F64 if kind is FeatureKind.STR else kind]


def rows_namedtuple(schema: Schema) -> type:
    """The generated namedtuple type for `schema`, one per distinct field name/type sequence.

    Example::

        rows_namedtuple((("price", float),))(price=np.array([1.0, 2.0]))
    """
    nt = _NAMEDTUPLES.get(schema)
    if nt is None:
        bad = [name for name, t in schema if t not in _SUPPORTED]
        if bad:
            raise TypeError(f"Rows[Item] field(s) {bad} must be float, int or bool for now")
        nt = _NAMEDTUPLES[schema] = namedtuple("Item_" + "_".join(n for n, _ in schema), [n for n, _ in schema])
    return nt


def rows_probe(schema: Schema) -> Any:
    """A zero-length instance of `schema`'s namedtuple: enough for numba to type the argument.

    Example::

        typeof(rows_probe((("price", float),)))
    """
    nt = rows_namedtuple(schema)
    return nt(*(np.empty(0, rows_dtype(t)) for _, t in schema))


def build_rows(values: np.ndarray, schema: Schema) -> np.ndarray:
    """One `schema` namedtuple of array views per row, sliced from flat per-field arrays.

    `values` holds one `list[Item]` (or `None`) per row.

    Example::

        build_rows(np.array([[{"price": 2.0}], None], object), (("price", float),))
    """
    nt = rows_namedtuple(schema)
    lengths = np.fromiter((0 if row is None else len(row) for row in values), np.int64, len(values))
    offsets = np.zeros(len(values) + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    fields = {name: np.empty(int(offsets[-1]), rows_dtype(ftype)) for name, ftype in schema}
    k = 0
    for row in values:
        if row is None:
            continue
        for item in row:
            for name, _ in schema:
                fields[name][k] = item[name]
            k += 1
    out = np.empty(len(values), object)
    for i in range(len(values)):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        out[i] = nt(*(fields[name][lo:hi] for name, _ in schema))
    return out
