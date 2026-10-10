"""`Struct[Item]`: a struct column as one record per row, for a compiled step.

Physical shape: a numpy record array with one field per `Item` field, so row
`i` is `records[i]` and a step reads `applicant["income"]` the same way in
Python and in a kernel. Fields are read out of Arrow's flat child arrays
through the boundary an ordinary column uses.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl

from decikit._engine.concurrency import Cache, once
from decikit._engine.ir.decls import KIND_DTYPES, Input, NullPolicy, base_annotation, feature_kind
from decikit.exceptions import MissingInputError
from decikit.ops.nested import struct_item
from decikit._engine.types import annotation_cache, columnar_item, item_schema

Schema = tuple[tuple[str, Any], ...]

FIELD_TYPES = (float, int, bool)
_DTYPES = Cache()
_MISSING = object()


def struct_schema(item: Any) -> Schema:
    """`Item`'s fields in declaration order, as `(name, type)` pairs."""
    return item_schema(item)


@annotation_cache
def to_dicts(annotation: Any) -> Callable[[Any], Any] | None:
    """How a Python step's `Struct[Item]` or `Columnar[Item]` value becomes the dicts every engine stores; else `None`.

    A step may return a struct as a tuple of its fields and rows as a list of
    them; a kernel's output reads back as dicts, so the Python paths store the same.
    """
    t = base_annotation(annotation)
    if (item := struct_item(t)) is not None:
        names = [name for name, _ in item_schema(item)]
        return lambda x: x if isinstance(x, dict) else dict(zip(names, x))
    if (item := columnar_item(t)) is not None:
        names = [name for name, _ in item_schema(item)]
        return lambda xs: [x if isinstance(x, dict) else dict(zip(names, x)) for x in xs]
    return None


def bad_field(schema: Schema) -> tuple[str, Any] | None:
    """The first field of `schema` a record can't hold, as `(name, type)`; `None` when all can."""
    return next(((name, t) for name, t in schema if t not in FIELD_TYPES), None)


def struct_dtype(schema: Schema) -> np.dtype:
    """The record dtype of `schema`, one per distinct field name/type sequence.

    Example::

        struct_dtype((("income", float), ("age", int)))
    """
    # The get comes first so a warm call builds no closure.
    return _DTYPES.get(schema) or once(_DTYPES, schema, lambda: _record_dtype(schema))


def _record_dtype(schema: Schema) -> np.dtype:
    bad = bad_field(schema)
    if bad is not None:
        raise TypeError(f"Struct[...] field {bad[0]!r} is {bad[1]}; a record field must be float, int or "
                        "bool. A field that is itself a list or struct is nested data, which Struct[...] "
                        "cannot hold: annotate the input as dict instead.")
    return np.dtype([(name, KIND_DTYPES[feature_kind(t)]) for name, t in schema], align=True)


def build_struct(schema: Schema, values: np.ndarray, source: pl.Series | None,
                 name: str, path: str) -> np.ndarray:
    """One record per row of a struct column, read from its Arrow child arrays.

    `values` holds one `dict` (or `None`) per row, and is read only when
    `source` is not the frame's own struct column. A null struct, a null
    field, a missing field or a column that is not a struct raises, naming
    the field.

    Example::

        build_struct((("income", float),), values, df["applicant"], "applicant", "p/afford")
    """
    out = np.empty(len(values), struct_dtype(schema))
    if source is not None and isinstance(source.dtype, pl.Struct):
        _from_arrow(schema, source, out, name, path)
    elif values.dtype == object:
        _from_dicts(schema, values, out, name, path)
    elif values.dtype == struct_dtype(schema):
        # Already the record array a compiled step produced.
        return values
    else:
        raise TypeError(f"input {name!r} of step {path!r} is declared Struct[...], but column {name!r} "
                        f"holds {values.dtype} values, not structs; declare the type the data has")
    return out


def _from_arrow(schema: Schema, source: pl.Series, out: np.ndarray, name: str, path: str) -> None:
    have = source.struct.fields
    absent = [f for f, _ in schema if f not in have]
    if absent:
        raise ValueError(f"input {name!r} of step {path!r} declares field(s) {absent} that column "
                         f"{name!r} does not have; it has {have}")
    # Zero-copy: polars hands back the child Series the struct already holds.
    fields = source.struct.unnest().select(f for f, _ in schema)
    fields.columns = [f"{name}.{f}" for f, _ in schema]
    from decikit._engine.boundary.extract import extract_frame

    declared = [Input(f"{name}.{f}", t, NullPolicy.OPTIONAL) for f, t in schema]
    extracted = extract_frame(fields, declared, path=path)
    # A null struct is a null column value, which each reader's null policy owns; polars pushes it
    # down into every child, so only the fields null on a row the struct itself has are the field's own.
    whole = source.is_not_null().to_numpy() if source.null_count() else None
    for (f, _), decl in zip(schema, declared):
        col = extracted.columns[decl.name]
        if col.has_nulls:
            missing = ~col.validity if whole is None else ~col.validity & whole
            if missing.any():
                raise MissingInputError(decl.name, path, int(missing.sum()), len(out),
                                        fix=f"Fill the field in the frame, or declare '{name}' as a plain `dict`.")
        # A null row carries the kind's fill (NaN or 0); the null policy stops it being read.
        out[f] = col.values


def _from_dicts(schema: Schema, values: np.ndarray, out: np.ndarray, name: str, path: str) -> None:
    fields = [f for f, _ in schema]
    views = [out[f] for f in fields]
    for i, row in enumerate(values):
        if not isinstance(row, Mapping):
            if row is None:
                raise MissingInputError(name, path, int(sum(x is None for x in values)), len(values))
            raise TypeError(f"input {name!r} of step {path!r} is declared Struct[...], but row {i} holds "
                            f"{type(row).__name__}, not a struct; declare the type the data has")
        for f, view in zip(fields, views):
            value = row.get(f, _MISSING)
            if value is _MISSING:
                raise ValueError(f"input {name!r} of step {path!r} has no field {f!r} on row {i}; "
                                 f"the value has {sorted(row)}")
            if value is None:
                raise MissingInputError(f"{name}.{f}", path, 1, len(values),
                                        fix=f"Fill the field, or declare '{name}' as a plain `dict`.")
            view[i] = value
