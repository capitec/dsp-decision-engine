"""The polars dtype an `each` gathers into, derived from the pipeline's declarations; every engine uses it."""
from __future__ import annotations

import typing
from typing import Any

import polars as pl

from decikit._engine.ir.decls import base_annotation
from decikit._engine.run.state import declared_dtype
from decikit._engine.types import item_schema
from decikit._engine.wiring.plan import NodeVersion, ScatterGather
from decikit.exceptions import EngineError


def gathered_dtype(sg: ScatterGather, gathers: dict, known: dict, schema: Any) -> pl.DataType:
    # As the gather builds it: every field of the item in its order, then each value the body writes.
    path = sg.node.origin.path
    column = version_dtype(sg.column, gathers, known, schema)
    if not (isinstance(column, pl.List) and isinstance(column.inner, pl.Struct)):
        how = (f"reaches it as {column}" if sg.column.producer is None
               else f"{sg.column.producer} writes as {named(sg.column.annotation)}")
        raise EngineError(
            f"the polars engine needs every column's dtype before any row arrives, and {path} gathers "
            f"'{sg.out.name}' from the items of '{sg.column.name}', which {how}, with no item type. Annotate "
            f"the step that writes it as a list of a TypedDict naming the item's fields, or run the pipeline "
            f"on the numba engine.")
    fields = {f.name: f.dtype for f in column.inner.fields}
    known = {**known, **{v.id: fields[name] for name, v in sg.fields if name in fields}}
    for v in sg.new_fields:
        if (dtype := version_dtype(v, gathers, known, schema)) is None:
            raise EngineError(
                f"the polars engine needs every column's dtype before any row arrives, and {path} gathers "
                f"'{v.name}', written by {v.producer} as {named(v.annotation)}, into every item of "
                f"'{sg.out.name}'. Annotate {v.producer} with a type polars can name (a TypedDict, "
                f"list[float], str), or run the pipeline on the numba engine.")
        fields[v.name] = dtype
    return pl.List(pl.Struct(fields))


def version_dtype(v: NodeVersion, gathers: dict, known: dict, schema: Any) -> pl.DataType | None:
    # `known` holds the dtype of each item field the enclosing eaches read; `schema` the input columns'.
    if v.id in known:
        return known[v.id]
    if v.id in gathers:
        return gathered_dtype(gathers[v.id], gathers, known, schema)
    if v.producer is None:
        return schema.get(v.name)
    return annotated_dtype(v.annotation)


def annotated_dtype(annotation: Any) -> Any:
    if isinstance(annotation, pl.DataType) or (isinstance(annotation, type) and issubclass(annotation, pl.DataType)):
        return annotation
    t = base_annotation(annotation)
    if typing.is_typeddict(t):
        fields = {name: annotated_dtype(ft) for name, ft in item_schema(t)}
        return None if None in fields.values() else pl.Struct(fields)
    if typing.get_origin(t) is list and (inner := annotated_dtype(typing.get_args(t)[0])) is not None:
        return pl.List(inner)
    return declared_dtype(annotation)


def named(annotation: Any) -> str:
    # `list[dict].__name__` is "list".
    return str(annotation) if typing.get_args(annotation) else getattr(annotation, "__name__", str(annotation))


def fitted(s: pl.Series, dtype: pl.DataType) -> pl.Series:
    """`s` cast to `dtype` when that only adds null fields, reorders fields or widens values; else `s` itself."""
    if s.dtype == dtype or not _within(s.dtype, dtype):
        return s
    # polars casts a struct by field name, and a float to an int by truncation; casting back shows any loss.
    cast = s.cast(dtype, strict=False)
    return cast if cast.cast(s.dtype, strict=False).equals(s) else s


def _within(got: pl.DataType, want: pl.DataType) -> bool:
    if got == pl.Null:
        return True
    if isinstance(want, pl.Struct):
        fields = {f.name: f.dtype for f in want.fields}
        return isinstance(got, pl.Struct) and all(f.name in fields and _within(f.dtype, fields[f.name])
                                                  for f in got.fields)
    if isinstance(want, (pl.List, pl.Array)):
        return isinstance(got, (pl.List, pl.Array)) and _within(got.inner, want.inner)
    # A cast between time zones keeps the instant but not the wall clock a reader sees.
    return not got.is_nested() and getattr(got, "time_zone", None) == getattr(want, "time_zone", None)
