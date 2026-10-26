"""Holding a step to the dict, TypedDict, list, `Struct` or `Columnar` output it declares.

A step returning tuples that a kernel stores is checked once, at compile time,
against the type numba inferred for its body. Every other step runs in Python
(no kernel stores a dict) and is checked by pydantic on every call.
"""
from __future__ import annotations

import typing
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from numba.core import types
from pydantic import PydanticSchemaGenerationError, PydanticUserError, TypeAdapter, ValidationError

from decikit._engine.ir.decls import Output, base_annotation
from decikit._engine.types import columnar_item, item_schema, pydantic_ready
from decikit.ops.nested import struct_item
from decikit.ops.text import Text


def _bare(t: Any) -> bool:
    # Says nothing about its values: a plain `dict`, `list[dict]`, `dict[str, Any]`.
    origin, args = typing.get_origin(t), typing.get_args(t)
    if origin is dict:
        return len(args) < 2 or _bare(args[1])
    if origin is list:
        return not args or _bare(args[0])
    return t in (dict, list, Any, object)


def schema_of(annotation: Any) -> Any:
    """The type a step's output must have, or `None` when it declares no shape beyond `dict` / `list`.

    >>> schema_of(dict), schema_of(list[float])
    (None, list[float])
    """
    t = base_annotation(annotation)
    if (item := struct_item(t)) is not None:
        return item
    if (item := columnar_item(t)) is not None:
        return list[item]
    if typing.is_typeddict(t) or (typing.get_origin(t) in (dict, list) and not _bare(t)):
        return t
    return None


def is_untyped_container(annotation: Any) -> bool:
    """Whether the output is a plain `dict` or `list[dict]`, with no shape declared."""
    t = base_annotation(annotation)
    return (typing.get_origin(t) or t) in (dict, list) and _bare(t)


# Raw `lru_cache`, not `annotation_cache`: a step asks this on every call, and its key is hashed
# straight, a fifth of the cost; an annotation carrying unhashable metadata is built uncached.
@lru_cache(maxsize=256)
def _prepared(annotation: Any) -> tuple[TypeAdapter, list[str] | None, bool] | None:
    t = schema_of(annotation)
    if t is None:
        return None
    try:
        adapter = TypeAdapter(list[typing.Optional[pydantic_ready(t)]])
    except (PydanticUserError, PydanticSchemaGenerationError):
        return None  # a type pydantic has no schema for is left unchecked
    # A `Struct` or `Columnar` step may return its records as tuples of the fields.
    t = base_annotation(annotation)
    item = struct_item(t) or columnar_item(t)
    return adapter, None if item is None else [name for name, _ in item_schema(item)], columnar_item(t) is not None


def _name(annotation: Any) -> str:
    return annotation.__qualname__ if isinstance(annotation, type) and typing.get_origin(annotation) is None else str(annotation)


def check_output(step: str, output: Output, values: Sequence, rows: np.ndarray | None = None) -> None:
    """Raise a `ValueError` naming `step`, the output, the row and field if a value isn't the declared type.

    Strict: an int is a valid float, a str or bool is not. A `None` is a
    null, left to the output's null policy. `rows` maps positions in `values`
    to the frame rows a step ran on.
    """
    try:
        prepared = _prepared(output.annotation)
    except TypeError:
        prepared = _prepared.__wrapped__(output.annotation)
    if prepared is None:
        return
    adapter, names, many = prepared
    if names is not None:
        def record(x):
            return dict(zip(names, x)) if isinstance(x, (tuple, list)) and len(x) == len(names) else x

        values = [None if x is None else [record(i) for i in x] if many else record(x) for x in values]
    values = values if type(values) is list else list(values)
    try:
        adapter.validate_python(values, strict=True)
        return
    except ValidationError:
        pass
    values = [_plain(x) for x in values]
    try:
        adapter.validate_python(values, strict=True)
    except ValidationError as e:
        err = e.errors(include_url=False)[0]
        row, *field = err["loc"]
        at = f" row {int(rows[row]) if rows is not None else row}"
        at += f", field {'.'.join(map(str, field))}" if field else ""
        raise ValueError(f"step {step}: output '{output.name}'{at}: {err['msg']} (got {err['input']!r}), "
                         f"declared {_name(output.annotation)}") from None


def _plain(x: Any) -> Any:
    # What the caller reads: a `Text` (from a `Columnar` input) is its `str`, a numpy scalar its Python number.
    if isinstance(x, Text):
        return x.text
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return type(x)(map(_plain, x)) if type(x) in (list, tuple) else x


def tuple_mismatch(output: Output, item: Any, parts: Sequence) -> str | None:
    """Why a tuple of numba types `parts` can't be one record of `item`, else `None`."""
    if len(parts) != len(item_schema(item)):
        return f"returns a tuple of {len(parts)} for output '{output.name}', declared {len(item_schema(item))} fields"
    for (name, t), part in zip(item_schema(item), parts):
        if not _fits(t, part):
            return f"returns {types.unliteral(part)} for field '{name}' of output '{output.name}', declared {_name(t)}"
    return None


def _fits(t: Any, nb: Any) -> bool:
    # Whether a value numba typed `nb` can be a record field declared `t`.
    nb = types.unliteral(nb)
    t = base_annotation(t)
    if t is float:
        return isinstance(nb, (types.Float, types.Integer))
    if t is int:
        return isinstance(nb, types.Integer)
    if t is bool:
        return isinstance(nb, types.Boolean)
    return True
