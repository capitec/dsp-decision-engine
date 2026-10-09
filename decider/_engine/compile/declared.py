"""Holding a step to the dict, TypedDict, list, `Struct` or `Columnar` output it declares.

A step returning tuples that a kernel stores is checked once, at compile time,
against the type numba inferred for its body. Every other step runs in Python
(no kernel stores a dict) and is checked on every call: directly for the usual shapes, by pydantic otherwise.
"""
from __future__ import annotations

import typing
from functools import lru_cache
from types import UnionType
from typing import Any, Callable, Sequence

import numpy as np
import typing_extensions as te
from numba.core import types
from pydantic import PydanticSchemaGenerationError, PydanticUserError, TypeAdapter, ValidationError

from decider._engine.ir.decls import Output, base_annotation
from decider._engine.types import columnar_item, item_schema, pydantic_ready
from decider.ops.nested import struct_item
from decider.ops.text import Text


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


_EXACT = {float: frozenset({float, int}), int: frozenset({int}), bool: frozenset({bool}), str: frozenset({str})}


def _fast(t: Any) -> Callable[[Any], bool] | None:
    # A test that is true only for values strict validation accepts as `t`, for the common shapes
    # (numbers, text, optional, list, TypedDict of those). Anything else, and anything it rejects,
    # goes to pydantic, which also words the error.
    if isinstance(t, type) and t in _EXACT:
        ok = _EXACT[t]
        return lambda x: type(x) in ok
    origin, args = typing.get_origin(t), typing.get_args(t)
    if origin in (typing.Union, UnionType) and len(args) == 2 and type(None) in args:
        inner = _fast(next(a for a in args if a is not type(None)))
        return None if inner is None else lambda x: x is None or inner(x)
    if origin is list and len(args) == 1:
        if isinstance(args[0], type) and args[0] in _EXACT:
            ok = _EXACT[args[0]]
            return lambda x: type(x) is list and ok.issuperset(map(type, x))
        inner = _fast(args[0])
        return None if inner is None else lambda x: type(x) is list and all(map(inner, x))
    if typing.is_typeddict(t):
        fields = {k: _fast(_required(h)) for k, h in te.get_type_hints(t, include_extras=True).items()}
        if None in fields.values():
            return None
        required = tuple(t.__required_keys__)

        def record(x):
            return (type(x) is dict and all(k in x for k in required)
                    and all(k not in x or f(x[k]) for k, f in fields.items()))
        return record
    return None


def _required(h: Any) -> Any:
    return typing.get_args(h)[0] if typing.get_origin(h) in (te.Required, te.NotRequired) else h


class Check:
    """What a step's output must be, prepared once per annotation; calling it raises `ValueError` if a value is not."""

    __slots__ = ("_adapter", "_names", "_many", "_fast")

    def __init__(self, adapter: TypeAdapter, names: list[str] | None, many: bool, fast: Any):
        self._adapter, self._names, self._many, self._fast = adapter, names, many, fast

    def __call__(self, step: str, output: Output, values: Sequence, rows: np.ndarray | None = None) -> None:
        # The usual outputs are lists of numbers or text; telling they are right costs less than asking pydantic.
        fast = self._fast
        if fast is not None and all(map(fast, values)):
            return
        adapter, names = self._adapter, self._names
        if names is not None:
            def record(x):
                return dict(zip(names, x)) if isinstance(x, (tuple, list)) and len(x) == len(names) else x

            many = self._many
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


# Raw `lru_cache`, not `annotation_cache`: a step asks this on every call, and its key is hashed
# straight, a fifth of the cost; an annotation carrying unhashable metadata is built uncached.
@lru_cache(maxsize=256)
def _prepared(annotation: Any) -> Check | None:
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
    # ponytail: a Struct or Columnar output always goes through pydantic, its records are tuples.
    fast = _fast(typing.Optional[schema_of(annotation)]) if item is None else None
    return Check(adapter, None if item is None else [name for name, _ in item_schema(item)],
                 columnar_item(t) is not None, fast)


def _name(annotation: Any) -> str:
    return annotation.__qualname__ if isinstance(annotation, type) and typing.get_origin(annotation) is None else str(annotation)


def output_check(output: Output) -> Check | None:
    """The `Check` for `output`, or `None` when it declares no shape beyond `dict` / `list`.

    Fetch it once, at bind, so a step does not look the annotation up on every call.
    """
    try:
        return _prepared(output.annotation)
    except TypeError:
        return _prepared.__wrapped__(output.annotation)


def check_output(step: str, output: Output, values: Sequence, rows: np.ndarray | None = None) -> None:
    """Raise a `ValueError` naming `step`, the output, the row and field if a value isn't the declared type.

    Strict: an int is a valid float, a str or bool is not. A `None` is a
    null, left to the output's null policy. `rows` maps positions in `values`
    to the frame rows a step ran on.
    """
    if (check := output_check(output)) is not None:
        check(step, output, values, rows)


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
