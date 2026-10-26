from __future__ import annotations

import sys
from enum import Enum
from functools import lru_cache, wraps
from types import UnionType
from typing import Annotated, Any, NamedTuple, Union, get_args, get_origin, get_type_hints

import numpy as np
import typing_extensions as te

from decikit.ops.measures import DecimalSpec
from decikit.ops.nested import Columnar
from decikit.ops.text import Text


class Representation(Enum):
    """Compiled storage representations available for semantic values."""

    SEMANTIC_STRING = "semantic_string"
    SEMANTIC_BYTES = "semantic_bytes"
    RAW_BYTES = "raw_bytes"


class RepresentationKey(NamedTuple):
    base_type: type
    raw_annotation: bool


REPRESENTATIONS: dict[RepresentationKey, Representation] = {
    RepresentationKey(str, False): Representation.SEMANTIC_STRING,
    RepresentationKey(bytes, False): Representation.SEMANTIC_BYTES,
    RepresentationKey(bytes, True): Representation.RAW_BYTES,
}


def plain_annotation(annotation: Any) -> Any:
    """`T` for an `Annotated[T, ...]`, else the annotation itself: what the engine runs.

    The metadata stays on the declaration, where tools read it.

    Example::

        plain_annotation(Annotated[float, Percent()])   # float
    """
    return annotation.__origin__ if get_origin(annotation) is Annotated else annotation


def hoist_metadata(annotation: Any) -> Any:
    """`Optional[Annotated[T, M]]` as the equivalent `Annotated[T | None, M]`, else unchanged.

    Declarations are canonicalised once so that `plain_annotation` and
    `metadata_of` each need to look in one place.
    """
    args = get_args(annotation)
    if get_origin(annotation) not in (Union, UnionType) or not any(get_origin(a) is Annotated for a in args):
        return annotation
    metadata = tuple(m for a in args if get_origin(a) is Annotated for m in a.__metadata__)
    return Annotated[(Union[tuple(plain_annotation(a) for a in args)], *metadata)]


def annotation_cache(fn):
    # A run asks these of the same annotations once per record, and they are pure. The key
    # is the plain type, so `Annotated` metadata a caller declared -- which may be a dict,
    # and unhashable -- never reaches a cache key.
    cached = lru_cache(maxsize=1024)(fn)

    @wraps(fn)
    def plain(annotation: Any, **kw: Any) -> Any:
        annotation = plain_annotation(annotation)
        try:
            return cached(annotation, **kw)
        except TypeError:
            # A param keeps the spelling it was declared with, so metadata a caller nested
            # inside `| None` can still reach here unhashable. The cache is an optimisation:
            # answer anyway.
            return fn(annotation, **kw)

    return plain


@annotation_cache
def columnar_item(annotation: Any) -> Any | None:
    """`Item` of a `Columnar[Item]` annotation, optional or not, else `None`."""
    if get_origin(annotation) is not Columnar and is_raw(annotation):
        annotation = raw_base(annotation)
    return get_args(annotation)[0] if get_origin(annotation) is Columnar else None


# Every run of a step reading `Columnar[Item]` or `Struct[Item]` asks for this; `get_type_hints` costs
# tens of microseconds.
@lru_cache(maxsize=None)
def item_schema(item: Any) -> tuple[tuple[str, Any], ...]:
    """`Item`'s fields in declaration order, as `(name, type)` pairs."""
    return tuple(get_type_hints(item).items())


def pydantic_ready(annotation: Any, _seen: tuple = ()) -> Any:
    """`annotation` with each `typing.TypedDict` in it rebuilt from `typing_extensions`, which pydantic takes on
    every Python; it refuses `typing`'s before 3.12.

    Example::

        TypeAdapter(list[pydantic_ready(Offer)]).validate_python([{"v": 1.0}])
    """
    if te.is_typeddict(annotation):
        if annotation in _seen:  # a TypedDict that holds itself stays as declared
            return annotation
        hints = te.get_type_hints(annotation, include_extras=True)
        ready = {k: pydantic_ready(v, (*_seen, annotation)) for k, v in hints.items()}
        if type(annotation).__module__ != "typing" and ready == hints:
            return annotation
        return te.TypedDict(annotation.__name__, {
            k: v if get_origin(v) in (te.Required, te.NotRequired) or k in annotation.__required_keys__
            else te.NotRequired[v] for k, v in ready.items()})
    args = get_args(annotation)
    ready = tuple(pydantic_ready(a, _seen) for a in args)
    if ready == args:
        return annotation
    origin = get_origin(annotation)
    if origin is Annotated:
        return Annotated[(ready[0], *annotation.__metadata__)]
    return Union[ready] if origin in (Union, UnionType) else origin[ready if len(ready) > 1 else ready[0]]


@annotation_cache
def is_raw(annotation: Any) -> bool:
    if annotation is Text or get_origin(annotation) is Columnar:
        return True
    return any(is_raw(arg) for arg in get_args(annotation) if arg is not type(None))


def text_as_str(annotation: Any) -> Any:
    # `Text` says how an input is read; a param or output holding one is the `str` it is.
    if annotation is Text:
        return str
    if get_origin(annotation) in (Union, UnionType) and Text in get_args(annotation):
        return Union[tuple(str if a is Text else a for a in get_args(annotation))]
    return annotation


@annotation_cache
def raw_base(annotation: Any) -> Any:
    if annotation is Text:
        return bytes
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return raw_base(args[0]) if args and is_raw(args[0]) else annotation


@annotation_cache
def representation_key(annotation: Any, *, row: bool = False) -> RepresentationKey:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    base = raw_base(args[0]) if len(args) == 1 else raw_base(annotation)
    raw = is_raw(annotation) or (row and base is bytes)
    return RepresentationKey(base, raw)


@annotation_cache
def representation_for(annotation: Any, *, row: bool = False) -> Representation:
    key = representation_key(annotation, row=row)
    try:
        return REPRESENTATIONS[key]
    except KeyError:
        raise TypeError(f"no representation for {key}") from None


def cast_values(name: str, value: Any, dtype: np.dtype, n: int) -> tuple[np.ndarray, np.ndarray | None]:
    raw = np.empty(n, object)
    try:
        raw[:] = value
    except ValueError:
        raise ValueError(f"can't set {name!r}: give one value or {n} values, not {value!r}") from None
    valid = np.array([v is not None for v in raw], bool)
    if dtype != object:
        try:
            # same_kind: 1.5 into an int column or "abc" into a float one is an error, not a silent truncation.
            raw = np.array([v if ok else 0 for v, ok in zip(raw, valid)]).astype(dtype, casting="same_kind")
        except (TypeError, ValueError):
            raise ValueError(f"can't set {name!r} to {value!r}: it doesn't cast to {dtype}") from None
    return raw, None if valid.all() else valid


class _NoDecimal:
    """Stands for Decimal until `decikit.ops.decimal` is imported, so nothing is one."""


def _decimals() -> tuple[type, type]:
    # Whatever annotates a value with a Decimal has imported `decikit.ops.decimal` already, and
    # importing it here would bring in numba, which `import decikit` must not.
    module = sys.modules.get("decikit.ops.decimal")
    return (module.Decimal, module.Currency) if module is not None else (_NoDecimal, _NoDecimal)


class DecimalInfo(NamedTuple):
    """What an annotation says about a `Decimal` value: its class, int64 scale and display units."""

    cls: type
    scale: int
    units: str | None
    precision: int = 19


def decimal_info(annotation: Any) -> DecimalInfo | None:
    """The `DecimalInfo` of `Decimal[p, s]`, `Currency` or `Currency["R"]` (optional or not), else `None`.

    A bare `Decimal` has no scale and raises `TypeError`.

    Example::

        decimal_info(Decimal[18, 2]).scale   # 2
    """
    try:
        return _decimal_info(annotation)
    except TypeError as e:
        if "unhashable" not in str(e):
            raise
        return _decimal_info.__wrapped__(annotation)


def _strip(annotation: Any) -> tuple[Any, DecimalSpec | None]:
    # The class under `Annotated[...]` and `| None`, and the DecimalSpec on the way.
    spec = None
    t = annotation
    for _ in range(3):
        if get_origin(t) is Annotated:
            spec = spec or next((m for m in t.__metadata__ if isinstance(m, DecimalSpec)), None)
            t = t.__origin__
        elif get_origin(t) in (Union, UnionType):
            args = [a for a in get_args(t) if a is not type(None)]
            if len(args) != 1:
                break
            t = args[0]
    return t, spec


def is_decimal(annotation: Any) -> bool:
    """Whether `annotation` is a Decimal or Currency, optional or not, with or without its scale.

    >>> is_decimal(int), is_decimal(Currency | None)
    (False, True)
    """
    try:
        return _is_decimal(annotation)
    except TypeError as e:
        if "unhashable" not in str(e):
            raise
        return _is_decimal.__wrapped__(annotation)


@lru_cache(maxsize=1024)
def _is_decimal(annotation: Any) -> bool:
    t, _ = _strip(annotation)
    return isinstance(t, type) and issubclass(t, _decimals()[0])


@lru_cache(maxsize=1024)
def _decimal_info(annotation: Any) -> DecimalInfo | None:
    t, spec = _strip(annotation)
    Decimal, Currency = _decimals()
    if not (isinstance(t, type) and issubclass(t, Decimal)):
        return None
    if spec is not None:
        return DecimalInfo(t, spec.scale, spec.units, spec.precision)
    if issubclass(t, Currency):
        return DecimalInfo(t, 4, None)
    raise TypeError("a Decimal annotation needs its precision and scale: write Decimal[18, 2], or use Currency")


def decimal_scale(annotation: Any) -> int | None:
    """The scale of a Decimal or Currency annotation, else `None`."""
    info = decimal_info(annotation)
    return None if info is None else info.scale


def decimal_boxer(annotation: Any) -> Any:
    """A function making the Python `Decimal` (or `Currency`) of a stored int64 for this annotation, else `None`."""
    info = decimal_info(annotation)
    if info is None:
        return None
    cls, scale, units, _ = info
    return lambda raw: cls.from_raw(raw, scale, units)


def to_raw(value: Any, scale: int) -> int:
    """The int64 a value is stored as at `scale`: a Decimal, int, float or str, rounded half-even.

    A step returning a float for a Decimal output is a `TypeError`, not a silent rounding.
    """
    Decimal = _decimals()[0]
    if isinstance(value, Decimal) or type(value) is int:
        return value.raw if getattr(value, "scale", None) == scale else Decimal(value, scale).raw
    raise TypeError(f"expected a Decimal or an int for a Decimal value, got {type(value).__name__}")


def to_raw_input(value: Any, scale: int) -> int:
    """`to_raw` for what a caller hands in: also a float, a str or a `decimal.Decimal`, by the construction rules."""
    Decimal = _decimals()[0]
    return to_raw(value, scale) if isinstance(value, (int, Decimal)) else Decimal(value, scale).raw



def raw_values(annotation: Any, value: Any) -> Any:
    """`value` (one value or a list of them, `None` for null) as the int64s a Decimal annotation stores; others unchanged."""
    scale = decimal_scale(annotation)
    if scale is None:
        return value
    if isinstance(value, list):
        return [None if x is None else to_raw_input(x, scale) for x in value]
    return None if value is None else to_raw_input(value, scale)
