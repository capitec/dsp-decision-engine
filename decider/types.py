from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any, Generic, NamedTuple, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")
_RAW_STR_CODES: dict[str, int] = {}


class Representation(Enum):
    """Compiled storage representations available for semantic values."""

    SEMANTIC_STRING = "semantic_string"
    SEMANTIC_BYTES = "semantic_bytes"
    RAW_STRING = "raw_string"
    RAW_BYTES = "raw_bytes"


class RepresentationKey(NamedTuple):
    base_type: type
    raw_annotation: bool


REPRESENTATIONS: dict[RepresentationKey, Representation] = {
    RepresentationKey(str, False): Representation.SEMANTIC_STRING,
    RepresentationKey(str, True): Representation.RAW_STRING,
    RepresentationKey(bytes, False): Representation.SEMANTIC_BYTES,
    RepresentationKey(bytes, True): Representation.RAW_BYTES,
}


class Raw(Generic[T]):
    """Marker annotation requesting Decider's internal representation of `T`."""


class Rows(Generic[T]):
    """Read a `list[dict]` column as one array per `Item` field, so a compiled step can loop over it.

    An `Item` field may be `float`, `int`, `bool` or `float | None` (a null
    reads as NaN); any other null raises, naming the item it is in. A null or
    absent list follows the input's null policy, and `missing_as([])` reads it
    as a row with no items.

    Reach for it when the work per item is heavy or the lists are long: it
    costs about 20 us a `score()` call to build the arrays, so a short list
    with a one-line body is faster left as a plain `list[dict]` step.

    Example::

        class Item(TypedDict):
            price: float

        def total(items: Rows[Item]) -> float:
            t = 0.0
            for j in range(len(items.price)):
                t += items.price[j]
            return t
    """


def rows_item(annotation: Any) -> Any | None:
    """`Item` of a `Rows[Item]` annotation, optional or not, else `None`."""
    if get_origin(annotation) is not Rows and is_raw(annotation):
        annotation = raw_base(annotation)
    return get_args(annotation)[0] if get_origin(annotation) is Rows else None


# Every run of a step reading `Rows[Item]` asks for this; `get_type_hints` costs tens of microseconds.
@lru_cache(maxsize=None)
def rows_schema(item: Any) -> tuple[tuple[str, Any], ...]:
    """`Item`'s fields in declaration order, as `(name, type)` pairs."""
    return tuple(get_type_hints(item).items())


def is_raw(annotation: Any) -> bool:
    if get_origin(annotation) in (Raw, Rows):
        return True
    return any(is_raw(arg) for arg in get_args(annotation) if arg is not type(None))


def raw_base(annotation: Any) -> Any:
    if get_origin(annotation) is not Raw:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return raw_base(args[0]) if args and is_raw(args[0]) else annotation
    args = get_args(annotation)
    return args[0] if is_raw(annotation) and args else annotation


def representation_key(annotation: Any, *, row: bool = False) -> RepresentationKey:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    base = raw_base(args[0]) if len(args) == 1 else raw_base(annotation)
    raw = is_raw(annotation) or (row and base in (str, bytes))
    return RepresentationKey(base, raw)


def representation_for(annotation: Any, *, row: bool = False) -> Representation:
    key = representation_key(annotation, row=row)
    try:
        return REPRESENTATIONS[key]
    except KeyError:
        raise TypeError(f"no representation for {key}") from None


def raw_str(value: str) -> int:
    """Return compiled code for a `Raw[str]` constant."""
    if not isinstance(value, str):
        raise TypeError("raw_str() requires a str")
    return _RAW_STR_CODES.setdefault(value, len(_RAW_STR_CODES))


def raw_string_codes() -> dict[str, int]:
    return dict(_RAW_STR_CODES)


