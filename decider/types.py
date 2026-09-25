from __future__ import annotations

from enum import Enum
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
    """Marker: `Item`'s list column, as one array per field, sliced per parent row."""


def rows_item(annotation: Any) -> Any | None:
    """`Item` of a `Rows[Item]` annotation, else `None`."""
    return get_args(annotation)[0] if get_origin(annotation) is Rows else None


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


