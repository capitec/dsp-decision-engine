from __future__ import annotations

from enum import Enum
from typing import Any, Generic, TypeVar, get_args, get_origin

T = TypeVar("T")
_RAW_STR_CODES: dict[str, int] = {}


class Representation(Enum):
    """Compiled storage representations available for semantic values."""

    SEMANTIC_STRING = "semantic_string"
    SEMANTIC_BYTES = "semantic_bytes"
    RAW_STRING = "raw_string"
    RAW_BYTES = "raw_bytes"


class Raw(Generic[T]):
    """Marker annotation requesting Decider's internal representation of `T`."""


def is_raw(annotation: Any) -> bool:
    if get_origin(annotation) is Raw:
        return True
    return any(is_raw(arg) for arg in get_args(annotation) if arg is not type(None))


def raw_base(annotation: Any) -> Any:
    if get_origin(annotation) is not Raw:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return raw_base(args[0]) if args and is_raw(args[0]) else annotation
    args = get_args(annotation)
    return args[0] if is_raw(annotation) and args else annotation


def raw_str(value: str) -> int:
    """Return compiled code for a `Raw[str]` constant."""
    if not isinstance(value, str):
        raise TypeError("raw_str() requires a str")
    return _RAW_STR_CODES.setdefault(value, len(_RAW_STR_CODES))


def raw_string_codes() -> dict[str, int]:
    return dict(_RAW_STR_CODES)


