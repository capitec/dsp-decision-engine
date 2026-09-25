from __future__ import annotations

from typing import Any, Generic, TypeVar, get_args, get_origin

T = TypeVar("T")


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


