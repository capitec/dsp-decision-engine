from __future__ import annotations

import threading
from enum import Enum
from functools import lru_cache, wraps
from types import UnionType
from typing import Annotated, Any, Generic, NamedTuple, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")
# One table per process: a constant from `raw_str()` and a value first seen at runtime must
# never be given the same code, so both draw from here.
_RAW_STR_CODES: dict[Any, int] = {}
_NEW_CODE = threading.Lock()


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


class Struct(Generic[T]):
    """Marker: a struct column of `Item`'s fields, as one record per row, read in a kernel."""


def plain_annotation(annotation: Any) -> Any:
    """`T` for an `Annotated[T, ...]`, else the annotation itself: what the engine runs.

    The metadata stays on the declaration, where tools read it.

    Example::

        plain_annotation(Annotated[float, Money()])   # float
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
def rows_item(annotation: Any) -> Any | None:
    """`Item` of a `Rows[Item]` annotation, else `None`."""
    return get_args(annotation)[0] if get_origin(annotation) is Rows else None


def struct_item(annotation: Any) -> Any | None:
    """`Item` of a `Struct[Item]` annotation, else `None`."""
    return get_args(annotation)[0] if get_origin(annotation) is Struct else None


def rows_schema(item: Any) -> tuple[tuple[str, Any], ...]:
    """`Item`'s fields in declaration order, as `(name, type)` pairs."""
    return tuple(get_type_hints(item).items())


@annotation_cache
def is_raw(annotation: Any) -> bool:
    if get_origin(annotation) in (Raw, Rows):
        return True
    return any(is_raw(arg) for arg in get_args(annotation) if arg is not type(None))


@annotation_cache
def raw_base(annotation: Any) -> Any:
    if get_origin(annotation) is not Raw:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return raw_base(args[0]) if args and is_raw(args[0]) else annotation
    args = get_args(annotation)
    return args[0] if is_raw(annotation) and args else annotation


@annotation_cache
def representation_key(annotation: Any, *, row: bool = False) -> RepresentationKey:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    base = raw_base(args[0]) if len(args) == 1 else raw_base(annotation)
    raw = is_raw(annotation) or (row and base in (str, bytes))
    return RepresentationKey(base, raw)


@annotation_cache
def representation_for(annotation: Any, *, row: bool = False) -> Representation:
    key = representation_key(annotation, row=row)
    try:
        return REPRESENTATIONS[key]
    except KeyError:
        raise TypeError(f"no representation for {key}") from None


def string_code(value: Any) -> int:
    """The `Raw[str]` code of `value`, assigned in first-seen order; `None` gets one of its own."""
    code = _RAW_STR_CODES.get(value)
    if code is None:
        with _NEW_CODE:
            code = _RAW_STR_CODES.setdefault(value, len(_RAW_STR_CODES))
    return code


def raw_str(value: str) -> int:
    """Return compiled code for a `Raw[str]` constant.

    Example::

        PRIVATE = raw_str("private")

        def is_private(sector: Raw[str]) -> bool:
            return sector == PRIVATE
    """
    if not isinstance(value, str):
        raise TypeError("raw_str() requires a str")
    return string_code(value)


