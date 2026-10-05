from __future__ import annotations

from enum import Enum
from functools import lru_cache, wraps
from types import UnionType
from typing import Annotated, Any, NamedTuple, Union, get_args, get_origin, get_type_hints

from decider._annotations import Columnar, Raw, Struct


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
def columnar_item(annotation: Any) -> Any | None:
    """`Item` of a `Columnar[Item]` annotation, optional or not, else `None`."""
    if get_origin(annotation) is not Columnar and is_raw(annotation):
        annotation = raw_base(annotation)
    return get_args(annotation)[0] if get_origin(annotation) is Columnar else None


def struct_item(annotation: Any) -> Any | None:
    """`Item` of a `Struct[Item]` annotation, else `None`."""
    return get_args(annotation)[0] if get_origin(annotation) is Struct else None


# Every run of a step reading `Columnar[Item]` or `Struct[Item]` asks for this; `get_type_hints` costs
# tens of microseconds.
@lru_cache(maxsize=None)
def item_schema(item: Any) -> tuple[tuple[str, Any], ...]:
    """`Item`'s fields in declaration order, as `(name, type)` pairs."""
    return tuple(get_type_hints(item).items())


@annotation_cache
def is_raw(annotation: Any) -> bool:
    if get_origin(annotation) in (Raw, Columnar):
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
