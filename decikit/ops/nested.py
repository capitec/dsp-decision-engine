"""How a list or struct column reaches a compiled step: `Columnar[Item]` and `Struct[Item]`."""
from __future__ import annotations

import typing
from typing import Any, Generic, TypeVar

__all__ = ["Columnar", "Struct"]

T = TypeVar("T")


class Columnar(Generic[T]):
    """Read a `list[dict]` column as one flat array per `Item` field, so a compiled step can loop over it.

    This is a columnar (struct-of-arrays) shape: `items.price` is one array of
    every item's `price`, read by item position `items.price[j]`. It is *not* a
    list of per-item dicts — `for i in items` iterates the *fields*, not the
    items, so index each field by position. An `Item` field may be `float`,
    `int`, `bool`, `str`, or `float | None` / `str | None` (a null reads as NaN
    / a null span); any other null raises, naming the item it is in. A field
    that is itself a list or struct is nested data, which `Columnar` cannot
    hold: annotate the input `list[dict]` instead (runs in Python; accept with
    `@allow_fallback`). A null or absent list follows the input's null policy,
    and `missing_as([])` reads it as a row with no items.

    Reach for it when the work per item is heavy or the lists are long: it
    costs about 20 us a `score()` call to build the arrays, so a short list
    with a one-line body is faster left as a plain `list[dict]` step.

    Example::

        class Item(TypedDict):
            price: float

        def total(items: Columnar[Item]) -> float:
            t = 0.0
            for j in range(len(items.price)):
                t += items.price[j]
            return t
    """


class Struct(Generic[T]):
    """Marker: a struct column of `Item`'s fields, as one record per row, read in a kernel."""


def struct_item(annotation: Any) -> Any | None:
    """`Item` of a `Struct[Item]` annotation, else `None`."""
    return typing.get_args(annotation)[0] if typing.get_origin(annotation) is Struct else None
