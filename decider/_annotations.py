"""Annotations users put on step arguments: what a value measures (`Money`, `Percent`, `Duration`) and how it is stored (`Columnar`, `Struct`).

A value's measure is declared with `Annotated`, so tools can show it as R 4000.00, 25% or 36 months."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class FieldMetadata:
    """What a step's input, output or param measures. Subclass it for a new kind.

    Attach it with `Annotated`; the engine ignores it and runs the plain type.
    The flow debugger formats values by it, and without it guesses from the
    name (and says it guessed).

    Example::

        from typing import Annotated
        from decider import Money, Percent

        def instalment(loan: Annotated[float, Money()], rate: Annotated[float, Percent()]) -> Annotated[float, Money()]:
            return loan * rate / 12
    """

    kind: ClassVar[str] = "value"

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, **dataclasses.asdict(self)}


@dataclass(frozen=True)
class Money(FieldMetadata):
    """A float amount of money, shown with `symbol`. Money kept exact is a `Currency`."""

    kind: ClassVar[str] = "money"
    symbol: str = "R"


@dataclass(frozen=True)
class DecimalSpec(FieldMetadata):
    """The precision and scale of a `Decimal[p, s]`, and the display `units` of a `Currency["R"]`."""

    kind: ClassVar[str] = "decimal"
    precision: int
    scale: int
    units: str | None = None


@dataclass(frozen=True)
class Percent(FieldMetadata):
    """A fraction shown as a percentage: 0.25 is 25%."""

    kind: ClassVar[str] = "percent"


@dataclass(frozen=True)
class Duration(FieldMetadata):
    """A length of time in `unit`s: `Duration("months")`."""

    kind: ClassVar[str] = "duration"
    unit: str = "months"


def metadata_of(annotation: Any) -> FieldMetadata | None:
    """The `FieldMetadata` in an `Annotated[...]` type, or `None`.

    Example::

        metadata_of(Annotated[float, Money()])   # Money(symbol='R')
    """
    if typing.get_origin(annotation) is not typing.Annotated:
        return None
    return next((m for m in annotation.__metadata__ if isinstance(m, FieldMetadata)), None)


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
