"""What a value measures, declared with `Annotated`, so tools can show it as R 4000.00, 25% or 36 months."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any, ClassVar


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
    """An amount of money: `Money()` for rands, `Money(cents=True)` for an int of cents."""

    kind: ClassVar[str] = "money"
    symbol: str = "R"
    cents: bool = False


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

        metadata_of(Annotated[float, Money()])   # Money(symbol='R', cents=False)
    """
    if typing.get_origin(annotation) is not typing.Annotated:
        return None
    return next((m for m in annotation.__metadata__ if isinstance(m, FieldMetadata)), None)
