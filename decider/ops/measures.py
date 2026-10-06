"""What a value measures, for tools to show it by: `Percent`, `Duration`, or your own `FieldMetadata`."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any, ClassVar

__all__ = ["DecimalSpec", "Duration", "FieldMetadata", "Percent", "metadata_of"]


@dataclass(frozen=True)
class FieldMetadata:
    """What a step's input, output or param measures. Subclass it for a new kind.

    Attach it with `Annotated`; the engine ignores it and runs the plain type.
    The flow debugger formats values by it, and without it guesses from the
    name (and says it guessed).

    Example::

        from typing import Annotated
        from decider.ops import Duration, Percent

        def instalment(rate: Annotated[float, Percent()], term: Annotated[int, Duration("months")]) -> float:
            return rate / term
    """

    kind: ClassVar[str] = "value"

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, **dataclasses.asdict(self)}


@dataclass(frozen=True)
class Percent(FieldMetadata):
    """A fraction shown as a percentage: 0.25 is 25%."""

    kind: ClassVar[str] = "percent"


@dataclass(frozen=True)
class Duration(FieldMetadata):
    """A length of time in `unit`s: `Duration("months")`."""

    kind: ClassVar[str] = "duration"
    unit: str = "months"


@dataclass(frozen=True)
class DecimalSpec(FieldMetadata):
    """The precision and scale of a `Decimal[p, s]`, and the display `units` of a `Currency["R"]`."""

    kind: ClassVar[str] = "decimal"
    precision: int
    scale: int
    units: str | None = None


def metadata_of(annotation: Any) -> FieldMetadata | None:
    """The `FieldMetadata` in an `Annotated[...]` type, or `None`.

    Example::

        metadata_of(Annotated[float, Percent()])   # Percent()
    """
    if typing.get_origin(annotation) is not typing.Annotated:
        return None
    return next((m for m in annotation.__metadata__ if isinstance(m, FieldMetadata)), None)
