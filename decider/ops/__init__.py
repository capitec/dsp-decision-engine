"""The value types a step declares, and the functions that work on them, one module per domain.

- `text`: `Text`, a string a compiled step reads in place.
- `decimal`: `Decimal` and `Currency` fixed-point numbers, and rounding with an explicit mode.
- `nested`: `Columnar[Item]` and `Struct[Item]`, list and struct columns a compiled step reads.
- `measures`: `Percent`, `Duration` and `FieldMetadata`, what a value measures, for display.

Import a type from here, a function through its module::

    from decider.ops import Currency, Text, decimal
    decimal.round(x, 2, decimal.Rounding.HALF_EVEN)
"""
from __future__ import annotations

from typing import Any

from decider.ops.measures import Duration, FieldMetadata, Percent
from decider.ops.nested import Columnar, Struct
from decider.ops.text import Text

__all__ = ["Columnar", "Currency", "Decimal", "Duration", "FieldMetadata", "Percent", "Struct", "Text"]


def __getattr__(name: str) -> Any:
    # `decimal` brings in numba, which `import decider` must not.
    if name in ("Currency", "Decimal"):
        from decider.ops import decimal

        return getattr(decimal, name)
    raise AttributeError(f"module 'decider.ops' has no attribute {name!r}")
