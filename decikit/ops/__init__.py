"""The value types a step declares, and the functions that work on them, one module per domain.

- `text`: `Text`, a string a compiled step reads in place.
- `decimal`: `Decimal` and `Currency` fixed-point numbers, and rounding with an explicit mode.
- `nested`: `Columnar[Item]` and `Struct[Item]`, list and struct columns a compiled step reads.
- `measures`: `Percent`, `Duration` and `FieldMetadata`, what a value measures, for display.

Import a type from here, a function through its module::

    from decikit.ops import Currency, Text, decimal
    decimal.round(x, 2, decimal.Rounding.HALF_EVEN)
"""
from __future__ import annotations

from typing import Any

from decikit.ops.measures import Duration, FieldMetadata, Percent
from decikit.ops.nested import Columnar, Struct
from decikit.ops.text import Text

__all__ = ["Columnar", "Currency", "Decimal", "Duration", "FieldMetadata", "Percent", "Struct", "Text"]


def __getattr__(name: str) -> Any:
    # `decimal` brings in numba, which `import decikit` must not.
    if name in ("Currency", "Decimal"):
        from decikit.ops import decimal

        return getattr(decimal, name)
    raise AttributeError(f"module 'decikit.ops' has no attribute {name!r}")
