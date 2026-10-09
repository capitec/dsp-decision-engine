"""A frame column of dicts and lists no type describes: keys as text, one kind of value per field."""
from __future__ import annotations

from typing import Any, Sequence


def loosen(name: str, values: Sequence) -> list:
    """`values` with every dict key turned into text, so JSON, CSV and Arrow can carry them.

    Raises `ValueError` if one field holds different kinds of value (a number
    beside text): polars would turn the numbers into text. An int beside a
    float is fine, it widens. Declare a TypedDict to get the field checked
    where the step returns it.
    """
    kinds: dict[tuple, set[str]] = {}

    def walk(x: Any, path: tuple) -> Any:
        if x is None:
            return x
        if isinstance(x, dict):
            kinds.setdefault(path, set()).add("dict")
            return {str(k): walk(v, (*path, str(k))) for k, v in x.items()}
        if isinstance(x, list):
            kinds.setdefault(path, set()).add("list")
            return [walk(v, (*path, "[]")) for v in x]
        kinds.setdefault(path, set()).add("number" if isinstance(x, (int, float)) and not isinstance(x, bool)
                                          else type(x).__name__)
        return x

    out = [walk(x, ()) for x in values]
    for path, found in kinds.items():
        if len(found) > 1:
            field = ".".join(path) or "(the value)"
            raise ValueError(f"output '{name}', field {field}: holds {' and '.join(sorted(found))} in different rows; "
                             "return one type per field, or declare a TypedDict")
    return out
