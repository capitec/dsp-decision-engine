"""Infinite and NaN numbers across MCP, which carries plain JSON: spelled `"-Infinity"`, `"Infinity"`, `"NaN"`.

Without this a reply would show an open-ended bound as `null`, and a document read and
written back would silently change what its bounds mean.
"""
from __future__ import annotations

import functools
import math
import typing as t

_NUMBERS = {"-Infinity": -math.inf, "Infinity": math.inf, "NaN": math.nan}


def spell(x: t.Any) -> t.Any:
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if x != x else "Infinity" if x > 0 else "-Infinity"
    if isinstance(x, dict):
        return {k: spell(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [spell(v) for v in x]
    return x


def unspell(x: t.Any) -> t.Any:
    # ponytail: any argument string spelled exactly so is read as the number, so a String
    # column value "Infinity" can't be written through the tools; patch the file if one ever is.
    if isinstance(x, str):
        return _NUMBERS.get(x, x)
    if isinstance(x, dict):
        return {k: unspell(v) for k, v in x.items()}
    if isinstance(x, list):
        return [unspell(v) for v in x]
    return x


def spelled(fn: t.Callable) -> t.Callable:
    """`fn` taking and returning the spelled form; `functools.wraps` keeps its signature for the tool schema."""
    @functools.wraps(fn)
    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        return spell(fn(*unspell(list(args)), **unspell(kwargs)))
    return wrapper
