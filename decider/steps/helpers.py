from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

_HELPER_SIGNATURES = "__decider_helper_signatures__"
_PYTHON_ONLY = "__decider_python_only__"


def helper(*, signatures: Iterable[tuple[tuple[type, ...], type]]) -> Callable:
    """Mark a function for compiled use with its supported input and output types.

    Example::

        @helper(signatures=[((float,), float), ((int,), int)])
        def discounted(price: float | int) -> float | int:
            return round(price * 0.9, 2)
    """
    declared = tuple((tuple(inputs), output) for inputs, output in signatures)
    if not declared:
        raise ValueError("helper() requires at least one signature")
    if any(not inputs or any(not isinstance(t, type) for t in inputs) or not isinstance(output, type)
           for inputs, output in declared):
        raise TypeError("helper signatures must be ((input types...), output type) pairs")

    def mark(fn: Callable) -> Callable:
        setattr(fn, _HELPER_SIGNATURES, declared)
        return fn

    return mark


def python_only(fn: Callable) -> Callable:
    """Mark a function as an intentional Python execution boundary."""
    setattr(fn, _PYTHON_ONLY, True)
    return fn


def helper_signatures(fn: Callable) -> tuple[tuple[tuple[type, ...], type], ...] | None:
    return getattr(fn, _HELPER_SIGNATURES, None)


def is_python_only(fn: Callable) -> bool:
    return bool(getattr(fn, _PYTHON_ONLY, False))
