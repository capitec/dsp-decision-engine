from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

_HELPER_SIGNATURES = "__decider_helper_signatures__"
_ALLOW_FALLBACK = "__decider_allow_fallback__"


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


def allow_fallback(fn: Any) -> Any:
    """Accept how this step runs: no `FallbackWarning`, and `strict_compile=True` allows it.

    On a step it says the author accepts a step compiled modes can't put in a
    kernel; `Executable.fallbacks()` still reports it, marked
    `@allow_fallback`. On a function a step calls it marks an intentional
    Python boundary, so the step runs in Python rather than compiling it.

    Example::

        @allow_fallback
        def order_total(items: list[dict]) -> float:
            return sum(item["price"] for item in items)
    """
    from decider.steps.base import Step
    from decider.steps.function import FunctionStep

    target = fn.fn if isinstance(fn, FunctionStep) else fn
    if isinstance(target, Step) or not callable(target):
        raise TypeError(
            f"@allow_fallback takes a function, or step() of one, not {type(fn).__name__}. A tree, table, "
            "scorecard, branch or loop falls back as a whole: filter decider.FallbackWarning instead."
        )
    setattr(target, _ALLOW_FALLBACK, True)
    return fn


def helper_signatures(fn: Callable) -> tuple[tuple[tuple[type, ...], type], ...] | None:
    return getattr(fn, _HELPER_SIGNATURES, None)


def allows_fallback(fn: Any) -> bool:
    return bool(getattr(fn, _ALLOW_FALLBACK, False))
