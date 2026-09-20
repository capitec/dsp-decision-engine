"""`assert_no_recompile` — the property the entire config story rests on
(doc 08 §2), as a first-class, reusable assertion.

Doc 05 §9 acceptance criterion 5, verbatim: "Retuning any params bundle
leaves `driver.signatures` at length 1; changing a field's *type* adds one
(negative control)." A param is resolved to a plain kernel *argument*, never
baked into generated source (doc 05 §4.2) — so a retune has nothing for
numba to recompile against, and `compile.driver.build_driver`'s in-process
cache is keyed by structure (which steps, in which order, in which fuse()
groups, materialising which names) and never by params at all
(`compile/driver.py`'s own docstring). This asserts the general form of that
guarantee against any pipeline, not just the framework's own flagship
fixture.
"""
from __future__ import annotations

from typing import Any, Mapping

import polars as pl

from decider2.compile.driver import build_driver
from decider2.runtime.invoke import DEFAULT_BUILD_DIR


def _driver_for(pipeline: Any) -> Any:
    """Rebuild the `Driver` that `pipeline.apply()` itself is using.

    Safe to call freestanding, without racing a real compile of its own:
    `build_driver`'s cache is keyed by structure and `DEFAULT_BUILD_DIR` is
    the only build directory `Pipeline.apply` ever uses (it has no
    `build_dir=` of its own to override it with) — so as long as
    `pipeline.apply(...)` already ran at least once, this returns the exact
    same cached object rather than triggering a fresh build.

    Simplification, matching the equivalent helper independently written for
    `tests/test_runtime_serve.py`: terminal names are taken from
    `pipeline.interface.terminals` alone, not from any `.emit()` on top of
    it. A pipeline that relies on `.emit()` to expose extra names may see
    this rebuild a *different* driver than `.apply()` actually used — a
    false failure, not a missed recompile — so pass an emit-free pipeline
    (or emit the same names some other way) when using this function.
    """
    steps, group_ids, owners, _ = pipeline.flatten_for_runtime()
    terminal_names = frozenset(pipeline.interface.terminals)
    return build_driver(
        list(steps), list(group_ids), owners=list(owners),
        build_dir=DEFAULT_BUILD_DIR, terminal_names=terminal_names,
    )


def assert_no_recompile(
    pipeline: Any,
    frame: pl.DataFrame,
    params_a: Mapping[str, Mapping[str, Any]] | None,
    params_b: Mapping[str, Mapping[str, Any]] | None,
    *,
    shared: Mapping[str, Any] | None = None,
) -> None:
    """Run `pipeline.apply(frame, params=params_a)`, then `params=params_b`,
    and assert the retune left the compiled driver untouched: same
    `driver.signatures` length, and — stronger, and the literal claim doc 08
    §2's config story rests on — the exact same `Driver` object, not a
    rebuilt one that merely happens to have an equal signature count.

    `params_a`/`params_b` must be a *value*-only retune (same fields, same
    declared types, different numbers) — that is what this function checks
    holds. Doc 05 §9 criterion 5's own negative control is changing a
    field's declared *type*, which correctly recompiles (grows
    `driver.signatures` by one); passing such a pair here will correctly
    fail this assertion rather than silently pass it, since a type change
    is precisely a structural change the driver-cache key is NOT indifferent
    to.
    """
    pipeline.apply(frame, params=params_a, shared=shared)
    driver_before = _driver_for(pipeline)
    before = len(driver_before.signatures)

    pipeline.apply(frame, params=params_b, shared=shared)
    driver_after = _driver_for(pipeline)
    after = len(driver_after.signatures)

    if after != before:
        raise AssertionError(
            "retuning params recompiled the driver (doc 05 §9 criterion 5, "
            f"doc 08 §2): driver.signatures went from {before} to {after} "
            f"between params_a={dict(params_a or {})!r} and "
            f"params_b={dict(params_b or {})!r}."
        )
    if driver_after is not driver_before:
        raise AssertionError(
            "retuning params rebuilt the driver (doc 08 §2's 'no compile' "
            "claim): driver.signatures length was unchanged "
            f"({before}), but params_b={dict(params_b or {})!r} produced a "
            "new Driver object rather than reusing the one params_a="
            f"{dict(params_a or {})!r} built — build_driver's cache key "
            "must be structural, never params-dependent."
        )
