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

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping

import polars as pl
from numba.core import event as _nb_event

from decider2.compile.driver import build_driver
from decider2.runtime.invoke import DEFAULT_BUILD_DIR

__all__ = [
    "assert_no_recompile",
    "count_new_compiles",
    "CompileCount",
    "assert_no_compilation_after_warmup",
]


@dataclass
class CompileCount:
    """How many GENUINE numba compilations happened while a
    `count_new_compiles()` block was open — not "how many functions got
    `@njit`-decorated" (decoration alone compiles nothing; numba is lazy by
    default) and not "how many cache-hit dispatches happened" (a dispatch
    served from an already-compiled specialisation, warm OR cold-but-cached,
    fires no `"numba:compile"` event at all — verified empirically while
    building this: a first call fires exactly one `START`/`END` pair per
    specialisation actually compiled; a repeat call with the same argument
    types fires zero). `.count` is the number of specialisations compiled;
    Stage 2's own acceptance test (this module's report) is `.count == 0`
    after `Pipeline.precompile()`.
    """

    count: int = 0
    kinds: list = field(default_factory=list)


@contextmanager
def count_new_compiles():
    """`with count_new_compiles() as counted: ...; counted.count` — the
    number of numba specialisations ACTUALLY compiled (not merely
    dispatched, cache-hit or otherwise) inside the block, via numba's own
    `"numba:compile"` event (`numba.core.dispatcher._Dispatcher.get_call_
    template`... in practice fired once per real `_compile_for_args`, the
    same place `NUMBA_DEBUG_CACHE` logging hooks into). This is `Pipeline.
    precompile()`'s own verification tool, and the acceptance test for the
    whole feature (this stage's report): a `precompile()` that still leaves
    something to compile on the real first request is worse than none,
    because it promises a guarantee it does not deliver.
    """
    result = CompileCount()
    with _nb_event.install_recorder("numba:compile") as recorder:
        yield result
    starts = [
        ev for ev in recorder.buffer if ev[1].status is _nb_event.EventStatus.START
    ]
    result.count = len(starts)
    result.kinds = [ev[1].data for ev in starts]


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


def assert_no_compilation_after_warmup(
    pipeline: Any,
    *,
    shared: Mapping[str, Any] | None = None,
    extra: "list[tuple[Any, ...]] | None" = None,
) -> "CompileCount":
    """Doc 05 §8's revised guarantee, as a first-class assertion: run
    `pipeline.precompile(shared=shared)`, then drive the pipeline again
    (`extra`, a list of `(frame_or_record, kwargs)` pairs — defaults to one
    more `apply()`/`score()` pair with the same synthetic row `precompile()`
    itself used) and assert that NOTHING compiles the second time.

    This is Stage 2's own acceptance test (this stage's report): a
    `precompile()` that still leaves something to compile on a real request
    is worse than none, because it promises a guarantee it does not
    deliver. Returns the `CompileCount` from the post-warm-up run, so a
    caller that wants the detail (not just the boolean) can have it.
    """
    pipeline.precompile(shared=shared)

    with count_new_compiles() as counted:
        if extra:
            for target, kwargs in extra:
                if isinstance(target, pl.DataFrame):
                    pipeline.apply(target, shared=shared, mode="fused", **kwargs)
                else:
                    pipeline.score(target, shared=shared, **kwargs)
        else:
            record = {inp.name: _warmup_value(inp) for inp in pipeline.interface.inputs}
            frame = pl.DataFrame({k: [v] for k, v in record.items()})
            pipeline.apply(frame, shared=shared, mode="fused")
            pipeline.score(record, shared=shared)

    if counted.count:
        raise AssertionError(
            "compilation happened AFTER precompile()/warm() "
            f"({counted.count} numba specialisation(s): {counted.kinds!r}) — "
            "precompile() missed a specialisation the real request path "
            "still needed. A warm-up that misses one is worse than none: it "
            "promises a guarantee it does not deliver."
        )
    return counted


def _warmup_value(inp: Any) -> Any:
    # 1/1.0, not 0/0.0 — see `decider2.graph.pipeline._dummy_value`'s
    # docstring: a zero denominator is common enough in a real pipeline
    # (a ratio) to raise `ZeroDivisionError` out of a warm-up call itself.
    if inp.annotation is bool:
        return False
    if inp.annotation is int:
        return 1
    if inp.annotation in (str, bytes):   # bytes: a tree's string feature (a span), fed as text
        return ""
    return 1.0
