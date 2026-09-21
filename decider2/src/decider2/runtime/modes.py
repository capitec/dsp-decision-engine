"""The three execution modes over an already-built `Driver` — doc 02 §3.1.

| mode          | steps            | driver              |
|---------------|------------------|----------------------|
| `interpreted` | Python           | Python               |
| `stepped`     | njit, individual | Python, one at a time|
| `fused`       | njit, inlined    | njit                 |

All three walk the *same* flat, already-ordered step list and the same
name -> array registry, so a divergence between two adjacent rungs localises
to exactly the layer between them (doc 02 §3.1's equivalence ladder):
`interpreted != stepped` means numba changed a step's semantics; `stepped !=
fused` means fusion/inlining did. `stepped` reuses the *fallback* mechanism's
step callables verbatim (doc 05 §6) — a step that didn't survive njit is
called as plain Python in both, which is why doc 02 §3.1 calls `stepped`
"structurally identical to the fallback path" rather than a separate thing
to maintain.

Every runner takes `n` (row count) explicitly rather than inferring it from
an array, because `score()` (doc 02 §3.5) drives the exact same runners with
`n == 1` and no polars in sight — "the same kernel answers a single record"
is doc 05 §9's acceptance criterion 2, not a separate code path.

**`fused` runs segments, not steps.** `run_fused` below just calls `seg.run(
registry, resolved, n)` for each of `driver.segments` in order — a
`CompiledSegment`/`FallbackSegment` (`decider2.compile.driver`) owns how it
runs itself, including the per-row calling convention (`ResolvedParams`,
`_row_kwargs`, ...), which is why this module imports those back from there
rather than defining a second copy: `run_interpreted`/`run_stepped` still
need the very same convention to drive a step one at a time, outside any
`Segment`, for the other two rungs of the equivalence ladder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np

from decider2.compile.driver import (
    Driver,
    ResolvedParams,
    _return_dtype,
    _row_kwargs,
    _signature,
    build_driver,
)
from decider2.types import Step

Registry = dict  # name -> np.ndarray, plus "__valid__<name>" companions

_MISSING = object()


def _step_fn(driver: Driver, owner: str | None, step_name: str) -> Any:
    if owner is not None:
        fn = driver.step_fns.get((owner, step_name), _MISSING)
        if fn is not _MISSING:
            return fn
    return driver.step_fns[step_name]


def run_interpreted(
    steps: Sequence[Step],
    registry: Registry,
    resolved: ResolvedParams,
    n: int,
    *,
    owners: "Sequence[str] | None" = None,
) -> Registry:
    """Plain Python steps, plain Python driver — the reference semantics
    (doc 02 §3.1). No numba anywhere in this function."""
    owners = list(owners) if owners is not None else [s.name for s in steps]
    for step, owner in zip(steps, owners):
        sig = _signature(step.fn)
        out = np.empty(n, dtype=_return_dtype(step.fn))
        for i in range(n):
            out[i] = step.fn(**_row_kwargs(step, owner, sig, registry, resolved, i))
        registry[step.name] = out
        # Doc 03 §3.3/§7: every version of a waterfall value stays reachable
        # by its producing module's name, alongside the plain "live" (most
        # recent) value the next step's wiring actually reads.
        registry[f"{step.name}@{owner}"] = out
    return registry


def run_stepped(
    driver: Driver,
    steps: Sequence[Step],
    registry: Registry,
    resolved: ResolvedParams,
    n: int,
    *,
    owners: "Sequence[str] | None" = None,
) -> Registry:
    """Njit'd steps (real compiled code), Python driver, one step at a time
    — driver-level step-through and production numerics without inlining
    (doc 02 §3.1). `driver.step_fns[(owner, name)]` is the plain function
    instead for any step that didn't survive njit (doc 05 §6's fallback
    path, reused verbatim rather than duplicated)."""
    owners = list(owners) if owners is not None else [s.name for s in steps]
    for step, owner in zip(steps, owners):
        fn = _step_fn(driver, owner, step.name)
        sig = _signature(step.fn)
        out = np.empty(n, dtype=_return_dtype(step.fn))
        for i in range(n):
            out[i] = fn(**_row_kwargs(step, owner, sig, registry, resolved, i))
        registry[step.name] = out
        registry[f"{step.name}@{owner}"] = out
    return registry


def run_fused(driver: Driver, registry: Registry, resolved: ResolvedParams, n: int) -> Registry:
    """Njit'd steps inlined into an njit driver — production, batch and
    realtime (doc 02 §3.1). Segments run in order, each one running itself
    (`decider2.compile.driver.Segment.run`) and threading numpy arrays
    through `registry`: a value a later segment needs is looked up there
    regardless of whether it came from the original frame or an earlier
    segment's own output — "intermediates between two kernels stay in numpy,
    they do not round-trip through polars" (doc 05 §7).

    A `FallbackSegment` (doc 05 §6) is exactly one step, run row-by-row in
    plain Python — the *only* place in `fused` mode Python executes, and
    only for the node that could not be compiled, never its neighbours.
    """
    for seg in driver.segments:
        seg.run(registry, resolved, n)
    return registry


# ---------------------------------------------------------------------------
# The three modes as a registry of classes — doc 02 §3.1. `runtime.invoke.
# _run` looks a caller's `mode=` string up in `MODES` once, instead of an
# if/elif ladder over "interpreted"/"stepped"/"fused": adding a fourth mode
# is one class here (each one owning whether/how it builds a `Driver` and
# which runner above it drives), never a new branch anywhere else.
# ---------------------------------------------------------------------------


class Mode(ABC):
    """One rung of doc 02 §3.1's equivalence ladder. `.run()` takes the
    same arguments `runtime.invoke._run` used to branch on `mode ==
    "..."` for, so a caller of `MODES[name].run(...)` cannot tell it isn't
    still one function — it's registered as a class only so a new mode
    doesn't touch this dispatch again."""

    name: ClassVar[str]

    @staticmethod
    @abstractmethod
    def run(
        steps: Sequence[Step],
        group_ids: Sequence[int],
        owners: Sequence[str],
        registry: Registry,
        resolved: ResolvedParams,
        n: int,
        *,
        terminal_names: frozenset,
        build_dir: "str | Path",
    ) -> Registry: ...


class Interpreted(Mode):
    """Plain Python steps, plain Python driver (doc 02 §3.1) — the
    reference semantics. Never builds a `Driver`: nothing here is
    compiled, so there is nothing to compile ahead of running it."""

    name: ClassVar[str] = "interpreted"

    @staticmethod
    def run(steps, group_ids, owners, registry, resolved, n, *, terminal_names, build_dir):
        del group_ids, terminal_names, build_dir  # unused: no Driver to build
        return run_interpreted(steps, registry, resolved, n, owners=owners)


class Stepped(Mode):
    """Njit'd steps (real compiled code), Python driver, one step at a
    time — driver-level step-through and production numerics without
    inlining (doc 02 §3.1)."""

    name: ClassVar[str] = "stepped"

    @staticmethod
    def run(steps, group_ids, owners, registry, resolved, n, *, terminal_names, build_dir):
        driver = build_driver(
            list(steps), list(group_ids), owners=list(owners),
            build_dir=build_dir, terminal_names=terminal_names,
        )
        return run_stepped(driver, steps, registry, resolved, n, owners=owners)


class Fused(Mode):
    """Njit'd steps inlined into an njit driver — production, batch and
    realtime (doc 02 §3.1)."""

    name: ClassVar[str] = "fused"

    @staticmethod
    def run(steps, group_ids, owners, registry, resolved, n, *, terminal_names, build_dir):
        driver = build_driver(
            list(steps), list(group_ids), owners=list(owners),
            build_dir=build_dir, terminal_names=terminal_names,
        )
        return run_fused(driver, registry, resolved, n)


MODES: dict[str, type[Mode]] = {cls.name: cls for cls in (Interpreted, Stepped, Fused)}
