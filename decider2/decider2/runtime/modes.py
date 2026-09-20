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
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from decider2.compile import codegen
from decider2.compile.driver import Driver, numpy_dtype
from decider2.types import Step

Registry = dict  # name -> np.ndarray, plus "__valid__<name>" companions

_MISSING = object()


def _signature(fn) -> "inspect.Signature":
    """`inspect.signature`, with the same `eval_str=True`-then-fall-back
    `decider2.params.harvest_signature` already uses: a step's module using
    `from __future__ import annotations` stringifies every annotation, and
    without this a `-> int`/`-> bool` return type would compare equal to
    nothing in `numpy_dtype`'s table and silently fall back to float64 —
    exactly the bug this module exists to close."""
    try:
        return inspect.signature(fn, eval_str=True)
    except (NameError, TypeError):
        return inspect.signature(fn)


def _return_dtype(fn) -> np.dtype:
    """Doc 00 §2 / doc 03 §1 / doc 05 §9 criterion 4: a step's declared
    return annotation decides the array dtype it is written into, so an
    `-> int`/`-> bool` step survives the boundary as its own dtype instead
    of being forced onto float64 (which silently degrades an Int64 above
    2**53, and cannot hold a Boolean at all)."""
    return numpy_dtype(_signature(fn).return_annotation)


def _scalar_arg(resolved: "ResolvedParams", owner: str | None, step_name: str, param_name: str) -> Any:
    """`(owner, step_name, param_name)` first — the collision-proof key
    `resolve_params` now writes (doc 03 §4.1/§10: a step's OUTPUT name is
    not unique across modules, the waterfall idiom, doc 03 §3.2). Falls
    back to the plain `(step_name, param_name)` key for a caller that built
    a `ResolvedParams` by hand without an owner in mind — every such caller
    in this codebase has no name collision to begin with, so the fallback
    is exact, not approximate."""
    if owner is not None:
        value = resolved.per_step_scalar.get((owner, step_name, param_name), _MISSING)
        if value is not _MISSING:
            return value
    return resolved.per_step_scalar[(step_name, param_name)]


def _bundle_arg(resolved: "ResolvedParams", owner: str | None, step_name: str) -> Any:
    if owner is not None:
        value = resolved.per_step_bundle.get((owner, step_name), _MISSING)
        if value is not _MISSING:
            return value
    return resolved.per_step_bundle[step_name]


def _step_fn(driver: Driver, owner: str | None, step_name: str) -> Any:
    if owner is not None:
        fn = driver.step_fns.get((owner, step_name), _MISSING)
        if fn is not _MISSING:
            return fn
    return driver.step_fns[step_name]


@dataclass(frozen=True)
class ResolvedParams:
    """One validated bundle per invocation (doc 03 §4), already converted to
    the shapes a step call needs — never a dict a step reads by string key,
    which would defeat static typing under numba.

    - `per_step_scalar[(step_name, param_name)]` — an individual
      `param()`-declared field (doc 03 §4.4): the function is called with
      this as an ordinary argument, exactly as a direct call would be.
    - `per_step_bundle[step_name]` — a step with a bare `params` argument
      (doc 03 §4.2) gets one NamedTuple.
    - `shared` — the single reserved bundle (doc 03 §4.2), or `None` if
      nothing in this pipeline reads it.
    """

    per_step_scalar: dict
    per_step_bundle: dict
    shared: Any | None = None


def _row_kwargs(
    step: Step,
    owner: "str | None",
    sig: "inspect.Signature",
    registry: Registry,
    resolved: ResolvedParams,
    i: int,
) -> dict:
    kwargs: dict[str, Any] = {}
    param_by_name = {d.name: d for d in step.params}
    for pname in sig.parameters:
        if pname == "params":
            kwargs["params"] = _bundle_arg(resolved, owner, step.name)
        elif pname == "shared":
            kwargs["shared"] = resolved.shared
        elif pname in param_by_name:
            kwargs[pname] = _scalar_arg(resolved, owner, step.name, pname)
        else:
            valid_key = f"__valid__{pname}"
            if valid_key in registry and not bool(registry[valid_key][i]):
                kwargs[pname] = None
            else:
                kwargs[pname] = registry[pname][i]
    return kwargs


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


def _build_call_args(
    plan: codegen.KernelPlan, registry: Registry, resolved: ResolvedParams, out_arrays: dict
) -> list:
    args: list = []
    for role in codegen.kernel_signature(plan):
        if role.kind == "array":
            args.append(registry[role.input_name])
        elif role.kind == "valid":
            args.append(registry[f"__valid__{role.input_name}"])
        elif role.kind == "param_scalar":
            args.append(_scalar_arg(resolved, role.owner, role.step_name, role.param_name))
        elif role.kind == "params_bundle":
            args.append(_bundle_arg(resolved, role.owner, role.step_name))
        elif role.kind == "shared":
            args.append(resolved.shared)
        elif role.kind == "output":
            args.append(out_arrays[role.output_name])
        else:  # pragma: no cover - exhaustive over ArgRole.kind
            raise AssertionError(f"unhandled arg role {role.kind!r}")
    return args


def run_fused(driver: Driver, registry: Registry, resolved: ResolvedParams, n: int) -> Registry:
    """Njit'd steps inlined into an njit driver — production, batch and
    realtime (doc 02 §3.1). Segments run in order, threading numpy arrays
    through `registry`: a value a later segment needs is looked up there
    regardless of whether it came from the original frame or an earlier
    segment's own output — "intermediates between two kernels stay in numpy,
    they do not round-trip through polars" (doc 05 §7).

    A `fallback` segment (doc 05 §6) is exactly one step, run row-by-row in
    plain Python — the *only* place in `fused` mode Python executes, and
    only for the node that could not be compiled, never its neighbours.
    """
    for seg in driver.segments:
        owner_by_name = dict(zip((s.name for s in seg.steps), seg.owners)) if seg.owners else {}

        if seg.kind == "fallback":
            step = seg.steps[0]
            owner = seg.owners[0] if seg.owners else step.name
            fn = _step_fn(driver, owner, step.name)
            sig = _signature(step.fn)
            out = np.empty(n, dtype=_return_dtype(step.fn))
            for i in range(n):
                out[i] = fn(**_row_kwargs(step, owner, sig, registry, resolved, i))
            registry[step.name] = out
            registry[f"{step.name}@{owner}"] = out
            continue

        assert seg.plan is not None and seg.kernel_fn is not None
        step_by_name = {s.name: s for s in seg.steps}
        out_arrays = {
            name: np.empty(n, dtype=_return_dtype(step_by_name[name].fn))
            for name in seg.required_outputs
        }
        args = _build_call_args(seg.plan, registry, resolved, out_arrays)
        seg.kernel_fn(*args)
        for name, arr in out_arrays.items():
            registry[name] = arr
            registry[f"{name}@{owner_by_name.get(name, name)}"] = arr
    return registry
