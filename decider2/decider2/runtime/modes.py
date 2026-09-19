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
from decider2.compile.driver import Driver
from decider2.types import Step

Registry = dict  # name -> np.ndarray, plus "__valid__<name>" companions


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
    sig: "inspect.Signature",
    registry: Registry,
    resolved: ResolvedParams,
    i: int,
) -> dict:
    kwargs: dict[str, Any] = {}
    param_by_name = {d.name: d for d in step.params}
    for pname in sig.parameters:
        if pname == "params":
            kwargs["params"] = resolved.per_step_bundle[step.name]
        elif pname == "shared":
            kwargs["shared"] = resolved.shared
        elif pname in param_by_name:
            kwargs[pname] = resolved.per_step_scalar[(step.name, pname)]
        else:
            valid_key = f"__valid__{pname}"
            if valid_key in registry and not bool(registry[valid_key][i]):
                kwargs[pname] = None
            else:
                kwargs[pname] = registry[pname][i]
    return kwargs


def run_interpreted(
    steps: Sequence[Step], registry: Registry, resolved: ResolvedParams, n: int
) -> Registry:
    """Plain Python steps, plain Python driver — the reference semantics
    (doc 02 §3.1). No numba anywhere in this function."""
    for step in steps:
        sig = inspect.signature(step.fn)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = step.fn(**_row_kwargs(step, sig, registry, resolved, i))
        registry[step.name] = out
    return registry


def run_stepped(
    driver: Driver,
    steps: Sequence[Step],
    registry: Registry,
    resolved: ResolvedParams,
    n: int,
) -> Registry:
    """Njit'd steps (real compiled code), Python driver, one step at a time
    — driver-level step-through and production numerics without inlining
    (doc 02 §3.1). `driver.step_fns[name]` is the plain function instead for
    any step that didn't survive njit (doc 05 §6's fallback path, reused
    verbatim rather than duplicated)."""
    for step in steps:
        fn = driver.step_fns[step.name]
        sig = inspect.signature(step.fn)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = fn(**_row_kwargs(step, sig, registry, resolved, i))
        registry[step.name] = out
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
            args.append(resolved.per_step_scalar[(role.step_name, role.param_name)])
        elif role.kind == "params_bundle":
            args.append(resolved.per_step_bundle[role.step_name])
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
        if seg.kind == "fallback":
            step = seg.steps[0]
            fn = driver.step_fns[step.name]
            sig = inspect.signature(step.fn)
            out = np.empty(n, dtype=np.float64)
            for i in range(n):
                out[i] = fn(**_row_kwargs(step, sig, registry, resolved, i))
            registry[step.name] = out
            continue

        assert seg.plan is not None and seg.kernel_fn is not None
        out_arrays = {name: np.empty(n, dtype=np.float64) for name in seg.required_outputs}
        args = _build_call_args(seg.plan, registry, resolved, out_arrays)
        seg.kernel_fn(*args)
        for name, arr in out_arrays.items():
            registry[name] = arr
    return registry
