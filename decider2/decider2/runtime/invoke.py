"""apply() and score() — the two entry points, one kernel (doc 02 §3.5).

**Contract with the graph layer.** `decider2.graph.pipeline.Pipeline.apply`/
`.score` (and `Module.apply`/`.score`, which wrap a standalone module as a
one-element pipeline) call straight into this module's `apply`/`score` with
an already-flattened, already-ordered `steps` tuple, a `group_ids` tuple (one
id per step, doc 05 §7's "one kernel per module" — each module's steps share
an id), the pipeline's inferred `interface`, plain (unqualified) `emit`
names, `drop` names, and a `ParamSpace` per module instance (see
`Pipeline.flatten_for_runtime`), which is what makes doc 03 §10's
"namespaced by module instance name" true for a multi-step module. This module does not import
`decider2.graph` itself — it only needs the `Step`/`Interface` seams from
`decider2.types` — which keeps the dependency direction one-way and this
layer buildable/testable before or without the graph layer.

`apply()`/`score()` share one code path below `_run`: `score()` wraps its
single record into 1-row arrays and drives the exact same segments `apply()`
does — "the same kernel answers a single record with no polars involvement"
is doc 05 §9's acceptance criterion 2, not a separate fast path to keep in
sync by hand.

**Batch extraction/write-back defers to `decider2.boundary`** (doc 02 §6) —
`extract_frame` for the dtype ladder, null routing and per-column extraction
(doc 05 §1-§2), `write_back`/`KernelOutputs` for the dtype-grouped,
layout-per-entry-point output convention (doc 05 §3.1). This module's job is
the glue: resolving params (doc 03 §4), driving the three modes, and
reconciling a `NullRouting`'s routed-away rows back into a full-length frame
(doc 03 §1: "a null must produce a decision, not an exception" — decided
here only as far as "don't silently drop the row"; rendering the decision
itself is `observe/`'s job, doc 00-BUILD.md Layer 5, not built yet).

`score()`'s single-record extraction has no boundary-layer equivalent to
call into — `decider2.boundary/__init__.py`'s own docstring says whole-row
bulk request marshalling (doc 05 §3.1b) "is not part of this package's
surface" — so it stays a **minimal reference implementation** here, marked
below, mirroring `decider2.boundary.nulls`' REQUIRED-null routing semantics
by hand rather than reimplementing the frame-shaped functions over one row.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from decider2.compile.driver import build_driver
from decider2.runtime import modes
from decider2.runtime.modes import ResolvedParams
from decider2.types import Decision, Input, Interface, MissingInputPolicy, NullPolicy, Step

DEFAULT_BUILD_DIR = Path(".decider2_cache")

_VALID_MODES = ("interpreted", "stepped", "fused")


# ---------------------------------------------------------------------------
# Params resolution (doc 03 §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpace:
    """One module instance's params namespace — the unit doc 03 §10 says
    overrides are keyed by: "Params are namespaced by module instance name."

    The graph layer owns the module→steps mapping; this layer owns
    validation and distribution, so the mapping is handed across as data
    (`Pipeline.flatten_for_runtime()` builds these) rather than by
    `runtime` importing `graph` and walking it, which would reverse the
    dependency direction this module's docstring commits to.

    - `module` — the namespace a caller writes in `params={...}`.
    - `step_names` — every step inside it; one module's params model is
      flat across its steps (doc 03 §4.1, enforced at build time by
      `graph.module._check_no_duplicate_param_names`).
    - `model` — the module's pydantic model, harvested or explicit (§4.4).
    - `bound` — values frozen by `.bind()` (§4.3). They stay *runtime
      values with a fixed default*, so they are merged in here, as ordinary
      kernel arguments, and never baked into generated source: binding must
      not trigger a recompile (§4.3, verbatim).
    """

    module: str
    step_names: tuple[str, ...]
    model: Any | None = None
    bound: Mapping[str, Any] = field(default_factory=dict)


def _default_param_spaces(steps: Sequence[Step]) -> tuple[ParamSpace, ...]:
    """One namespace per step, for a caller driving this layer without a
    graph (the scratch tests, `decider2.compile` users). Correct by
    construction for a bare-function module, whose single step is named
    after the module (doc 03 §5.3) — and the only honest default available
    when nobody told us how steps group into modules."""
    from decider2.params import build_params_model

    out: list[ParamSpace] = []
    for step in steps:
        if not step.params and not step.reads_params:
            continue
        out.append(
            ParamSpace(
                module=step.name,
                step_names=(step.name,),
                model=build_params_model(step.name, step.params),
            )
        )
    return tuple(out)


def _unknown_namespace_error(key: str, known: Sequence[str]) -> ValueError:
    """Doc 03 §10: a misspelled param is a hard error, not silence — which
    has to cover the *namespace* too, or `params={"affordabilty": {...}}`
    returns the untuned answer with no signal (doc 03 §2.1: "no error,
    different decisions ... the worst failure mode the design can have").
    Suggestions follow §10's own `use`-id message shape."""
    import difflib

    near = difflib.get_close_matches(key, list(known), n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    return ValueError(
        f"params has no module instance {key!r} ({len(known)} tunable: "
        f"{', '.join(sorted(known)) or 'none'}).{hint} Params are namespaced "
        "by module instance name (doc 03 §10)."
    )


def resolve_params(
    steps: Sequence[Step],
    overrides: Mapping[str, Mapping[str, Any]] | None,
    *,
    shared_overrides: Mapping[str, Any] | None = None,
    param_spaces: Sequence[ParamSpace] | None = None,
) -> ResolvedParams:
    """Build one `ResolvedParams` bundle for this invocation.

    `overrides` is keyed by **module instance name** (doc 03 §10), not step
    name: a multi-step module's knobs live in one flat namespace named after
    the module (§4.1), so `params={"band": {"lo": 20.0}}` must reach a step
    called something else entirely. `param_spaces` carries that mapping; its
    absence falls back to one namespace per step (see `_default_param_spaces`).

    Three things are hard errors rather than silence, all for doc 03 §2.1's
    reason — a retune that quietly does nothing is a wrong-answer bug with no
    signal:
      - an unknown namespace (`_unknown_namespace_error`);
      - an unknown field inside a known one (pydantic `extra="forbid"`,
        §10, set by `params.build_params_model`);
      - overriding a `.bind()`-frozen value, which §4.3 says "leaves the
        caller-facing params interface" — so a caller naming it is working
        from a stale interface, not expressing a preference.

    A `param()`-declared field validates through the model
    `decider2.params.build_params_model` builds from its harvested
    `FieldInfo` — the same object a hand-written pydantic model would be
    (doc 03 §4.4) — so `pipeline.apply(frame, params={"cap_by_income_band":
    {"cap": 999.0}})` raises through pydantic exactly as
    `CapByIncomeBandParams(cap=999.0)` would. Every resolved value ends up as
    a kernel *argument* (doc 05 §4.2), never a literal in generated source,
    which is why retuning never recompiles (`Driver.signatures` staying the
    same length across a retune, doc 05 §9 acceptance criterion 5).
    """
    overrides = overrides or {}
    spaces = tuple(param_spaces) if param_spaces is not None else _default_param_spaces(steps)
    by_module = {sp.module: sp for sp in spaces}
    step_by_name = {s.name: s for s in steps}

    for key in overrides:
        if key not in by_module:
            raise _unknown_namespace_error(key, tuple(by_module))

    per_step_scalar: dict = {}
    per_step_bundle: dict = {}

    for sp in spaces:
        raw = dict(overrides.get(sp.module, {}))
        frozen = sorted(set(raw) & set(sp.bound))
        if frozen:
            raise ValueError(
                f"params['{sp.module}'] sets {frozen}, which "
                f"{'is' if len(frozen) == 1 else 'are'} frozen by "
                f"{sp.module}.bind(). A bound value left the caller-facing "
                "params interface (doc 03 §4.3) — unbind it at composition, "
                "or drop it from the override."
            )
        merged = {**dict(sp.bound), **raw}
        values = sp.model(**merged).model_dump() if sp.model is not None else merged

        for sname in sp.step_names:
            step = step_by_name.get(sname)
            if step is None:
                continue
            if step.reads_params:
                fields = tuple(values.keys())
                bundle_cls = collections.namedtuple(f"_{sname}_params", fields or ("_empty",))
                per_step_bundle[sname] = (
                    bundle_cls(**values) if fields else bundle_cls(_empty=None)
                )
                continue
            # A step reading named params takes only the fields it declared:
            # its siblings' knobs share the module's namespace but are not
            # arguments to this function (doc 03 §4.1).
            for decl in step.params:
                if decl.name in values:
                    per_step_scalar[(sname, decl.name)] = values[decl.name]

    shared = None
    if any(s.reads_shared for s in steps):
        # No SharedParams model is reachable from a flat step list (doc 03
        # §4.2 wants one declared once per app/pipeline); unvalidated here —
        # see module docstring / report.
        raw_shared = dict(shared_overrides or {})
        fields = tuple(raw_shared.keys())
        shared_cls = collections.namedtuple("_shared_params", fields or ("_empty",))
        shared = shared_cls(**raw_shared) if fields else shared_cls(_empty=None)

    return ResolvedParams(per_step_scalar, per_step_bundle, shared)


# ---------------------------------------------------------------------------
# The shared runner
# ---------------------------------------------------------------------------


def _default_build_dir(build_dir: "str | Path | None") -> Path:
    return Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR


def _run(
    steps: Sequence[Step],
    group_ids: Sequence[int],
    registry: dict,
    resolved: ResolvedParams,
    n: int,
    *,
    mode: str,
    terminal_names: frozenset,
    build_dir: "str | Path | None",
) -> dict:
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_VALID_MODES}")
    if mode == "interpreted":
        return modes.run_interpreted(steps, registry, resolved, n)

    driver = build_driver(
        list(steps), list(group_ids), build_dir=_default_build_dir(build_dir), terminal_names=terminal_names
    )
    if mode == "stepped":
        return modes.run_stepped(driver, steps, registry, resolved, n)
    return modes.run_fused(driver, registry, resolved, n)


# ---------------------------------------------------------------------------
# apply() — doc 02 §3.5, batch. Extraction/write-back via decider2.boundary.
# ---------------------------------------------------------------------------


def apply(
    steps: Sequence[Step],
    frame: pl.DataFrame,
    *,
    interface: Interface,
    group_ids: Sequence[int],
    params: Mapping[str, Mapping[str, Any]] | None = None,
    shared: Mapping[str, Any] | None = None,
    origin: str | None = None,
    mode: str = "fused",
    emit: Sequence[str] = (),
    drop: Sequence[str] = (),
    param_spaces: Sequence["ParamSpace"] | None = None,
    policy: MissingInputPolicy | None = None,
    build_dir: "str | Path | None" = None,
) -> pl.DataFrame:
    """Batch: polars in, polars out (doc 02 §3.5). See module docstring for
    the exact contract this is called under.

    `origin=` is provenance for the audit record (doc 03 §6, `observe/
    audit.py`) — accepted so the signature matches the spec, but not yet
    recorded anywhere: `observe/` doesn't exist yet (doc 00-BUILD.md Layer
    5). `policy=` (`MissingInputPolicy`) is accepted for the same reason —
    the graph layer does not thread `pipeline.missing_input_policy` through
    as of this writing (see report) — and defaults to `MissingInputPolicy()`
    (REFER) exactly as `decider2.boundary.route_required_nulls` does.

    The returned frame is additive (doc 03 §7): every input column, plus
    every terminal name, plus `emit`, minus `drop`. A row a `REQUIRED` null
    routed away (doc 03 §1) still appears, with its terminal columns null —
    rendering the actual `Decision` is `observe/`'s job.
    """
    del origin
    from decider2.boundary import (
        DtypeGroup, KernelOutputs, Layout, extract_frame, resolve_kept_input_columns, write_back,
    )

    steps = list(steps)
    group_ids = list(group_ids)
    terminal_names = frozenset(interface.terminals) | frozenset(emit)

    extracted = extract_frame(frame, interface.inputs, policy=policy)

    registry: dict[str, Any] = {}
    for name, ec in extracted.columns.items():
        registry[name] = ec.values.astype(np.float64, copy=False)
        if ec.validity is not None:
            registry[f"__valid__{name}"] = ec.validity

    resolved = resolve_params(steps, params, shared_overrides=shared, param_spaces=param_spaces)
    n = extracted.kernel_frame.height
    registry = _run(
        steps, group_ids, registry, resolved, n, mode=mode, terminal_names=terminal_names, build_dir=build_dir
    )

    names_tuple = tuple(name for name in sorted(terminal_names) if name in registry)
    routed = extracted.routing.routed_count > 0
    if routed:
        arrays = [_scatter_back(registry[name], extracted.routing.mask) for name in names_tuple]
        base_frame = frame
    else:
        arrays = [registry[name] for name in names_tuple]
        base_frame = extracted.kernel_frame

    outputs = KernelOutputs()
    if arrays:
        stacked = np.stack(arrays, axis=0)
        outputs = KernelOutputs(float64=DtypeGroup(names=names_tuple, array=stacked, layout=Layout.COLUMN_MAJOR))

    keep = resolve_kept_input_columns(base_frame.columns, overwritten=names_tuple, dropped=drop)
    return write_back(base_frame, outputs, keep=keep)


def _scatter_back(kernel_values: np.ndarray, routed_mask: np.ndarray) -> np.ndarray:
    """Doc 03 §1: a routed row is never silently dropped from the batch
    result. `routed_mask[i]` True means row `i` never reached the kernel;
    its slot here is left `nan` rather than the array being shorter than the
    frame it is about to `hstack` onto.

    This is `nan`, not a genuine polars null, because
    `decider2.boundary.writeback.DtypeGroup` carries a plain numpy array
    with **no validity mask** — there is currently no way to write an actual
    null through `write_back()` at all. That is adequate for a float
    terminal (this module's own report flags it) but would not work for an
    int64 or bool terminal, which have no NaN equivalent; closing that gap
    means `DtypeGroup` growing an optional validity array, which is
    `decider2.boundary`'s surface to extend, not this module's to route
    around."""
    full = np.full(len(routed_mask), np.nan, dtype=np.float64)
    full[~routed_mask] = kernel_values
    return full


# ---------------------------------------------------------------------------
# score() — doc 02 §3.5, realtime. Minimal reference extraction (see
# module docstring): decider2.boundary has no per-record equivalent to call.
# ---------------------------------------------------------------------------


def _extract_scalar(inp: Input, value: Any) -> tuple[np.ndarray, "np.ndarray | None"]:
    """MINIMAL BOUNDARY SHIM — see module docstring. Every input must
    already be a plain Python number/bool; no dtype ladder (doc 05 §1.5)."""
    if value is not None:
        arr = np.array([float(value)], dtype=np.float64)
        return (arr, np.array([True])) if inp.null_policy is NullPolicy.OPTIONAL else (arr, None)
    if inp.null_policy in (NullPolicy.MISSING_AS, NullPolicy.NOT_APPLICABLE_AS):
        return np.array([float(inp.fill)], dtype=np.float64), None
    return np.array([0.0], dtype=np.float64), np.array([False])


def score(
    steps: Sequence[Step],
    record: Mapping[str, Any],
    *,
    interface: Interface,
    group_ids: Sequence[int],
    params: Mapping[str, Mapping[str, Any]] | None = None,
    shared: Mapping[str, Any] | None = None,
    origin: str | None = None,
    mode: str = "fused",
    emit: Sequence[str] = (),
    param_spaces: Sequence["ParamSpace"] | None = None,
    policy: MissingInputPolicy | None = None,
    build_dir: "str | Path | None" = None,
) -> dict:
    """Realtime, single record (doc 02 §3.5). Bypasses polars entirely.

    Takes a **dict**, never per-field keyword arguments — measured, not a
    style choice (EXPERIMENTS.md §N2): at 400 inputs, kwargs binding alone
    costs 1190 us (5.95% of a 20 ms budget) against 60.1 us (0.30%) for a
    dict carrying the same data.
    """
    del origin
    steps = list(steps)
    group_ids = list(group_ids)
    policy = policy or MissingInputPolicy()
    terminal_names = frozenset(interface.terminals) | frozenset(emit)

    routed_reason: str | None = None
    registry: dict[str, Any] = {}
    for inp in interface.inputs:
        if inp.name not in record:
            continue
        value = record[inp.name]
        if value is None and inp.null_policy is NullPolicy.REQUIRED:
            if inp.name in policy.raise_for:
                raise ValueError(
                    f"step argument {inp.name!r} is declared required (no `| "
                    f"None`) but the record's value for {inp.name!r} is null, "
                    f"and {inp.name!r} is in raise_for (doc 03 §1)."
                )
            routed_reason = routed_reason or inp.name
            continue
        values, valid = _extract_scalar(inp, value)
        registry[inp.name] = values
        if valid is not None:
            registry[f"__valid__{inp.name}"] = valid

    out = dict(record)
    if routed_reason is not None:
        # Doc 03 §1: "a null must be able to produce a decision, not an
        # exception." Rendering the decision fully is observe/'s job (not
        # built); surfaced here rather than silently computing nothing.
        out["decision"] = policy.default.value if isinstance(policy.default, Decision) else policy.default
        out["reason"] = policy.reason
        out["routed_on"] = routed_reason
        return out

    resolved = resolve_params(steps, params, shared_overrides=shared, param_spaces=param_spaces)
    registry = _run(
        steps, group_ids, registry, resolved, 1, mode=mode, terminal_names=terminal_names, build_dir=build_dir
    )

    for name in terminal_names:
        if name in registry:
            value = registry[name][0]
            out[name] = value.item() if hasattr(value, "item") else value
    return out
