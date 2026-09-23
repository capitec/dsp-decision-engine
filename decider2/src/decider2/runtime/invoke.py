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
surface" — so it stays a **minimal reference implementation**, mirroring
`decider2.boundary.nulls`' REQUIRED-null routing semantics by hand rather
than reimplementing the frame-shaped functions over one row. It lives in
`decider2.runtime.plan.ScorePlan` (BOUNDARY-REWORK.md Stage 4): the
schema-invariant half of a call — the flattened steps, the interface, the
resolved dtypes, `resolve_params`' structural lookups (`ParamsPlan`, below)
and the pooled 1-row buffers — is built once per pipeline and held, and
`score()` here builds and runs one such plan for a caller without a
`Pipeline` to hold it.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from decider2.compile.driver import numpy_dtype
from decider2.runtime import modes
from decider2.runtime.bundles import bundle_class
from decider2.runtime.modes import ResolvedParams
from decider2.types import Decision, Input, Interface, MissingInputPolicy, NullPolicy, Step

DEFAULT_BUILD_DIR = Path(".decider2_cache")

# `resolve_params` builds one namedtuple bundle per `params`-reading step
# and (at most) one for `shared`, every single call. The CLASS of that
# bundle has two stability requirements, both met by `decider2.runtime.
# bundles.bundle_class` (see its module docstring for the numba mechanics):
#
# - within a process: the SAME class object for the same field set, every
#   call. `collections.namedtuple(...)` is not memoised by the stdlib, and
#   handing a `packed` step's `fn` (`decider2.compile.driver._call_step_row`
#   / `PackedCompiledSegment.run`) a structurally identical but distinct
#   class per call made `Driver.signatures` grow once per call under GC
#   pressure in a long-lived process — exactly the "retuning recompiles"
#   failure doc 08 §2 forbids.
# - across processes: a class the NEXT process can resolve to its own
#   class for the same fields, because numba's on-disk cache key for a
#   `shared`-taking kernel is the class's identity, pickled. Memoising per
#   process was not enough for that — every process still recompiled every
#   `shared`-reading table kernel, forever — so the class is now registered
#   under a deterministic, self-describing name that pickles by reference.
_bundle_class = bundle_class


def _plain_name(name: str) -> str:
    """The un-qualified half of a possibly `name@module`-qualified emit
    (doc 03 §7)."""
    return name.split("@", 1)[0]


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
    Suggestions follow §10's own `use`-id message shape.

    `decider2.resolve.suggest_names` — not `decider2.graph`'s did-you-mean —
    is what this calls: this module must not import the graph layer (see
    module docstring), and `decider2.resolve` is the one shared, dependency-
    free implementation both layers use (over-engineering audit: there used
    to be two, differently tuned)."""
    from decider2.resolve import suggest_names

    near = suggest_names(key, known, n=3)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    return ValueError(
        f"params has no module instance {key!r} ({len(known)} tunable: "
        f"{', '.join(sorted(known)) or 'none'}).{hint} Params are namespaced "
        "by module instance name (doc 03 §10)."
    )


def _check_str_inputs_are_covered_by_params(steps: Sequence[Step]) -> None:
    """Doc 05 §1.5 + EXPERIMENTS.md §O: a `str`-typed input enters the kernel
    as an int32 dictionary code, never as text (`compile.driver.numpy_dtype`).
    The *only* thing a step can do with it is compare it against a `str`-typed
    `param()` — whose value `resolve_params` below encodes through that same
    column's categories, so the comparison the author wrote stays a plain
    `==` on codes.

    A step that reads a `str` input but declares no `str` param has no such
    vehicle: any comparison against a literal in its body is code-vs-text
    and evaluates to `False` on every row, silently — the one outcome doc 05
    §9's acceptance bar forbids (`test_a_string_input_is_never_silently_
    zeroed`). Checked here, once, before any row runs, independent of
    whether an override was even supplied — `_default_param_spaces` skips a
    step with no `param()` fields entirely, so nothing later in this
    function would ever see it.
    """
    for step in steps:
        str_inputs = tuple(inp for inp in step.inputs if inp.annotation is str)
        if not str_inputs:
            continue
        if any(decl.annotation is str for decl in step.params):
            continue
        names = ", ".join(sorted(i.name for i in str_inputs))
        example = str_inputs[0].name
        raise ValueError(
            f"step '{step.name}' reads str-typed input(s) {names} but "
            f"declares no str-typed param() (doc 05 §1.5): '{example}' "
            "enters the kernel as a dictionary code, never as text, so a "
            "bare string literal in the step body (e.g. "
            f"`{example} == \"private\"`) compares a code against text and "
            "is silently False on every row. Declare the literal as a "
            f"param instead — e.g. `private: str = param(\"private\")` — "
            f"and compare {example} against that."
        )


def _resolve_str_param_code(
    step: Step,
    param_name: str,
    value: str,
    categories: Mapping[str, Sequence[str]] | None,
) -> np.int32:
    """§O: a `str`-typed param's *value* is resolved to its column's
    dictionary code at param-resolution time — the kernel argument is an
    `int32` code, never text, matching `compile.driver._numba_type(str)`.

    - exactly one `str`-typed input on the step -> unambiguous, use it;
    - none, or more than one -> a clear error (do not guess which column);
    - the literal is absent from that column's categories -> `np.int32(-1)`,
      a code no real row can ever carry, so the comparison the author wrote
      is simply always `False` rather than raising (§O, verbatim: "a literal
      absent from the data ... never matches").
    """
    str_inputs = tuple(inp for inp in step.inputs if inp.annotation is str)
    if len(str_inputs) == 0:
        raise ValueError(
            f"step '{step.name}' param '{param_name}' is declared `str` "
            "but the step reads no str-typed input for its literal to "
            "compare against (doc 05 §1.5)."
        )
    if len(str_inputs) > 1:
        names = ", ".join(sorted(i.name for i in str_inputs))
        raise ValueError(
            f"step '{step.name}' reads {len(str_inputs)} str-typed inputs "
            f"({names}); param '{param_name}' doesn't say which column its "
            "literal compares against, and decider2 will not guess. Split "
            "the step so each str param has exactly one str input, or name "
            "the column explicitly."
        )
    column = str_inputs[0].name
    cats = (categories or {}).get(column)
    if cats is None:
        raise ValueError(
            f"step '{step.name}' param '{param_name}' compares against "
            f"column '{column}', but no dictionary categories were "
            f"extracted for it — is '{column}' really a str-typed column?"
        )
    try:
        code = tuple(cats).index(value)
    except ValueError:
        code = -1  # sentinel: no real row's code is ever -1 (§O)
    return np.int32(code)


@dataclass(frozen=True)
class ParamsPlan:
    """The schema-invariant half of `resolve_params`, built once per step
    list and reused for every call (BOUNDARY-REWORK.md Stage 4): the
    str-input coverage check, the module -> `ParamSpace` map, the
    `(owner, step name)`-then-plain-name step lookups, the list of
    `shared`-reading steps and each declared `shared_fields` projection's
    bundle class. Everything here is a pure function of `steps`,
    `param_spaces` and `owners` — the frozen structure — and never of a
    caller's `params=`/`shared=` values, which `resolve()` merges per call.

    Cached only by `decider2.runtime.plan.ScorePlan`, whose own validity
    rule (the identity of the `Pipeline` fields it was built from) covers
    this; `resolve_params()` below builds a fresh one per call for
    `apply()` and any direct caller, so the two entry points run the same
    merge and cannot disagree.
    """

    steps: tuple[Step, ...]
    owners: tuple[str, ...]
    spaces: tuple[ParamSpace, ...]
    by_module: Mapping[str, ParamSpace]
    # `(space, ((step name, Step), ...))` per space — the steps that space's
    # values are distributed over, resolved once via the `(owner, name)`-
    # then-plain-name lookup `params_plan` documents.
    space_steps: tuple[tuple[ParamSpace, tuple[tuple[str, Step], ...]], ...]
    shared_reading_steps: tuple[str, ...]
    # `(owner, Step, projection bundle class)` per step declaring
    # `Step.shared_fields` (`types.py`).
    shared_projections: tuple[tuple[str, Step, type], ...]

    def resolve(
        self,
        overrides: Mapping[str, Mapping[str, Any]] | None,
        *,
        shared_overrides: Mapping[str, Any] | None = None,
        categories: Mapping[str, Sequence[str]] | None = None,
    ) -> ResolvedParams:
        """The per-call half: validate and merge this call's values over
        the plan. `resolve_params` states the contract; this is its body."""
        overrides = overrides or {}
        by_module = self.by_module
        for key in overrides:
            if key not in by_module:
                raise _unknown_namespace_error(key, tuple(by_module))

        per_step_scalar: dict = {}
        _plain_owner: dict = {}
        _ambiguous_plain: set = set()
        per_step_bundle: dict = {}

        for sp, members in self.space_steps:
            raw = overrides.get(sp.module)
            raw = dict(raw) if raw else {}
            if raw and sp.bound:
                frozen = sorted(set(raw) & set(sp.bound))
                if frozen:
                    raise ValueError(
                        f"params['{sp.module}'] sets {frozen}, which "
                        f"{'is' if len(frozen) == 1 else 'are'} frozen by "
                        f"{sp.module}.bind(). A bound value left the caller-facing "
                        "params interface (doc 03 §4.3) — unbind it at composition, "
                        "or drop it from the override."
                    )
            merged = {**sp.bound, **raw}
            values = sp.model(**merged).model_dump() if sp.model is not None else merged

            for sname, step in members:
                if step.reads_params:
                    fields = tuple(values.keys())
                    if not fields:
                        # Doc 03 §4.2's contract applies to any bare bundle
                        # argument, not only `shared`: a step reading `params`
                        # needs at least one field, or there is nothing to hand
                        # it — `namedtuple(..., ("_empty",))` used to be
                        # attempted here and fail with a confusing "field names
                        # cannot start with an underscore", naming an internal
                        # placeholder rather than the module's actual problem.
                        raise ValueError(
                            f"module '{sp.module}' step '{sname}' reads a bare "
                            "`params` argument, but the module declares no "
                            "param() field and no params= model — there is "
                            "nothing to pass it (doc 03 §4.2)."
                        )
                    bundle_cls = _bundle_class(f"_{sname}_params", fields)
                    bundle = bundle_cls(**values)
                    per_step_bundle[sname] = bundle
                    per_step_bundle[(sp.module, sname)] = bundle
                    continue
                # A step reading named params takes only the fields it declared:
                # its siblings' knobs share the module's namespace but are not
                # arguments to this function (doc 03 §4.1).
                for decl in step.params:
                    if decl.name in values:
                        value = values[decl.name]
                        if decl.annotation is str:
                            # Doc 05 §1.5 + §O: the param stays a `str` in the
                            # params document and the pydantic model (a business
                            # user writes "government", never a code) — encoded
                            # to its column's int32 code only here, at the
                            # kernel-argument boundary.
                            value = _resolve_str_param_code(step, decl.name, value, categories)
                        per_step_scalar[(sp.module, sname, decl.name)] = value
                        # The unqualified key is kept only for callers that drive
                        # runtime/ directly with no graph above them, where module
                        # and step name coincide. In a waterfall two modules DO
                        # share a step name (doc 03 §3.2), and this key is then
                        # ambiguous. Poison it rather than let the last writer win
                        # — a lookup that falls through to it would otherwise read
                        # another rule's threshold and return a plausible wrong
                        # number, which doc 03 §2.1 calls the worst failure mode
                        # the design can have.
                        plain = (sname, decl.name)
                        if plain in _plain_owner and _plain_owner[plain] != sp.module:
                            per_step_scalar.pop(plain, None)
                            _ambiguous_plain.add(plain)
                        elif plain not in _ambiguous_plain:
                            _plain_owner[plain] = sp.module
                            per_step_scalar[plain] = value

        shared = None
        per_step_shared: dict = {}
        shared_reading_steps = self.shared_reading_steps
        if shared_reading_steps:
            raw_shared = dict(shared_overrides or {})
            if not raw_shared:
                # Same "_empty" trap as the params bundle above, and the same
                # fix: a bare `shared` argument declares a required-fields
                # contract (doc 03 §4.2) — supplying none of it is a build-time
                # authoring gap, not a namedtuple implementation detail.
                raise ValueError(
                    f"step(s) {', '.join(shared_reading_steps)} read a bare "
                    "`shared` argument, but no shared={...} was supplied. Doc "
                    "03 §4.2: using `shared` declares a required-fields "
                    "contract, checked at composition — supply the field(s) "
                    "those steps read via apply(..., shared={...})."
                )
            fields = tuple(raw_shared.keys())
            shared_cls = _bundle_class("_shared_params", fields)
            shared = shared_cls(**raw_shared)

            # A step that declares WHICH keys it reads (`types.Step.shared_
            # fields` — a table's row/output steps) gets a bundle of exactly
            # those, so its numba type is its own contract and never grows or
            # changes identity with whatever else the caller merged into
            # `shared=` (see `Step.shared_fields` for the measured reasons).
            # Checked here, by name, rather than left to surface as a numba
            # typing error from inside a kernel.
            for owner, step, proj_cls in self.shared_projections:
                missing = [k for k in step.shared_fields if k not in raw_shared]
                if missing:
                    raise ValueError(
                        f"step '{step.name}' (module '{owner}') reads shared "
                        f"field(s) {missing} that shared= does not supply. A "
                        "table's rows travel in `shared` (doc 03 §4.2, doc 08 "
                        "§3.4): pass its `.shared`, or several tables' merged "
                        "(shared={**a.shared, **b.shared})."
                    )
                bundle = proj_cls(*(raw_shared[k] for k in step.shared_fields))
                per_step_shared[step.name] = bundle
                per_step_shared[(owner, step.name)] = bundle

        return ResolvedParams(per_step_scalar, per_step_bundle, shared, per_step_shared)


def params_plan(
    steps: Sequence[Step],
    *,
    param_spaces: Sequence[ParamSpace] | None = None,
    owners: Sequence[str] | None = None,
) -> ParamsPlan:
    """Build the `ParamsPlan` for `steps`: every lookup `resolve_params`
    used to rebuild per call, plus `_check_str_inputs_are_covered_by_params`,
    run exactly once here."""
    _check_str_inputs_are_covered_by_params(steps)
    steps_t = tuple(steps)
    spaces = tuple(param_spaces) if param_spaces is not None else _default_param_spaces(steps_t)
    by_module = {sp.module: sp for sp in spaces}
    owners_t = tuple(owners) if owners is not None else tuple(s.name for s in steps_t)
    # Doc 03 §4.1/§10: a step's OUTPUT name is not unique across modules —
    # that is the waterfall idiom (§3.2), `@step(output="term_cap")` on two
    # different rules. A flat `{step.name: step}` map silently collapses
    # them (the second module's Step overwrites the first's), so the wrong
    # step's OWN `.params` gets consulted and the first module's knob is
    # never populated at all. `(owner, step.name)` is unique — module
    # instance names are enforced unique per pipeline
    # (`graph.pipeline._check_unique_instance_names`) — so key by that
    # instead; `step_by_name` survives only as the fallback for a caller
    # bypassing the graph layer, where no such collision exists to begin
    # with (`_default_param_spaces`'s own module==step.name convention).
    step_by_owner_name = {(o, s.name): s for o, s in zip(owners_t, steps_t)}
    step_by_name = {s.name: s for s in steps_t}
    space_steps = []
    for sp in spaces:
        members = []
        for sname in sp.step_names:
            step = step_by_owner_name.get((sp.module, sname), step_by_name.get(sname))
            if step is not None:
                members.append((sname, step))
        space_steps.append((sp, tuple(members)))
    shared_reading_steps = tuple(s.name for s in steps_t if s.reads_shared)
    shared_projections = tuple(
        (owner, step, _bundle_class("_shared_params", tuple(step.shared_fields)))
        for owner, step in zip(owners_t, steps_t)
        if step.reads_shared and step.shared_fields is not None
    )
    return ParamsPlan(
        steps=steps_t,
        owners=owners_t,
        spaces=spaces,
        by_module=by_module,
        space_steps=tuple(space_steps),
        shared_reading_steps=shared_reading_steps,
        shared_projections=shared_projections,
    )


def resolve_params(
    steps: Sequence[Step],
    overrides: Mapping[str, Mapping[str, Any]] | None,
    *,
    shared_overrides: Mapping[str, Any] | None = None,
    param_spaces: Sequence[ParamSpace] | None = None,
    owners: Sequence[str] | None = None,
    categories: Mapping[str, Sequence[str]] | None = None,
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

    `categories` (column name -> its dictionary, `ExtractedFrame.
    categories`) is what a `str`-typed param's literal is resolved against
    (`_resolve_str_param_code`) — checked unconditionally, in `params_plan`,
    because a step with no `param()` fields at all has no entry in `spaces`
    to walk (`_default_param_spaces` skips it), so a bare string literal in
    its body would otherwise never be checked here.

    Implemented as `params_plan(...)` (the structural half) followed by
    `ParamsPlan.resolve` (the value merge): the single implementation
    `score()`'s cached plan also runs, so the two cannot drift.
    """
    plan = params_plan(steps, param_spaces=param_spaces, owners=owners)
    return plan.resolve(overrides, shared_overrides=shared_overrides, categories=categories)


# ---------------------------------------------------------------------------
# The shared runner
# ---------------------------------------------------------------------------


def _default_build_dir(build_dir: "str | Path | None") -> Path:
    return Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR


def _run(
    steps: Sequence[Step],
    group_ids: Sequence[int],
    owners: Sequence[str],
    registry: dict,
    resolved: ResolvedParams,
    n: int,
    *,
    mode: str,
    terminal_names: frozenset,
    build_dir: "str | Path | None",
) -> dict:
    """Doc 02 §3.1: three modes, one shared kernel. Which one runs is a
    lookup into `modes.MODES` (a registry of `Mode` classes, one per mode
    name) rather than an if/elif over `mode ==`, so a fourth mode is one
    class in `runtime/modes.py` and never a new branch here."""
    try:
        mode_cls = modes.MODES[mode]
    except KeyError:
        raise ValueError(f"unknown mode {mode!r}; expected one of {tuple(modes.MODES)}") from None
    return mode_cls.run(
        steps, group_ids, owners, registry, resolved, n,
        terminal_names=terminal_names, build_dir=_default_build_dir(build_dir),
    )


# ---------------------------------------------------------------------------
# apply() — doc 02 §3.5, batch. Extraction/write-back via decider2.boundary.
# ---------------------------------------------------------------------------


def apply(
    steps: Sequence[Step],
    frame: pl.DataFrame,
    *,
    interface: Interface,
    group_ids: Sequence[int],
    owners: Sequence[str] | None = None,
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
    owners = list(owners) if owners is not None else [s.name for s in steps]

    # Doc 03 §2.1 row 3, "the one error that matters": a name available from
    # both an upstream module output and the input frame is a build error,
    # not a silent pick between them. A name that is ALSO one of this
    # pipeline's own declared leaf inputs is exempted: that is the waterfall
    # self-read idiom (§3.2) seeded straight from the frame, e.g. a
    # standalone `cap_by_income`-style module applied directly — there the
    # frame column is the intended, sole source of the first version, not
    # an unintended second one competing with it.
    leaf_names = {i.name for i in interface.inputs}
    shadowed = (set(frame.columns) & set(interface.outputs)) - leaf_names
    if shadowed:
        name = sorted(shadowed)[0]
        raise ValueError(
            f"'{name}' is produced by this pipeline and is also a column "
            "already present in the input frame. Doc 03 §2.1 row 3: a name "
            "available from both an upstream module output and the input "
            "frame is a build error — qualify it, e.g. rename/drop the "
            f"frame column '{name}', or relabel the producing module's "
            f"output (`module.relabel(writes={{'{name}': '{name}_2'}})`), "
            "rather than silently shadowing one with the other."
        )

    terminal_names = frozenset(interface.terminals) | frozenset(emit)
    # `emit` may carry a `name@module` qualifier (doc 03 §7); the kernel
    # plan below only knows plain step names (a step is never named with an
    # "@" in it), so the qualifier has to come back off before deciding
    # what a kernel segment must materialise as a required output.
    plain_terminal_names = frozenset(_plain_name(n) for n in terminal_names)

    extracted = extract_frame(frame, interface.inputs, policy=policy)

    registry: dict[str, Any] = {}
    for name, ec in extracted.columns.items():
        # Every column arrives already in the dtype the kernel is typed
        # against: the boundary's kind table (`boundary.dtypes.kind_for`)
        # is row for row `compile.driver.numpy_dtype`'s, so there is no
        # `astype` here any more (docs/BOUNDARY-REWORK.md §1.2). Doc 05
        # §1.5: "a string never enters a kernel as a string" — a
        # `str`-declared column arrived as an int32 dictionary code, whose
        # dictionary is `extracted.categories` below. A bare string
        # literal in the step body still cannot be compared against that
        # code (`resolve_params` below only encodes a *declared*
        # `param()`), so that case still fails loudly rather than silently
        # comparing wrong — see `test_a_string_input_is_never_silently_
        # zeroed`.
        registry[name] = ec.values
        if ec.validity is not None:
            registry[f"__valid__{name}"] = ec.validity

    resolved = resolve_params(
        steps, params, shared_overrides=shared, param_spaces=param_spaces, owners=owners,
        categories=extracted.categories,
    )
    n = extracted.kernel_frame.height
    registry = _run(
        steps, group_ids, owners, registry, resolved, n,
        mode=mode, terminal_names=plain_terminal_names, build_dir=build_dir,
    )

    names_tuple = tuple(name for name in sorted(terminal_names) if name in registry)
    routed = extracted.routing.routed_count > 0
    base_frame = frame if routed else extracted.kernel_frame

    # Doc 00 §2 / doc 03 §1 / doc 05 §9 criterion 4: each name keeps its
    # step's own declared dtype rather than being forced onto one shared
    # float64 group. A qualified `name@owner` (doc 03 §7's version chain)
    # resolves against the step that specific owner produced; a plain name
    # resolves against whichever step wrote it *last* — the same "most
    # recent producer wins" rule that already decides its live value
    # (doc 03 §2.1).
    step_by_owner_name = {(o, s.name): s for o, s in zip(owners, steps)}
    last_step_by_name = {s.name: s for s in steps}

    def _step_for(name: str) -> Step | None:
        if "@" in name:
            plain, owner = name.split("@", 1)
            return step_by_owner_name.get((owner, plain))
        return last_step_by_name.get(name)

    def _dtype_for(name: str) -> np.dtype:
        step = _step_for(name)
        if step is None:
            return np.dtype(np.float64)
        if step.output_annotation is not None:
            return numpy_dtype(step.output_annotation)
        try:
            sig = inspect.signature(step.fn, eval_str=True)
        except (NameError, TypeError):
            sig = inspect.signature(step.fn)
        ann = sig.return_annotation
        return numpy_dtype(ann if ann is not inspect.Signature.empty else float)

    by_dtype: dict[np.dtype, list[str]] = {}
    for name in names_tuple:
        by_dtype.setdefault(_dtype_for(name), []).append(name)

    def _group_for(dtype: np.dtype) -> "DtypeGroup | None":
        names = by_dtype.get(dtype)
        if not names:
            return None
        if routed:
            arrays = [_scatter_back(registry[n], extracted.routing.mask, dtype) for n in names]
        else:
            arrays = [registry[n].astype(dtype, copy=False) for n in names]
        stacked = np.stack(arrays, axis=0)
        return DtypeGroup(names=tuple(names), array=stacked, layout=Layout.COLUMN_MAJOR)

    outputs = KernelOutputs(
        float64=_group_for(np.dtype(np.float64)),
        int64=_group_for(np.dtype(np.int64)),
        bool_=_group_for(np.dtype(np.bool_)),
    )

    keep = resolve_kept_input_columns(base_frame.columns, overwritten=names_tuple, dropped=drop)
    return write_back(base_frame, outputs, keep=keep)


def _scatter_back(
    kernel_values: np.ndarray, routed_mask: np.ndarray, dtype: "np.dtype | None" = None
) -> np.ndarray:
    """Doc 03 §1: a routed row is never silently dropped from the batch
    result. `routed_mask[i]` True means row `i` never reached the kernel;
    its slot here is left as a placeholder rather than the array being
    shorter than the frame it is about to `hstack` onto.

    `nan` for a float terminal — not a genuine polars null, because
    `decider2.boundary.writeback.DtypeGroup` carries a plain numpy array
    with **no validity mask** — there is currently no way to write an actual
    null through `write_back()` at all. An int64/bool terminal has no NaN
    equivalent, so it gets 0/False instead: still a placeholder standing in
    for "no genuine null yet", not a claim that 0/False is the routed row's
    real answer. Closing that gap for real means `DtypeGroup` growing an
    optional validity array, which is `decider2.boundary`'s surface to
    extend, not this module's to route around."""
    dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
    if np.issubdtype(dtype, np.floating):
        fill_value: Any = np.nan
    elif dtype == np.bool_:
        fill_value = False
    else:
        fill_value = 0
    full = np.full(len(routed_mask), fill_value, dtype=dtype)
    full[~routed_mask] = kernel_values
    return full


# ---------------------------------------------------------------------------
# score() — doc 02 §3.5, realtime. Minimal reference extraction (see
# module docstring): decider2.boundary has no per-record equivalent to call.
# The per-record marshal itself lives in `decider2.runtime.plan.ScorePlan`
# (BOUNDARY-REWORK.md Stage 4): the record's array dtype is the SAME
# `compile.driver.numpy_dtype(inp.annotation)` `apply()` uses (doc 00 §2 /
# doc 03 §1 / doc 05 §9 criterion 4, review finding 2), not a blanket
# float64 — forcing an `int` input through float64 silently loses precision
# above 2**53 and makes a step that indexes by it fail to type under numba
# even though the identical step compiles under `apply()`. Whatever
# `apply()` does for an annotation, `score()` does too.
# ---------------------------------------------------------------------------


def score(
    steps: Sequence[Step],
    record: Mapping[str, Any],
    *,
    interface: Interface,
    group_ids: Sequence[int],
    owners: Sequence[str] | None = None,
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

    Builds a `decider2.runtime.plan.ScorePlan` from these arguments and
    runs it once — the reference entry point for a caller with no
    `Pipeline` to hang a cached plan on (the scratch tests). `Pipeline.
    score` holds the plan across calls (`Pipeline.score_plan`), which is
    where the single-record budget is actually met (BOUNDARY-REWORK.md
    Stage 4); both run the identical `ScorePlan.run`.
    """
    del origin
    from decider2.runtime.plan import ScorePlan

    plan = ScorePlan.build(
        steps, interface=interface, group_ids=group_ids, owners=owners, mode=mode,
        emit=emit, param_spaces=param_spaces, policy=policy, build_dir=build_dir,
    )
    return plan.run(record, params=params, shared=shared)
