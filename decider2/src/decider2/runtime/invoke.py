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
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from decider2.compile.driver import numpy_dtype
from decider2.runtime import modes
from decider2.runtime.modes import ResolvedParams
from decider2.types import Decision, Input, Interface, MissingInputPolicy, NullPolicy, Step

DEFAULT_BUILD_DIR = Path(".decider2_cache")

# `resolve_params` builds one namedtuple bundle per `params`-reading step
# and (at most) one for `shared`, every single call — `collections.
# namedtuple(...)` is not memoised by the stdlib, so calling it fresh each
# time used to hand a `packed` step's `fn` (`types.Step.packed`, which
# receives `shared`/a bare `params` bundle as a plain positional argument —
# `decider2.compile.driver._call_step_row`/`PackedCompiledSegment.run`) a
# STRUCTURALLY IDENTICAL BUT DISTINCT class object on every call. Measured
# while building this stage: numba's own type-identity handling for that
# pattern is NOT reliably stable under GC pressure from a long-lived
# process (structurally identical namedtuple classes are supposed to type
# as one thing, and mostly do, but a long test suite — or a long-running
# server — eventually hits a case where they don't, and `Driver.signatures`
# grows once per call, unboundedly, exactly the "retuning recompiles"
# failure doc 08 §2 forbids). Memoising the class itself, keyed by its
# field set, removes the reliance on that numba behaviour entirely: the
# SAME Python class object reaches numba every time the field set is the
# same, so there is only ever one type to begin with.
_BUNDLE_CLASS_CACHE: dict[tuple[str, tuple[str, ...]], type] = {}


def _bundle_class(prefix: str, fields: tuple[str, ...]) -> type:
    key = (prefix, fields)
    cls = _BUNDLE_CLASS_CACHE.get(key)
    if cls is None:
        cls = collections.namedtuple(prefix, fields)
        _BUNDLE_CLASS_CACHE[key] = cls
    return cls


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
    (`_resolve_str_param_code`) — checked unconditionally, before the loop
    below, because a step with no `param()` fields at all has no entry in
    `spaces` to walk (`_default_param_spaces` skips it), so a bare string
    literal in its body would otherwise never be checked here.
    """
    _check_str_inputs_are_covered_by_params(steps)
    overrides = overrides or {}
    spaces = tuple(param_spaces) if param_spaces is not None else _default_param_spaces(steps)
    by_module = {sp.module: sp for sp in spaces}
    owners = list(owners) if owners is not None else [s.name for s in steps]
    # Doc 03 §4.1/§10: a step's OUTPUT name is not unique across modules —
    # that is the waterfall idiom (§3.2), `@step(output="term_cap")` on two
    # different rules. A flat `{step.name: step}` map silently collapses
    # them (the second module's Step overwrites the first's), so the wrong
    # step's OWN `.params` gets consulted below and the first module's knob
    # is never populated at all. `(owner, step.name)` is unique — module
    # instance names are enforced unique per pipeline
    # (`graph.pipeline._check_unique_instance_names`) — so key by that
    # instead; `step_by_name` survives only as the fallback for a caller
    # bypassing the graph layer, where no such collision exists to begin
    # with (`_default_param_spaces`'s own module==step.name convention).
    step_by_owner_name = {(o, s.name): s for o, s in zip(owners, steps)}
    step_by_name = {s.name: s for s in steps}

    for key in overrides:
        if key not in by_module:
            raise _unknown_namespace_error(key, tuple(by_module))

    per_step_scalar: dict = {}
    _plain_owner: dict = {}
    _ambiguous_plain: set = set()
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
            step = step_by_owner_name.get((sp.module, sname), step_by_name.get(sname))
            if step is None:
                continue
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
    shared_reading_steps = tuple(s.name for s in steps if s.reads_shared)
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

    return ResolvedParams(per_step_scalar, per_step_bundle, shared)


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

    input_by_name = {i.name: i for i in interface.inputs}
    registry: dict[str, Any] = {}
    for name, ec in extracted.columns.items():
        annotation = input_by_name[name].annotation if name in input_by_name else float
        # Doc 05 §1.5: "a string never enters a kernel as a string" — a
        # `str`-declared column already arrived as a dictionary code
        # (`extract_frame`/`EntryMode.CODES`); this just settles it onto the
        # int32 the kernel is typed against (`compile.driver.numpy_dtype`).
        # A bare string literal in the step body still cannot be compared
        # against that code (`resolve_params` below only encodes a
        # *declared* `param()`), so that case still fails loudly rather
        # than silently comparing wrong — see
        # `test_a_string_input_is_never_silently_zeroed`.
        registry[name] = ec.values.astype(numpy_dtype(annotation), copy=False)
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
# ---------------------------------------------------------------------------


def _extract_scalar(inp: Input, value: Any) -> tuple[np.ndarray, "np.ndarray | None"]:
    """MINIMAL BOUNDARY SHIM — see module docstring. Every input must
    already be a plain Python number/bool; no dtype ladder (doc 05 §1.5).

    Doc 00 §2 / doc 03 §1 / doc 05 §9 criterion 4 (review finding 2): the
    array dtype is the SAME `compile.driver.numpy_dtype(inp.annotation)`
    `apply()` uses (`runtime.invoke.apply`, `annotation = input_by_name[name
    ].annotation ...`), not a blanket `float64`. Forcing every input through
    `float(value)` onto `np.float64` silently loses precision above 2**53
    for an `int`-annotated input, and produces a `float64` argument for a
    step that indexes a tuple/list by that same int — which numba refuses
    (`getitem(..., float64)`) even though the identical step compiles fine
    under `apply()`, where the column really is int64. Matching apply()'s
    resolution here — instead of inventing a new, independently-correct one
    — is the fix: whatever apply() would do for this annotation, score()
    now does too.
    """
    dtype = numpy_dtype(inp.annotation)
    if value is not None:
        arr = np.array([value], dtype=dtype)
        return (arr, np.array([True])) if inp.null_policy is NullPolicy.OPTIONAL else (arr, None)
    if inp.null_policy in (NullPolicy.MISSING_AS, NullPolicy.NOT_APPLICABLE_AS):
        return np.array([inp.fill], dtype=dtype), None
    return np.array([0], dtype=dtype), np.array([False])


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
    """
    del origin
    steps = list(steps)
    group_ids = list(group_ids)
    owners = list(owners) if owners is not None else [s.name for s in steps]
    policy = policy or MissingInputPolicy()
    terminal_names = frozenset(interface.terminals) | frozenset(emit)

    routed_reason: str | None = None
    registry: dict[str, Any] = {}
    record_categories: dict[str, tuple[str, ...]] = {}
    for inp in interface.inputs:
        # Doc 03 §1 (review finding 4): absent and null share one path. A
        # key missing from `record` entirely used to `continue` past this
        # input with nothing written to `registry` for it — the kernel then
        # raised a bare `KeyError` three frames deep, in
        # `modes._build_call_args`/`_row_kwargs`, naming no column and no
        # policy. `record.get(inp.name)` folds "absent" and "present but
        # None" into the same `value`, so `missing_as`/`not_applicable_as`
        # fill an ABSENT key exactly as they fill a present null, and a
        # REQUIRED input that is absent routes through `MissingInputPolicy`
        # exactly like a REQUIRED null does — naming the column, never a
        # raw exception from three frames away.
        value = record.get(inp.name)
        if value is None and inp.null_policy is NullPolicy.REQUIRED:
            if inp.name in policy.raise_for:
                raise ValueError(
                    f"step argument {inp.name!r} is declared required (no `| "
                    f"None`) but the record has no usable value for "
                    f"{inp.name!r} (absent or null), and {inp.name!r} is in "
                    "raise_for (doc 03 §1)."
                )
            routed_reason = routed_reason or inp.name
            continue
        if inp.annotation is str and isinstance(value, str):
            # A str input enters the kernel as a dictionary code (doc 05
            # §1.5). apply() takes the dictionary from the column; a single
            # record has no column, so the record IS its own dictionary: this
            # value is code 0, and _resolve_str_param_code then gives the
            # literal 0 when it matches and -1 when it does not. `a == b` over
            # one row is exactly `code(a) == code(b)` under that mapping, so
            # score() and apply() agree without score() needing a declared
            # vocabulary it has no way to know.
            record_categories[inp.name] = (value,)
            registry[inp.name] = np.array([0], dtype=np.int32)
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

    resolved = resolve_params(
        steps, params, shared_overrides=shared, param_spaces=param_spaces,
        owners=owners, categories=record_categories,
    )
    registry = _run(
        steps, group_ids, owners, registry, resolved, 1,
        mode=mode, terminal_names=terminal_names, build_dir=build_dir,
    )

    for name in terminal_names:
        if name in registry:
            value = registry[name][0]
            out[name] = value.item() if hasattr(value, "item") else value
    return out
