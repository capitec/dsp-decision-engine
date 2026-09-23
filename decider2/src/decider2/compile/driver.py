"""Build compiled kernels from an ordered step list — doc 05 §4.3 (driver
shape), §5 (compiled variants), §6 (fallback), §7 (fusion grouping); doc 02
§3.1 (three modes), §3.3 (variants decided by the author), §1.2 (fusion is
explicit).

**Grouping is the caller's decision, not this module's.** `group_ids` names
which contiguous run of steps is one fuse()-authored kernel; absent a
`fuse()`, the default the caller should pass is one distinct id per step —
"one kernel per module" (doc 05 §7) — because a flat pipeline of bare
functions is itself a sequence of one-step modules (doc 03 §5.3). There is no
grouping heuristic in here and none should be added: doc 02 §1.2 measured
fusion at 0.11x-1071x depending on body cost, at fixed module count, and
withdrew the one heuristic ("cap at ~6-9 steps") an earlier draft proposed.

**Fallback is a kernel-boundary decision, never a per-node one** (doc 05 §6).
A group containing one un-njit-able step is *split around it* here: the
steps before and after stay in their own compiled segment(s), the offending
step runs in Python, and the group's kernel-per-module contract is preserved
everywhere it can be. Only numba's own compile-failure exceptions are treated
as "needs a fallback" (see `_FALLBACK_TRIGGERS` below — not quite the single
class doc 05 §6 names, see that note) — a `ZeroDivisionError` in a step body
must propagate identically whether or not that step ends up compiled, or a
genuine bug gets silently relabelled as "the compiler couldn't handle this"
(EXPERIMENTS.md §B: `objmode` per row measured 77x, per-row `prange` 615x — a
per-node escape would be worse than the thing it rescues, which is why there
isn't one).

**A `Segment` runs itself.** `CompiledSegment`/`FallbackSegment` (a
discriminated union on `.kind`) each own their own `.run()`, so
`runtime.modes.run_fused` just calls it — no `seg.kind == "compiled"`
branch out there, and no `kernel_fn: ... | None  # set when kind ==
"compiled"` field shared by a kind that never sets it. The generic
per-row/per-segment calling convention (`ResolvedParams`, `_row_kwargs`,
`_build_call_args`, ...) lives here too, next to the classes that are the
only callers of it now — `runtime.modes.run_interpreted`/`run_stepped`
import it back for the two rungs of doc 02 §3.1's equivalence ladder that
drive a step one at a time outside any `Segment`.
"""
from __future__ import annotations

import inspect
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as _dc_field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Mapping, Sequence

import numpy as np
from numba import njit
from numba.core import types as nbtypes
from numba.core.errors import NumbaError, UnsupportedBytecodeError

# doc 05 §6: "Catch numba.core.errors.NumbaError only — never bare Exception."
# Verified empirically (numba 0.67.0) rather than taken on faith: every numba
# compile-failure class checked (TypingError, UnsupportedError, LoweringError,
# ForbiddenConstruct, ...) subclasses NumbaError *except* one —
# UnsupportedBytecodeError subclasses plain Exception directly. It is also
# exactly the error a step containing an ordinary `import` statement in its
# body raises (IMPORT_NAME is an unsupported opcode), which is doc 02 §3.2's
# own example of "weird Python... served by the fallback" (regex, string
# munging). Catching NumbaError alone would silently fail to route that case
# to the fallback path at all — not "catch too much", the opposite defect.
# So this tuple is named explicitly, not widened to bare Exception: a
# ZeroDivisionError is neither of these two classes and still propagates
# (doc 05 §9 acceptance criterion 6's own wording). See the report for this
# module for the full note.
_FALLBACK_TRIGGERS: tuple[type[BaseException], ...] = (NumbaError, UnsupportedBytecodeError)

from decider2.compile.kernel import KernelPlan, build_fused_kernel, kernel_signature
from decider2.types import Input, NullPolicy, Step

SegmentKind = Literal["compiled", "fallback"]

# Best-effort probe values by declared annotation, used only to force eager
# compilation at build time so a doomed step is caught before it is ever
# wired into a kernel (doc 02 §3.4: compilation belongs at build, not at
# startup). A real `decider2 build` drives this from the declared input
# schema (doc 07 §5: "build takes an input schema... the pipeline alone is
# not enough to compile"); this fallback exists so this module is usable
# stand-alone, ahead of that CLI existing.
_NUMBA_BY_ANNOTATION: dict[Any, Any] = {
    bool: nbtypes.boolean,
    int: nbtypes.int64,
    float: nbtypes.float64,
    # doc 05 §1.5 "Strings in detail": a string never enters a kernel as a
    # string — it enters as its dictionary code (EXPERIMENTS.md §O measured
    # the alternative, `typed.List[str]`, at 31x end to end). `str` maps to
    # the same int32 both here and in `_NUMPY_BY_ANNOTATION` below, for an
    # input column and for a `str`-declared param alike, so the two never
    # silently disagree about what a step's `sector: str` means.
    str: nbtypes.int32,
}

# The runtime-array counterpart of the table above (doc 00 §2: "money is
# scaled int64"; doc 03 §1 names Int64->float64 above 2**53 a REJECTED
# design; doc 05 §9 criterion 4 requires Boolean to round-trip). Kept next
# to `_NUMBA_BY_ANNOTATION`, keyed on the same three annotations, so the
# compile-time numba type and the runtime numpy dtype can never silently
# disagree about what a step's `-> int`/`-> bool` annotation means.
_NUMPY_BY_ANNOTATION: dict[Any, Any] = {
    bool: np.bool_,
    int: np.int64,
    float: np.float64,
    # See `_NUMBA_BY_ANNOTATION` above: a `str` annotation is a dictionary
    # code at the boundary, doc 05 §1.5.
    str: np.int32,
}


def numpy_dtype(annotation: Any) -> np.dtype:
    """The numpy dtype a declared `int`/`bool`/`float` annotation crosses
    the runtime boundary as; unannotated (or anything else) stays float64,
    the boundary's existing default for an undeclared column (doc 05
    §1.5)."""
    return np.dtype(_NUMPY_BY_ANNOTATION.get(annotation, np.float64))


def _numba_type(annotation: Any, *, optional: bool = False) -> Any:
    """The numba type a declared annotation compiles against.

    Unannotated falls back to float64, which is what the boundary supplies
    for an undeclared column (doc 05 §1.5's ladder puts everything numeric
    on float64 unless the author tightened it).
    """
    base = _NUMBA_BY_ANNOTATION.get(annotation, nbtypes.float64)
    return nbtypes.Optional(base) if optional else base


def _try_njit(step: Step, sample_values: Mapping[str, Any]) -> tuple[Callable, str | None]:
    """Attempt to compile one step in isolation.

    Returns `(callable, None)` on success — the callable is the njit
    dispatcher — or `(step.fn, reason)` on a genuine compile failure, where
    `callable` is the *original* plain function (stepped/fallback mode calls
    it directly, doc 02 §3.1: "stepped ... is also the fallback path").

    `_FALLBACK_TRIGGERS` is caught, never bare `Exception` (doc 05 §6). A step that
    reads a bare `params`/`shared` argument (doc 03 §4.2) can't be probed
    without a resolved bundle of the right NamedTuple type, so it is
    compiled lazily instead — still cached, just not verified at this call.

    A `packed` step (`types.Step.packed`) arrives with `fn` ALREADY a real
    njit dispatcher — `decider2.trees.encode`/`decider2.tables.encode`/
    `decider2.graph.control_flow` build it that way directly, never a plain
    function for this module to wrap. Re-`njit`-ing an already-jitted
    dispatcher is a hard `TypeError` ("a jit decorator was called on an
    already jitted function"), and probing it against a signature guessed
    from its (generic, two-argument `(args, params)`) Python signature would
    guess wrong — so this returns it as-is, un-probed: it was built
    njit-compilable by construction (the only operations inside it are
    array indexing and arithmetic over `decider2.trees.interpreter.
    walk_tree`/`decider2.tables.interpreter.scan_table`, both already
    `@njit` — `scan_table` `cache=True`, `walk_tree` `inline="always"`, so
    its body is spliced into the tree's `path_fn` and, from there, into
    `build_packed_kernel`'s per-row loop; see `walk_tree`'s docstring), so
    there is nothing here for `_FALLBACK_TRIGGERS` to ever legitimately
    catch.
    """
    if step.packed:
        return step.fn, None
    fn = njit(cache=True)(step.fn)
    if step.reads_params or step.reads_shared:
        return fn, None
    try:
        sig = inspect.signature(step.fn)
        input_by_name = {i.name: i for i in step.inputs}
        param_by_name = {p.name: p for p in step.params}
        signature: list[Any] = []
        for pname in sig.parameters:
            if pname in input_by_name:
                inp = input_by_name[pname]
                signature.append(_numba_type(
                    inp.annotation,
                    optional=inp.null_policy is NullPolicy.OPTIONAL,
                ))
            elif pname in param_by_name:
                decl = param_by_name[pname]
                signature.append(_numba_type(
                    decl.annotation if decl.annotation is not Any else type(decl.default)
                ))
            else:
                signature.append(nbtypes.float64)
        fn.compile(tuple(signature))
        return fn, None
    except _FALLBACK_TRIGGERS as exc:
        return step.fn, str(exc)


def _external_inputs(run_steps: Sequence[Step]) -> tuple[Input, ...]:
    """Inputs a run of steps needs from outside itself — anything not
    produced by an earlier step within the *same* run. First-seen order,
    never sorted or hashed (doc 05 §4.2's determinism requirement)."""
    produced_locally: set[str] = set()
    seen: set[str] = set()
    result: list[Input] = []
    for step in run_steps:
        for inp in step.inputs:
            if inp.name in produced_locally or inp.name in seen:
                continue
            result.append(inp)
            seen.add(inp.name)
        produced_locally.add(step.name)
    return tuple(result)


def _needed_from(steps: Sequence[Step], j: int, terminal_names: frozenset) -> set[str]:
    """Which names must be materialised as an array once a run ending right
    before index `j` finishes — a pipeline terminal/emitted name, or read by
    any step from `j` onward (a later segment in this group, or a later
    group entirely; both cross a kernel boundary as numpy, doc 05 §7:
    "intermediates... stay in numpy, they do not round-trip through
    polars").

    Deliberately keyed on the *run's own end index*, not a single pipeline-
    wide table: a value read only by a later step *inside the same run*
    (index < j) must never count here — that is the "lives in the kernel's
    registers and dies there" saving doc 03 §7 describes, and it only
    applies within one compiled segment. Fusing two steps together (an
    author's `fuse()`) is exactly what turns a cross-kernel requirement into
    a local one, by moving `j` past both of them at once.
    """
    needed = set(terminal_names)
    for later in steps[j:]:
        for inp in later.inputs:
            needed.add(inp.name)
    return needed



# ---------------------------------------------------------------------------
# Per-row/per-segment calling convention — how a resolved params bundle and
# a name -> array registry become one step's actual call, for whichever
# step callable a `Segment` ends up holding (njit dispatcher or plain
# Python fallback). Lives here, next to `Segment`, rather than in
# `runtime.modes` (its previous home): now that a segment runs itself
# (`Segment.run`, below), the calling convention is that method's own
# implementation detail, not a caller's. `runtime.modes.run_interpreted`/
# `run_stepped` still drive a step one at a time outside any `Segment`
# (doc 02 §3.1's `interpreted`/`stepped` rungs), so they import these
# straight back from here rather than a second copy existing.
# ---------------------------------------------------------------------------

_MISSING = object()

# `_signature`'s memo, keyed on the function OBJECT (weakly, so a step that
# is garbage-collected takes its entry with it). Only a signature whose
# annotations `eval`'d successfully is stored — the fallback result is
# recomputed each time so a name that becomes resolvable later still gets
# picked up, exactly as before the memo existed. What could make an entry
# stale: rebinding `fn.__annotations__` after the first call, which nothing
# in decider2 does (a `Step` is frozen and its `fn` is the author's own
# function). Measured reason (BOUNDARY-REWORK.md Stage 4): `eval_str=True`
# cost 11 `eval` calls per flagship `score()`, ~110 µs of a 420 µs call,
# for a result that never changes.
_SIGNATURES: "weakref.WeakKeyDictionary[Callable, inspect.Signature]" = weakref.WeakKeyDictionary()


def _signature(fn) -> "inspect.Signature":
    """`inspect.signature`, with the same `eval_str=True`-then-fall-back
    `decider2.params.harvest_signature` already uses: a step's module using
    `from __future__ import annotations` stringifies every annotation, and
    without this a `-> int`/`-> bool` return type would compare equal to
    nothing in `numpy_dtype`'s table and silently fall back to float64 —
    exactly the bug this module exists to close. Memoised per function
    object (`_SIGNATURES`, above)."""
    try:
        return _SIGNATURES[fn]
    except (KeyError, TypeError):
        pass
    try:
        sig = inspect.signature(fn, eval_str=True)
    except (NameError, TypeError):
        return inspect.signature(fn)
    try:
        _SIGNATURES[fn] = sig
    except TypeError:  # not weak-referenceable: served uncached, still correct
        pass
    return sig


def step_return_annotation(step: Step) -> Any:
    """The declared return type a step's array should be materialised as —
    `step.output_annotation` when the step is `packed` (its `fn` has the
    generic `(args, params)` signature and no return annotation of its own
    to read, `types.Step.packed`'s own docstring), else the ordinary
    `inspect.signature(step.fn).return_annotation`."""
    if step.output_annotation is not None:
        return step.output_annotation
    return _signature(step.fn).return_annotation


def _return_dtype(fn_or_step) -> np.dtype:
    """Doc 00 §2 / doc 03 §1 / doc 05 §9 criterion 4: a step's declared
    return annotation decides the array dtype it is written into, so an
    `-> int`/`-> bool` step survives the boundary as its own dtype instead
    of being forced onto float64 (which silently degrades an Int64 above
    2**53, and cannot hold a Boolean at all).

    Accepts either a plain callable (every pre-existing caller) or a
    `Step` (needed to see `packed`/`output_annotation` — a bare function
    has no such thing to read)."""
    if isinstance(fn_or_step, Step):
        return numpy_dtype(step_return_annotation(fn_or_step))
    return numpy_dtype(_signature(fn_or_step).return_annotation)


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
    - `per_step_shared[step_name]` / `[(owner, step_name)]` — for a step
      that declares `Step.shared_fields`, a bundle of ONLY those fields;
      `_shared_arg` picks it over `shared` for that step. Absent for every
      step that does not declare them (they get `shared` whole).
    """

    per_step_scalar: dict
    per_step_bundle: dict
    shared: Any | None = None
    per_step_shared: dict = _dc_field(default_factory=dict)


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


def _shared_arg(resolved: "ResolvedParams", owner: str | None, step: Step) -> Any:
    """The `shared` bundle THIS step is called with: its own projection
    (`Step.shared_fields`, built by `runtime.invoke.resolve_params`) when
    it declares one, else the whole bundle. Same owner-first, plain-name
    fallback as `_bundle_arg`, for the same reason."""
    if step.shared_fields is None:
        return resolved.shared
    if owner is not None:
        value = resolved.per_step_shared.get((owner, step.name), _MISSING)
        if value is not _MISSING:
            return value
    value = resolved.per_step_shared.get(step.name, _MISSING)
    if value is not _MISSING:
        return value
    # A `ResolvedParams` built by hand (a test driving `modes` directly)
    # with only `shared=` set: fall back to the whole bundle rather than
    # fail, since the whole bundle is a superset the step can read.
    return resolved.shared


def _row_kwargs(
    step: Step,
    owner: "str | None",
    sig: "inspect.Signature",
    registry: dict,
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


# The row-gathering machinery for packed steps — `_packed_args_kind`, the
# `_gather*`/`_fill_*` compiled gathers, the `_typed_*` family,
# `_packed_input_arrays`, `build_packed_kernel`/`_build_typed_kernel` and
# `_packed_row_args` — lives in `decider2.compile.gather` (docs/
# BOUNDARY-REWORK.md, Stage 1b). It is imported HERE, mid-module, rather than
# at the top because `gather._packed_row_args` calls `_scalar_arg` above and
# is annotated with `ResolvedParams`, which `gather` imports back from this
# module: both must already exist when `gather` first executes. Every moved
# name is re-exported so `from decider2.compile.driver import <name>` keeps
# working for the callers that predate the split.
from decider2.compile.gather import (  # noqa: F401 — re-exported, see above
    _BYTES_DUMMY,
    _TYPED_DTYPES,
    _TYPED_DUMMIES,
    _as_readonly,
    _as_readonly_f64,
    _build_typed_kernel,
    _fill_array,
    _fill_spans,
    _gather0,
    _gather1_raw,
    _gather_array,
    _make_row_gatherer,
    _packed_args_kind,
    _packed_input_arrays,
    _packed_row_args,
    _readonly_empty,
    _typed_counts,
    _typed_input_arrays,
    _typed_layout,
    _typed_params,
    _typed_row_args,
    build_packed_kernel,
)


def _call_step_row(
    fn: Callable,
    step: Step,
    owner: "str | None",
    sig: "inspect.Signature",
    registry: dict,
    resolved: ResolvedParams,
    i: int,
) -> Any:
    """One step, one row — `interpreted`/`stepped`/`fallback`'s shared call
    site. Branches on `step.packed` so those three rungs (the only callers
    that still drive a step one row at a time in plain Python) agree with
    the fused kernel's own packed-call builder on what a packed `fn`
    means."""
    if step.packed:
        args, params = _packed_row_args(step, owner, registry, resolved, i)
        if step.reads_shared:
            return fn(args, params, _shared_arg(resolved, owner, step))
        return fn(args, params)
    return fn(**_row_kwargs(step, owner, sig, registry, resolved, i))


def _build_call_args(
    plan: KernelPlan, registry: dict, resolved: ResolvedParams, out_arrays: dict, n: int,
    roles: "Sequence[Any] | None" = None,
) -> tuple:
    """`(n, cols, valids, params_all, outs)` — the fused kernel's five
    arguments (`compile.kernel.build_fused_kernel`), each tuple packed in
    `kernel_signature` order so the kernel's own argument-source map, built
    from the same list, indexes the right element. `roles` is that same
    list, precomputed by a caller that holds it (`CompiledSegment._call_plan`)
    so it is not rebuilt per call; omitted, it is derived from `plan` here."""
    cols: list = []
    valids: list = []
    params_all: list = []
    outs: list = []
    for role in (kernel_signature(plan) if roles is None else roles):
        if role.kind == "array":
            cols.append(registry[role.input_name])
        elif role.kind == "valid":
            valids.append(registry[f"__valid__{role.input_name}"])
        elif role.kind == "param_scalar":
            params_all.append(_scalar_arg(resolved, role.owner, role.step_name, role.param_name))
        elif role.kind == "params_bundle":
            params_all.append(_bundle_arg(resolved, role.owner, role.step_name))
        elif role.kind == "shared":
            params_all.append(resolved.shared)
        elif role.kind == "output":
            outs.append(out_arrays[role.output_name])
        else:  # pragma: no cover - exhaustive over ArgRole.kind
            raise AssertionError(f"unhandled arg role {role.kind!r}")
    return n, tuple(cols), tuple(valids), tuple(params_all), tuple(outs)


class Segment(ABC):
    """One contiguous run within a fuse()-group: either fully compiled
    (`CompiledSegment`) or (doc 05 §6) exactly one un-njit-able step
    running in Python (`FallbackSegment`). The blast radius of a bad node
    is the kernel it was going into, never the whole pipeline and never
    just that one node inside a still-compiled kernel — there is no such
    thing as the latter (EXPERIMENTS.md §B).

    A discriminated union of the two variants below, each carrying only the
    fields its own kind needs — no more `kernel_fn: Callable | None  # set
    when kind == "compiled"` on a shared shape — and each running itself
    (`.run`), so `runtime.modes.run_fused` and `runtime.serve.ServeHandle.
    gil_report` no longer branch on `.kind` themselves; they call the
    method. `.kind` survives as a plain string class attribute (not a type
    check) purely because the driver cache key and a couple of tests still
    read it that way (`seg.kind == "compiled"`).
    """

    kind: ClassVar[SegmentKind]
    steps: tuple[Step, ...]
    external_inputs: tuple[Input, ...]
    required_outputs: tuple[str, ...]
    owners: tuple[str, ...]                  # module instance name per `steps` entry

    @abstractmethod
    def run(self, registry: dict, resolved: ResolvedParams, n: int) -> None:
        """Run this segment for `n` rows, writing its outputs — plain and
        `name@owner`-qualified (doc 03 §3.3/§7) — into `registry` in
        place. The *only* place `fused` mode (`runtime.modes.run_fused`)
        executes plain Python is a `FallbackSegment`'s own override of
        this, and only for the node that could not be compiled, never its
        neighbours."""

    @property
    def signatures(self) -> tuple:
        """This segment's contribution to `Driver.signatures` (doc 05 §9
        criterion 5) — nothing, unless overridden. Only `CompiledSegment`
        has any numba specialisations to report."""
        return ()

    @abstractmethod
    def gil_report_entry(self) -> dict[str, Any]:
        """This segment's one row of `ServeHandle.gil_report()` (doc 00
        §2c)."""


@dataclass(frozen=True)
class CompiledSegment(Segment):
    """A run of steps that all survived njit, fused into one compiled
    kernel (doc 05 §7)."""

    steps: tuple[Step, ...]
    external_inputs: tuple[Input, ...]
    required_outputs: tuple[str, ...]
    owners: tuple[str, ...]
    kernel_fn: Callable
    plan: KernelPlan

    kind: ClassVar[SegmentKind] = "compiled"

    @cached_property
    def _call_plan(self) -> "tuple[tuple[Any, ...], tuple[tuple[str, np.dtype, str], ...]]":
        """The schema-invariant half of `run`, computed once per segment:
        `kernel_signature(self.plan)`'s roles, and `(name, dtype, owner)`
        per required output. A segment is a frozen value built once per
        `Driver`, so nothing this reads can change after construction;
        `cached_property` writes straight into `__dict__`, which a frozen
        dataclass permits. Before this (BOUNDARY-REWORK.md Stage 4) `run`
        re-derived both per call, and `_return_dtype`'s `inspect.signature`
        was the single largest cost of a single-record `score()`."""
        owner_by_name = dict(zip((s.name for s in self.steps), self.owners)) if self.owners else {}
        step_by_name = {s.name: s for s in self.steps}
        outputs = tuple(
            (name, _return_dtype(step_by_name[name]), owner_by_name.get(name, name))
            for name in self.required_outputs
        )
        return tuple(kernel_signature(self.plan)), outputs

    def run(self, registry: dict, resolved: ResolvedParams, n: int) -> None:
        # Doc 05 §7: "intermediates ... stay in numpy, they do not
        # round-trip through polars" — a value a later segment needs is
        # looked up in `registry` regardless of whether it came from the
        # original frame or an earlier segment's own output.
        roles, outputs = self._call_plan
        out_arrays = {name: np.empty(n, dtype=dtype) for name, dtype, _ in outputs}
        self.kernel_fn(*_build_call_args(self.plan, registry, resolved, out_arrays, n, roles))
        for name, _, owner in outputs:
            arr = out_arrays[name]
            registry[name] = arr
            registry[f"{name}@{owner}"] = arr

    @property
    def signatures(self) -> tuple:
        return tuple(self.kernel_fn.signatures)

    def gil_report_entry(self) -> dict[str, Any]:
        names = [s.name for s in self.steps]
        return {
            "kernel": self.plan.group_name,
            "steps": names,
            # Doc 00 §2c: "a group releases the GIL only when every step in
            # it asked to" — one step in the kernel that did not is enough
            # to keep it held, because the kernel is one call.
            "holds_gil": not all(s.nogil for s in self.steps),
        }


@dataclass(frozen=True)
class PackedCompiledSegment(Segment):
    """Exactly one `types.Step.packed` step (a tree/table/Branch/Loop
    `Step`, doc 08 §3.4), run as a genuinely compiled, `@njit`'d per-row
    loop (`build_packed_kernel`) — `kind == "compiled"`, never `"fallback"`:
    no Python runs per row, unlike `FallbackSegment`. Not fused with
    neighbouring steps — see `build_packed_kernel`'s own docstring for why,
    and this module's report for the measured cost.
    """

    steps: tuple[Step, ...]
    external_inputs: tuple[Input, ...]
    required_outputs: tuple[str, ...]
    owners: tuple[str, ...]
    kernel_fn: Callable

    kind: ClassVar[SegmentKind] = "compiled"

    def run(self, registry: dict, resolved: ResolvedParams, n: int) -> None:
        step = self.steps[0]
        owner = self.owners[0] if self.owners else step.name
        arrays = _packed_input_arrays(step, registry)
        params = tuple(_scalar_arg(resolved, owner, step.name, p.name) for p in step.params)
        if step.typed_args:
            params = _typed_params(step, params)
        out = np.empty(n, dtype=_return_dtype(step))
        if step.reads_shared:
            self.kernel_fn(arrays, params, _shared_arg(resolved, owner, step), n, out)
        else:
            self.kernel_fn(arrays, params, n, out)
        registry[step.name] = out
        registry[f"{step.name}@{owner}"] = out

    @property
    def signatures(self) -> tuple:
        return tuple(self.kernel_fn.signatures)

    def gil_report_entry(self) -> dict[str, Any]:
        step = self.steps[0]
        return {
            "kernel": f"packed:{step.name}",
            "steps": [step.name],
            "holds_gil": not step.nogil,
        }


@dataclass(frozen=True)
class FallbackSegment(Segment):
    """Exactly one un-njit-able step, run row-by-row in plain Python (doc
    05 §6) — a kernel-boundary decision, never a per-node one: the steps
    before and after stay in their own `CompiledSegment`(s)."""

    steps: tuple[Step, ...]
    external_inputs: tuple[Input, ...]
    required_outputs: tuple[str, ...]
    owners: tuple[str, ...]
    fallback_reason: str

    kind: ClassVar[SegmentKind] = "fallback"

    def run(self, registry: dict, resolved: ResolvedParams, n: int) -> None:
        step = self.steps[0]
        owner = self.owners[0] if self.owners else step.name
        sig = _signature(step.fn)
        out = np.empty(n, dtype=_return_dtype(step))
        for i in range(n):
            out[i] = _call_step_row(step.fn, step, owner, sig, registry, resolved, i)
        registry[step.name] = out
        registry[f"{step.name}@{owner}"] = out

    def gil_report_entry(self) -> dict[str, Any]:
        names = [s.name for s in self.steps]
        return {
            "kernel": f"fallback:{names[0]}",
            "steps": names,
            "holds_gil": True,  # a fallback segment runs plain Python
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class Driver:
    """A fully built pipeline: an ordered list of segments, plus an
    individually-compiled (or, for a fallback step, plain) callable per step
    name for `stepped` mode — which drives them one at a time regardless of
    how `fused` mode grouped them (doc 02 §3.1)."""

    segments: tuple[Segment, ...]
    step_fns: dict  # step name -> callable

    @property
    def signatures(self) -> list:
        """Every numba specialisation across every compiled segment.

        Doc 05 §9's acceptance criterion 5: retuning any params bundle must
        leave this at the same length — params arrive as *kernel arguments*
        (`decider2.compile.kernel`'s `params_all` tuple), never baked into
        the kernel, so a value-only retune changes nothing this list depends
        on. Changing a field's *type* (e.g. a param toggling `float` <->
        `float | None`) is the named negative control and *should* grow it
        by one.
        """
        out: list = []
        for seg in self.segments:
            out.extend(seg.signatures)
        return out


# A Driver is a compiled artefact, not a per-call object. Doc 02 §3.4:
# "Compilation happens at image build, not at startup." Doc 05 §6: "Cache the
# decision per node so a doomed compile isn't retried every call." Rebuilding
# per call still *works* — numba's on-disk cache reloads rather than
# recompiles — but it constructs a fresh dispatcher per step per call, which
# measured at 81 ms p50 over 30 modules against a 20-100 ms budget, and it
# discards the fallback decision the docstring above promises to keep.
#
# Keyed by the structural identity of the request: which steps, in which
# order, in which fuse() groups, materialising which names. Params are NOT in
# the key and must never be -- they arrive as kernel arguments, and a retune
# reusing this entry is exactly the guarantee doc 08 §2 rests on.
_DRIVER_CACHE: dict[tuple, "Driver"] = {}


def _driver_key(
    steps: Sequence[Step],
    group_ids: Sequence[int],
    owners: Sequence[str],
    build_dir: "str | Path",
    terminal_names: frozenset,
    parallel_group_ids: frozenset,
    fastmath_group_ids: frozenset,
) -> tuple:
    return (
        tuple((s.name, s.fn) for s in steps),
        tuple(group_ids),
        tuple(owners),
        str(build_dir),
        terminal_names,
        parallel_group_ids,
        fastmath_group_ids,
    )


def clear_driver_cache() -> None:
    """Drop every memoised Driver. For tests, and for a generation swap that
    must not inherit a predecessor's compiled artefacts."""
    _DRIVER_CACHE.clear()


def build_driver(
    steps: Sequence[Step],
    group_ids: Sequence[int],
    *,
    build_dir: "str | Path",
    terminal_names: frozenset = frozenset(),
    parallel_group_ids: frozenset = frozenset(),
    fastmath_group_ids: frozenset = frozenset(),
    sample_values: Mapping[str, Any] | None = None,
    owners: Sequence[str] | None = None,
) -> Driver:
    """Compile `steps` (already in valid execution order — topological
    sort/ordering is the graph layer's responsibility, doc 02 §6's
    `runtime/plan.py`) into a `Driver`.

    `group_ids[i]` is the fuse()-group of `steps[i]`; a group boundary is
    never crossed even when both sides would compile (doc 05 §7: "never fuse
    across module boundaries unless the author wrote fuse()"). Within one
    group, an un-njit-able step forces an additional, unrequested split —
    that is the fallback mechanism (doc 05 §6), not a second grouping
    policy.

    `owners[i]` is the module instance name `steps[i]` belongs to (doc 03
    §4.1/§10). Omitted (the default, for every caller that drives this
    module directly without the graph layer — the scratch tests, a bare
    `decider2.compile` user), each step is treated as its own owner, which
    is exactly what a bare-function module already means (doc 03 §5.3) and
    reproduces this function's behaviour from before `owners` existed. A
    step's *output* name is not unique across modules — that is the
    waterfall idiom, doc 03 §3.2 — so `compiled`/`step_fns` below are keyed
    by `(owner, step.name)`, never `step.name` alone, or the second
    module's entry silently overwrites the first's. The plain `step.name`
    key is *also* written, for the same name, as a courtesy to a caller
    that never supplied `owners` and only ever indexes `step_fns` by name
    (there is no collision to resolve in that case: it is the same
    fallback identity `_default_param_spaces` already assumes).
    """
    steps = list(steps)
    group_ids = list(group_ids)
    owners = list(owners) if owners is not None else [s.name for s in steps]
    if len(owners) != len(steps):
        raise ValueError("owners and steps must be the same length")
    key = _driver_key(
        steps, group_ids, owners, build_dir, terminal_names,
        parallel_group_ids, fastmath_group_ids,
    )
    cached = _DRIVER_CACHE.get(key)
    if cached is not None:
        return cached
    if len(steps) != len(group_ids):
        raise ValueError("steps and group_ids must be the same length")
    # `build_dir` no longer receives any file from this module (the fused
    # kernel is built in memory, `decider2.compile.kernel`); it stays in the
    # signature, and in `_driver_key` above, because every caller passes it
    # and two builds against different directories were never one Driver.
    sample_values = sample_values or {}
    terminal_names = frozenset(terminal_names)

    compiled = {(o, s.name): _try_njit(s, sample_values) for o, s in zip(owners, steps)}

    segments: list[Segment] = []
    step_fns: dict[Any, Callable] = {}
    i = 0
    n = len(steps)
    while i < n:
        step = steps[i]
        owner = owners[i]
        fn0, reason0 = compiled[(owner, step.name)]
        if reason0 is not None:
            step_fns[step.name] = step.fn
            step_fns[(owner, step.name)] = step.fn
            needed = _needed_from(steps, i + 1, terminal_names)
            segments.append(
                FallbackSegment(
                    steps=(step,),
                    owners=(owner,),
                    external_inputs=_external_inputs([step]),
                    required_outputs=(step.name,) if step.name in needed else (),
                    fallback_reason=reason0,
                )
            )
            i += 1
            continue

        if step.packed:
            # A tree/table/Branch/Loop step (doc 08 §3.4): `fn` is already
            # a real njit dispatcher (`_try_njit` returned it unprobed —
            # see that function's own docstring), so `compiled[(owner,
            # step.name)]` above IS `(step.fn, None)`, never a fallback
            # reason. Never joins a fusion run with a neighbour (see
            # `build_packed_kernel`'s docstring): each becomes its own
            # genuinely COMPILED (never Python-per-row) segment.
            step_fns[step.name] = fn0
            step_fns[(owner, step.name)] = fn0
            kernel_fn = build_packed_kernel(step)
            segments.append(
                PackedCompiledSegment(
                    steps=(step,),
                    owners=(owner,),
                    external_inputs=step.inputs,
                    required_outputs=(step.name,),
                    kernel_fn=kernel_fn,
                )
            )
            i += 1
            continue

        gid = group_ids[i]
        run: list[Step] = [step]
        run_owners: list[str] = [owner]
        j = i + 1
        while (
            j < n
            and group_ids[j] == gid
            and not steps[j].packed
            and compiled[(owners[j], steps[j].name)][1] is None
        ):
            run.append(steps[j])
            run_owners.append(owners[j])
            j += 1

        for s, o in zip(run, run_owners):
            fn = compiled[(o, s.name)][0]
            step_fns[s.name] = fn
            step_fns[(o, s.name)] = fn

        needed = _needed_from(steps, j, terminal_names)
        required = tuple(s.name for s in run if s.name in needed)
        external = _external_inputs(run)
        plan = KernelPlan(
            group_name=f"g{gid}_" + "_".join(s.name for s in run),
            steps=tuple(run),
            owners=tuple(run_owners),
            external_inputs=external,
            required_outputs=required,
            reads_shared=any(s.reads_shared for s in run),
            parallel=gid in parallel_group_ids,
            fastmath=gid in fastmath_group_ids,
        )
        # The kernel closes over each step's OWN dispatcher (the same one
        # `step_fns` holds for `stepped` mode), so a step defined inside
        # another function fuses like any other — the generated-source
        # strategy's "not reachable at module scope" fallback no longer has
        # a cause to exist.
        kernel_fn = build_fused_kernel(
            plan,
            [compiled[(o, s.name)][0] for s, o in zip(run, run_owners)],
            [_return_dtype(s) for s in run],
        )
        segments.append(
            CompiledSegment(
                steps=tuple(run),
                owners=tuple(run_owners),
                external_inputs=external,
                required_outputs=required,
                kernel_fn=kernel_fn,
                plan=plan,
            )
        )
        i = j

    driver = Driver(segments=tuple(segments), step_fns=step_fns)
    _DRIVER_CACHE[key] = driver
    return driver
