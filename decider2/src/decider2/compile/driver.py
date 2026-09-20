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
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

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

from decider2.compile import cache, codegen
from decider2.compile.codegen import KernelPlan
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
    """
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


@dataclass(frozen=True)
class Segment:
    """One contiguous run within a fuse()-group: either fully compiled, or
    (doc 05 §6) exactly one un-njit-able step running in Python. The blast
    radius of a bad node is the kernel it was going into, never the whole
    pipeline and never just that one node inside a still-compiled kernel —
    there is no such thing as the latter (EXPERIMENTS.md §B)."""

    kind: SegmentKind
    steps: tuple[Step, ...]
    external_inputs: tuple[Input, ...]
    required_outputs: tuple[str, ...]
    owners: tuple[str, ...] = ()             # module instance name per `steps` entry
    kernel_fn: Callable | None = None       # set when kind == "compiled"
    plan: KernelPlan | None = None          # set when kind == "compiled"
    fallback_reason: str | None = None      # set when kind == "fallback"


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
        (`decider2.compile.codegen`), never baked into generated source, so a
        value-only retune changes nothing this list depends on. Changing a
        field's *type* (e.g. a param toggling `float` <-> `float | None`)
        is the named negative control and *should* grow it by one.
        """
        out: list = []
        for seg in self.segments:
            if seg.kind == "compiled" and seg.kernel_fn is not None:
                out.extend(seg.kernel_fn.signatures)
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
    build_dir = Path(build_dir)
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
                Segment(
                    kind="fallback",
                    steps=(step,),
                    owners=(owner,),
                    external_inputs=_external_inputs([step]),
                    required_outputs=(step.name,) if step.name in needed else (),
                    fallback_reason=reason0,
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
        try:
            source = codegen.emit_kernel_source(plan)
            cached = cache.get_or_build(source, build_dir)
        except ImportError as exc:
            # The generated kernel FILE imports each step by
            # `fn.__module__`/`fn.__name__` (doc 05 §4.1: "every generated
            # driver is written to a real .py file before anything imports
            # it" — required for numba's cache to survive a fresh process,
            # EXPERIMENTS.md §J2). A step defined inside another function
            # (a closure, e.g. a test helper) njit-compiles just fine on
            # its own — `compiled[(o, s.name)]` above already proved that —
            # but has no module-level name that import line can reach.
            # That is a property of *this* fusion strategy, not a genuine
            # runtime bug in the step, so it gets the same treatment doc 05
            # §6 gives an un-njit-able step: split it out of the compiled
            # kernel rather than fail the whole build. `step_fns` already
            # holds each step's own (successfully compiled) dispatcher from
            # the loop just above, so `stepped`/`fused`'s fallback path
            # still runs compiled code per row, one step at a time — the
            # only thing lost is fusing this run into one kernel call.
            for s, o in zip(run, run_owners):
                s_required = (s.name,) if s.name in needed else ()
                segments.append(
                    Segment(
                        kind="fallback",
                        steps=(s,),
                        owners=(o,),
                        external_inputs=_external_inputs([s]),
                        required_outputs=s_required,
                        fallback_reason=(
                            f"kernel source could not import '{s.fn.__name__}' "
                            f"from '{s.fn.__module__}' ({exc!r}); it is not "
                            "reachable at module scope"
                        ),
                    )
                )
            i = j
            continue
        segments.append(
            Segment(
                kind="compiled",
                steps=tuple(run),
                owners=tuple(run_owners),
                external_inputs=external,
                required_outputs=required,
                kernel_fn=cached.module.kernel,
                plan=plan,
            )
        )
        i = j

    driver = Driver(segments=tuple(segments), step_fns=step_fns)
    _DRIVER_CACHE[key] = driver
    return driver
