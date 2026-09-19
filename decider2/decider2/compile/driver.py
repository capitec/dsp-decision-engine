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

from numba import njit
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
_PROBE_BY_ANNOTATION: dict[Any, Any] = {bool: True, int: 1, float: 1.0}


def _probe_value(annotation: Any) -> Any:
    return _PROBE_BY_ANNOTATION.get(annotation, 1.0)


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
        kwargs: dict[str, Any] = {}
        for pname in sig.parameters:
            if pname in sample_values:
                kwargs[pname] = sample_values[pname]
            elif pname in input_by_name:
                inp = input_by_name[pname]
                kwargs[pname] = (
                    None if inp.null_policy is NullPolicy.OPTIONAL else _probe_value(inp.annotation)
                )
            elif pname in param_by_name:
                kwargs[pname] = param_by_name[pname].default
        fn(**kwargs)
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


def build_driver(
    steps: Sequence[Step],
    group_ids: Sequence[int],
    *,
    build_dir: "str | Path",
    terminal_names: frozenset = frozenset(),
    parallel_group_ids: frozenset = frozenset(),
    fastmath_group_ids: frozenset = frozenset(),
    sample_values: Mapping[str, Any] | None = None,
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
    """
    steps = list(steps)
    group_ids = list(group_ids)
    if len(steps) != len(group_ids):
        raise ValueError("steps and group_ids must be the same length")
    build_dir = Path(build_dir)
    sample_values = sample_values or {}
    terminal_names = frozenset(terminal_names)

    compiled = {s.name: _try_njit(s, sample_values) for s in steps}

    segments: list[Segment] = []
    step_fns: dict[str, Callable] = {}
    i = 0
    n = len(steps)
    while i < n:
        step = steps[i]
        fn0, reason0 = compiled[step.name]
        if reason0 is not None:
            step_fns[step.name] = step.fn
            needed = _needed_from(steps, i + 1, terminal_names)
            segments.append(
                Segment(
                    kind="fallback",
                    steps=(step,),
                    external_inputs=_external_inputs([step]),
                    required_outputs=(step.name,) if step.name in needed else (),
                    fallback_reason=reason0,
                )
            )
            i += 1
            continue

        gid = group_ids[i]
        run: list[Step] = [step]
        j = i + 1
        while j < n and group_ids[j] == gid and compiled[steps[j].name][1] is None:
            run.append(steps[j])
            j += 1

        for s in run:
            step_fns[s.name] = compiled[s.name][0]

        needed = _needed_from(steps, j, terminal_names)
        required = tuple(s.name for s in run if s.name in needed)
        external = _external_inputs(run)
        plan = KernelPlan(
            group_name=f"g{gid}_" + "_".join(s.name for s in run),
            steps=tuple(run),
            external_inputs=external,
            required_outputs=required,
            reads_shared=any(s.reads_shared for s in run),
            parallel=gid in parallel_group_ids,
            fastmath=gid in fastmath_group_ids,
        )
        source = codegen.emit_kernel_source(plan)
        cached = cache.get_or_build(source, build_dir)
        segments.append(
            Segment(
                kind="compiled",
                steps=tuple(run),
                external_inputs=external,
                required_outputs=required,
                kernel_fn=cached.module.kernel,
                plan=plan,
            )
        )
        i = j

    return Driver(segments=tuple(segments), step_fns=step_fns)
