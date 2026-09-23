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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as _dc_field
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
from decider2.types import FeatureKind, Input, NullPolicy, Step, feature_kind

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


# ---------------------------------------------------------------------------
# Packed-step kernels — a real, hand-written per-row compiled loop for a
# `types.Step.packed` step (a tree/table/Branch/Loop `Step`, doc 08 §3.4).
# A packed `fn` has the generic `(args, params[, shared])` calling
# convention rather than one named parameter per input, so it is driven by
# the row-gathering machinery below instead of `decider2.compile.kernel`'s
# fused kernel, which binds a step's arguments by its Python signature's
# parameter names.
#
# **A real, reported scope cut, not full fusion.** Each packed step becomes
# its OWN compiled segment — this does not fuse a tree's own matcher/path/
# output steps into ONE kernel call the way `compile.kernel.
# build_fused_kernel` fuses several ordinary steps sharing a `fuse()` group.
# Building that fusion generically means teaching that kernel's row body a
# second calling convention (fill a row buffer, call `fn(args, params)`),
# and was judged too large an addition to risk finishing untested in the
# pass that introduced packed steps.
#
# **Row gathering has no arity ceiling.** `args[k]` for `Step.inputs[k]`, `k`
# a RUNTIME row index, used to be one hand-written closure body per input
# COUNT (`_compose_args1`..`_compose_args16`, a bare dict lookup past which
# a 17-input step raised `ValueError` — the wall this pass exists to
# remove). A homogeneous tuple indexed by a runtime variable compiles and
# runs in nopython mode (verified; see this module's report), so once every
# per-input array is the SAME dtype, gathering one row is ONE small generic
# loop, never one function per width. `_make_row_gatherer` below picks
# between two FIXED shapes — never one per input count:
#
# - exactly one, non-float-annotated input -> `_gather1_raw`, a 1-tuple
#   preserving that input's own dtype EXACTLY (a hoisted string matcher's
#   own int32 code, or a path/row step's own int64 result — some packed
#   `fn`s index an array with this value, e.g. `values[idx]`, which a
#   forced float64 cast would silently break, doc 00 §2);
# - everything else (zero inputs, or every input float-annotated — a
#   tree's `feats`, a table's `vars_`, doc 08 §3.4's homogeneous UniTuples,
#   now a homogeneous float64 ARRAY instead) -> `_gather_array`/`_gather0`,
#   one runtime-indexed loop, however wide the step actually is.
#
# The float64-ness the array case relies on is established ONCE per
# `PackedCompiledSegment.run()` call, outside the per-row loop entirely
# (`_packed_input_arrays`, below) — a whole-column `np.ndarray.astype`
# rather than a per-row cast repeated `n` times — which is also what makes
# `arrays` (the tuple of per-input COLUMN arrays `_gather_array` indexes at
# a runtime position) homogeneous in the first place: a homogeneous tuple
# indexed by a runtime variable compiles in nopython mode; a heterogeneous
# one (mixed dtypes) does not, without `numba.literal_unroll`.
# ---------------------------------------------------------------------------


def _packed_args_kind(annotations: Sequence[Any]) -> str:
    """Which row-argument SHAPE a packed step's `(args, params)` call
    needs, decided once from `Step.inputs`' declared annotations — never
    per row, and never a family keyed on `len(annotations)`.

    `"raw1"`: exactly one input, not float-annotated — see this module's
    docstring above for why its dtype must survive untouched. `"array"`:
    everything else — zero inputs, or every input float-annotated —
    gathered into one homogeneous float64 array (`_gather_array`/
    `_gather0`), however many inputs there are.

    Raises for the one shape this can't represent: more than one input
    where not everything is float-annotated, which has no single
    homogeneous container to hold it. Every current producer of a
    multi-input packed step (`decider2.trees.encode`, `decider2.tables.
    encode`) already guarantees this never arises; a future one that needs
    a genuine int/float/str mix wants its own row-gathering, not this
    generic one.
    """
    n = len(annotations)
    if n == 1 and annotations[0] is not float:
        return "raw1"
    if n != 0 and any(a is not float for a in annotations):
        raise ValueError(
            f"a packed step with {n} inputs must have every input float-"
            "annotated once more than one input is gathered together "
            "(decider2.compile.driver._packed_args_kind); got annotations "
            f"{annotations!r}."
        )
    return "array"


@njit(cache=True)
def _gather0(arrays, i):
    """Zero inputs. Never indexes `arrays` (an empty tuple has no element
    type numba can infer for a dynamic read, even one never actually
    reached — the same reasoning `trees.encode._nonempty` documents), so
    this is its own case rather than a `len(arrays) == 0` branch inside
    `_gather_array`."""
    del arrays, i
    return np.empty(0, dtype=np.float64)


@njit(cache=True)
def _gather1_raw(arrays, i):
    """Exactly one input, not float-annotated — `arrays[0]` at a LITERAL
    (not runtime) index, so its own dtype survives untouched regardless of
    what it is."""
    return (arrays[0][i],)


@njit(cache=True)
def _gather_array(arrays, i):
    """One row, every input — `out[k] = arrays[k][i]` for `k` a RUNTIME
    index, over a homogeneous tuple of same-dtype (float64) 1D arrays.
    Replaces `_compose_args1`..`_compose_args16`: one generic loop instead
    of one hand-written function body per input count, so there is no
    width past which this raises. `arrays` is guaranteed homogeneous by
    `_packed_input_arrays`, below — nothing in here casts anything.

    Used by `_make_row_gatherer` for the `interpreted`/`stepped` row-at-a-
    time path (`_packed_row_args`), where a fresh array per call is the
    only option anyway. The `fused` per-row loop (`build_packed_kernel`)
    does NOT call this — seee `_fill_array` below for why."""
    n = len(arrays)
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        out[k] = arrays[k][i]
    return out


@njit(cache=True)
def _fill_array(arrays, i, out):
    """`_gather_array`'s own loop body, writing into a CALLER-OWNED buffer
    instead of allocating a fresh one — `build_packed_kernel`'s per-row
    `fused` loop hoists ONE buffer above the loop and reuses it every row
    (each row's values are fully consumed by `fn(...)` before the next
    row overwrites them, so reuse is safe), trading one heap allocation
    per row for zero. Measured: on a realistic 16-feature tree this is
    what keeps the array-based row gather (no arity ceiling) close to the
    previous per-count closure family's per-row cost instead of well
    behind it — see this module's report."""
    # `len(out)`, not `len(arrays)`: a typed step's per-kind tuple is padded
    # with one zero-length dummy when that kind has no inputs (an empty
    # tuple has no element type numba can infer for a runtime-indexed
    # read, `_gather0`'s own reason to exist), and `out` is the honest
    # count. Identical for the plain float64 case, where the two agree.
    for k in range(len(out)):
        out[k] = arrays[k][i]
    return out


# ---------------------------------------------------------------------------
# Typed row gather — `types.Step.typed_args`. A packed step whose inputs
# keep their own dtypes: `args` is a 6-tuple of per-kind row arrays in
# `FeatureKind` order, `params` a `(floats, ints)` pair. Same "homogeneous
# tuple indexed at a runtime position" mechanism as `_fill_array` above,
# applied once per kind instead of once — so there is still no width past
# which anything raises, and an Int64 column is compared as an int64
# (doc 03 §1.2) rather than rounded through float64 on the way in.
# ---------------------------------------------------------------------------

_TYPED_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.float64),   # F64
    np.dtype(np.int64),     # I64
    np.dtype(np.bool_),     # BOOL
    np.dtype(np.int32),     # CODE
    np.dtype(np.int64),     # STR — offsets into the byte buffer
)


def _readonly_empty(dtype: np.dtype) -> np.ndarray:
    arr = np.empty(0, dtype=dtype)
    arr.flags.writeable = False
    return arr


# One inert, zero-length, read-only array per kind: the padding entry for a
# kind with no inputs (see `_fill_array`). Read-only so it is the SAME numba
# type as a real column would be (`_as_readonly` below).
_TYPED_DUMMIES: tuple[np.ndarray, ...] = tuple(_readonly_empty(d) for d in _TYPED_DTYPES)
_BYTES_DUMMY: np.ndarray = _readonly_empty(np.dtype(np.uint8))


def _typed_layout(step: Step) -> tuple[FeatureKind, ...]:
    """`inputs[k]`'s kind, in order. Slot `j` of kind `K` is the `j`-th
    input of kind `K` here — the only rule, shared with the producer."""
    return tuple(feature_kind(inp.annotation) for inp in step.inputs)


def _typed_counts(step: Step) -> tuple[int, ...]:
    layout = _typed_layout(step)
    return tuple(sum(1 for k in layout if k == kind) for kind in FeatureKind)


def _as_readonly(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """`_as_readonly_f64`, for any dtype — same reasoning, same view trick."""
    out = arr if arr.dtype == dtype else arr.astype(dtype)
    out = out.view() if out is arr else out
    out.flags.writeable = False
    return out


def _typed_input_arrays(step: Step, registry: dict) -> tuple:
    """`(f64_cols, i64_cols, bool_cols, code_cols, str_offsets, str_bytes)`
    — one homogeneous tuple of whole-column arrays per kind, each padded
    with its kind's dummy when empty, plus the raw-string byte buffers.
    A `bytes`-annotated input reads `registry[name]` as the column's
    `offsets` (int64, length n+1) and `registry["__bytes__" + name]` as
    its `values` (uint8) — the two arrays polars' `_get_buffers()` hands
    back for a String column, untouched."""
    groups: list[list[np.ndarray]] = [[] for _ in FeatureKind]
    byte_bufs: list[np.ndarray] = []
    for inp, kind in zip(step.inputs, _typed_layout(step)):
        groups[kind].append(_as_readonly(registry[inp.name], _TYPED_DTYPES[kind]))
        if kind is FeatureKind.STR:
            byte_bufs.append(_as_readonly(registry[f"__bytes__{inp.name}"], np.dtype(np.uint8)))
    cols = tuple(
        tuple(g) if g else (_TYPED_DUMMIES[kind],) for kind, g in zip(FeatureKind, groups)
    )
    return cols + ((tuple(byte_bufs) if byte_bufs else (_BYTES_DUMMY,)),)


def _typed_params(step: Step, values: Sequence[Any]) -> tuple:
    """`(floats, ints)` — `values` (in `step.params` order) grouped by each
    `ParamDecl.annotation`, so the kernel can index each group at a
    runtime position (a homogeneous tuple) without slicing a mixed one
    (which numba cannot do from a closure constant — verified). Position
    within a group is first-appearance order in `step.params`, the rule
    the producer's own slot numbering follows. A `bool`-annotated param
    rides in the int group as 0/1."""
    floats: list = []
    ints: list = []
    for decl, value in zip(step.params, values):
        ann = decl.annotation if decl.annotation is not Any else type(decl.default)
        if ann is float:
            floats.append(float(value))
        elif ann in (int, bool):
            ints.append(int(value))
        else:
            raise ValueError(
                f"typed step '{step.name}': param '{decl.name}' is annotated "
                f"{ann!r}; a typed packed step takes float/int/bool params only."
            )
    return tuple(floats), tuple(ints)


@njit(cache=True)
def _fill_spans(offsets, i, out):
    """The raw-string slot's row gather: `out[2k], out[2k+1]` = the byte
    range of string feature `k` on row `i`, read straight off its polars
    `offsets` buffer. Never dereferences the bytes — that is a matcher's
    job (`FeatureKind.STR`)."""
    for k in range(len(out) // 2):
        out[2 * k] = offsets[k][i]
        out[2 * k + 1] = offsets[k][i + 1]
    return out


def _typed_row_args(arrays: tuple, counts: Sequence[int], i: int) -> tuple:
    """One row of a typed step as `_packed_row_args` needs it — the plain-
    Python (interpreted/stepped) counterpart of the kernel's own per-row
    gather in `build_packed_kernel`; same six-tuple, same slot rule.
    `counts[kind]` (from `_typed_counts`) says how many REAL columns each
    kind's tuple holds, so a padding dummy is never read."""
    f_cols, i_cols, b_cols, c_cols, s_cols, s_bytes = arrays
    nf, ni, nb, nc, ns = counts
    return (
        np.array([f_cols[k][i] for k in range(nf)], dtype=np.float64),
        np.array([i_cols[k][i] for k in range(ni)], dtype=np.int64),
        np.array([b_cols[k][i] for k in range(nb)], dtype=np.bool_),
        np.array([c_cols[k][i] for k in range(nc)], dtype=np.int32),
        np.array([v for k in range(ns) for v in (s_cols[k][i], s_cols[k][i + 1])], dtype=np.int64),
        s_bytes,
    )


def _make_row_gatherer(annotations: Sequence[Any]) -> Callable:
    """`(arrays, i) -> args`, exactly as wide as `step.inputs` (never
    padded: `step.fn` needs `len(args) == len(step.inputs)` precisely — a
    tree's `walk_tree` call, a table's `scan_table` call and a matcher's
    `for i in range(len(params))` loop all size themselves off the
    container they are actually handed)."""
    if len(annotations) == 0:
        return _gather0
    return _gather1_raw if _packed_args_kind(annotations) == "raw1" else _gather_array


def _as_readonly_f64(arr: np.ndarray) -> np.ndarray:
    """`arr`, as a float64 array numba sees as READ-ONLY — never mutating
    `arr` itself. `_gather_array` needs every entry of its `arrays` tuple
    to be the exact SAME numba array type for a runtime-indexed `arrays[k]`
    to type-check (a homogeneous tuple); numba's array type carries BOTH
    dtype AND a read-only bit, so leaving one column's own
    `flags.writeable` as-is (a plain, owned column stays writable; a
    column read straight off a polars-backed buffer often is not) makes
    two float64 columns two DIFFERENT types — confirmed empirically: a
    table with one such mismatched pair raised `TypingError: No
    implementation of function ... getitem ... Tuple(readonly array(
    float64, 1d, C), array(float64, 1d, C))`. `.view()` + forcing the
    VIEW's own `writeable` flag off (never the base's — verified: this
    does not affect `arr.flags.writeable`) makes an already-float64 column
    read-only for free, no copy; a wrongly-dtyped column still needs one
    real cast (`.astype`, which returns an already-fresh, independent
    array to mark read-only)."""
    out = arr if arr.dtype == np.float64 else arr.astype(np.float64)
    out = out.view() if out is arr else out
    out.flags.writeable = False
    return out


def _packed_input_arrays(step: Step, registry: dict) -> tuple:
    """The per-input COLUMN arrays `build_packed_kernel`'s kernel indexes
    at a runtime row `i` — normalised to float64 and read-only ONCE, for
    the whole column (`_as_readonly_f64`), exactly when `_gather_array` is
    what is going to read them at a runtime position; passed through with
    its own dtype (and writability) untouched for the single, non-float
    `_gather1_raw` input instead (`_packed_args_kind`). A whole-column
    numpy cast here, outside the per-row loop, replaces the per-row
    `_as_float`/`_as_is` cast the previous, per-input-count closure family
    did inside it — same total work, done once per column instead of once
    per row per column."""
    if step.typed_args:
        return _typed_input_arrays(step, registry)
    inputs = step.inputs
    if len(inputs) == 1 and inputs[0].annotation is not float:
        return (registry[inputs[0].name],)
    return tuple(_as_readonly_f64(registry[inp.name]) for inp in inputs)


def _build_typed_kernel(step: Step) -> Callable:
    """`build_packed_kernel` for a `typed_args` step: six row buffers
    hoisted above the loop (one per kind, the raw-string span buffer
    holding two int64s per string feature), each refilled per row from
    its own homogeneous column tuple, then ONE call into `fn` with the
    six-tuple. Zero allocations per row, as before; the extra cost over
    the single float64 gather is the five (mostly empty) extra fill loops
    and the tuple build — measured in `evaluation/typed-features`."""
    n_f, n_i, n_b, n_c, n_s = _typed_counts(step)
    n_s2 = 2 * n_s
    fn = step.fn

    # The row six-tuple is built ONCE, above the loop (its members are the
    # hoisted buffers, refilled in place per row): building a tuple of six
    # arrays per row measured ~+75 ns/row, the NRT bookkeeping on each
    # member. `fn` is expected to be `inline="always"` (a tree's `path_fn`
    # is) so the tuple never crosses a real call boundary either — see
    # `trees.interpreter.walk_tree` for the measured cost when it does.
    if step.reads_shared:
        @njit
        def kernel(arrays, params, shared, n, out):
            f_cols, i_cols, b_cols, c_cols, s_cols, s_bytes = arrays
            bf = np.empty(n_f, dtype=np.float64)
            bi = np.empty(n_i, dtype=np.int64)
            bb = np.empty(n_b, dtype=np.bool_)
            bc = np.empty(n_c, dtype=np.int32)
            bs = np.empty(n_s2, dtype=np.int64)
            row = (bf, bi, bb, bc, bs, s_bytes)
            for i in range(n):
                _fill_array(f_cols, i, bf)
                _fill_array(i_cols, i, bi)
                _fill_array(b_cols, i, bb)
                _fill_array(c_cols, i, bc)
                _fill_spans(s_cols, i, bs)
                out[i] = fn(row, params, shared)
        return kernel

    @njit
    def kernel(arrays, params, n, out):
        f_cols, i_cols, b_cols, c_cols, s_cols, s_bytes = arrays
        bf = np.empty(n_f, dtype=np.float64)
        bi = np.empty(n_i, dtype=np.int64)
        bb = np.empty(n_b, dtype=np.bool_)
        bc = np.empty(n_c, dtype=np.int32)
        bs = np.empty(n_s2, dtype=np.int64)
        row = (bf, bi, bb, bc, bs, s_bytes)
        for i in range(n):
            _fill_array(f_cols, i, bf)
            _fill_array(i_cols, i, bi)
            _fill_array(b_cols, i, bb)
            _fill_array(c_cols, i, bc)
            _fill_spans(s_cols, i, bs)
            out[i] = fn(row, params)
    return kernel


def build_packed_kernel(step: Step) -> Callable:
    """A genuinely compiled, `@njit`'d per-row loop for one packed step —
    `kernel(arrays, params, n, out)` (or `kernel(arrays, params, shared, n,
    out)` when `step.reads_shared`) — closing over `step.fn` (already an
    njit dispatcher) directly rather than re-importing it from a generated
    file. `arrays` is a tuple of this step's own external input arrays, in
    `step.inputs` order (`_packed_input_arrays`, above); `params` is the
    already-resolved params tuple for this one `apply()`/`score()` call
    (built once, doc 08 §2: retuning it is a values change, so it is a
    plain argument here too, never baked into the kernel). No `step.
    inputs` count raises here any more — see this module's report."""
    if step.typed_args:
        return _build_typed_kernel(step)

    annotations = tuple(inp.annotation for inp in step.inputs)
    n_inputs = len(annotations)
    fn = step.fn

    if n_inputs > 0 and _packed_args_kind(annotations) == "array":
        # The hot case (a tree's `feats`, a table's `vars_`): hoist ONE row
        # buffer above the per-row loop and reuse it every row instead of
        # letting `_gather_array` allocate a fresh one per row (`_fill_
        # array`'s own docstring) -- measured to matter on a realistic
        # body, see this module's report.
        if step.reads_shared:
            @njit
            def kernel(arrays, params, shared, n, out):
                buf = np.empty(n_inputs, dtype=np.float64)
                for i in range(n):
                    out[i] = fn(_fill_array(arrays, i, buf), params, shared)
            return kernel

        @njit
        def kernel(arrays, params, n, out):
            buf = np.empty(n_inputs, dtype=np.float64)
            for i in range(n):
                out[i] = fn(_fill_array(arrays, i, buf), params)
        return kernel

    # Zero inputs, or the single non-float input: `_gather0`/`_gather1_raw`
    # are already allocation-free (an empty array, a 1-tuple), so there is
    # no per-row buffer worth hoisting here.
    gather = _gather0 if n_inputs == 0 else _gather1_raw

    if step.reads_shared:
        @njit
        def kernel(arrays, params, shared, n, out):
            for i in range(n):
                out[i] = fn(gather(arrays, i), params, shared)
        return kernel

    @njit
    def kernel(arrays, params, n, out):
        for i in range(n):
            out[i] = fn(gather(arrays, i), params)
    return kernel


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
    `@njit(cache=True)`), so there is nothing here for `_FALLBACK_TRIGGERS`
    to ever legitimately catch.
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


def _packed_row_args(
    step: Step, owner: "str | None", registry: dict, resolved: ResolvedParams, i: int,
) -> "tuple[Any, tuple]":
    """`(args, params)` for one row of a `packed` step (`types.Step.
    packed`) — `args[k]` is `step.inputs[k]`'s value at row `i`,
    `params[k]` is `step.params[k]`'s value, both in declared order. A
    packed step never declares OPTIONAL/`reads_params` — doc 08 §3.4's
    data-shaped interiors have no use for them — so this reads
    `registry[inp.name][i]` directly, with no validity/bundle branching to
    do. `reads_shared` IS used (a table's row/output steps, doc 08 §3.4's
    free-interior row data) — see `_call_step_row`, which appends `shared`
    as a third positional argument when `step.reads_shared`.

    `args`' own SHAPE (a 1-tuple preserving one non-float input's exact
    dtype, or a float64 array otherwise) matches `_packed_args_kind`/
    `_make_row_gatherer`'s fused-kernel convention exactly, not by
    coincidence: `step.fn` is the SAME njit dispatcher either way
    (`interpreted`/`stepped` call it straight from here, `fused` calls it
    through `build_packed_kernel`'s per-row loop), so calling it with a
    DIFFERENT argument shape per mode would just compile a second,
    needless specialisation of it rather than reuse the one `fused` mode
    already warmed.
    """
    values = tuple(_scalar_arg(resolved, owner, step.name, p.name) for p in step.params)
    if step.typed_args:
        # Same six-tuple / `(floats, ints)` pair the fused kernel builds
        # (`_build_typed_kernel`), so `step.fn` compiles ONE specialisation
        # that every mode shares — the reason the plain path below also
        # mirrors its fused counterpart's shape exactly.
        arrays = _typed_input_arrays(step, registry)
        return _typed_row_args(arrays, _typed_counts(step), i), _typed_params(step, values)
    inputs = step.inputs
    if len(inputs) == 1 and inputs[0].annotation is not float:
        args: Any = (registry[inputs[0].name][i],)
    else:
        args = np.array([float(registry[inp.name][i]) for inp in inputs], dtype=np.float64)
    return args, values


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
    plan: KernelPlan, registry: dict, resolved: ResolvedParams, out_arrays: dict, n: int
) -> tuple:
    """`(n, cols, valids, params_all, outs)` — the fused kernel's five
    arguments (`compile.kernel.build_fused_kernel`), each tuple packed in
    `kernel_signature` order so the kernel's own argument-source map, built
    from the same list, indexes the right element."""
    cols: list = []
    valids: list = []
    params_all: list = []
    outs: list = []
    for role in kernel_signature(plan):
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

    def run(self, registry: dict, resolved: ResolvedParams, n: int) -> None:
        # Doc 05 §7: "intermediates ... stay in numpy, they do not
        # round-trip through polars" — a value a later segment needs is
        # looked up in `registry` regardless of whether it came from the
        # original frame or an earlier segment's own output.
        owner_by_name = dict(zip((s.name for s in self.steps), self.owners)) if self.owners else {}
        step_by_name = {s.name: s for s in self.steps}
        out_arrays = {
            name: np.empty(n, dtype=_return_dtype(step_by_name[name]))
            for name in self.required_outputs
        }
        self.kernel_fn(*_build_call_args(self.plan, registry, resolved, out_arrays, n))
        for name, arr in out_arrays.items():
            registry[name] = arr
            registry[f"{name}@{owner_by_name.get(name, name)}"] = arr

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
