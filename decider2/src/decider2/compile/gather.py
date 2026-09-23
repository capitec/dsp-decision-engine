"""Row gathering for packed steps — the machinery `decider2.compile.driver`
drives a `types.Step.packed` step through: how one row's `args` is shaped
(`_packed_args_kind`), the compiled per-row gathers (`_gather0`/
`_gather1_raw`/`_gather_array`/`_fill_array`), the typed-row family
(`_typed_*`, `_fill_spans`), the whole-column normalisation done once per
`PackedCompiledSegment.run()` (`_packed_input_arrays`), the per-step
compiled loops (`build_packed_kernel`, `_build_typed_kernel`) and the
plain-Python row-at-a-time counterpart the `interpreted`/`stepped` rungs
use (`_packed_row_args`).

Moved out of `decider2.compile.driver` unchanged (docs/BOUNDARY-REWORK.md,
Stage 1b) so the boundary stages can edit the two halves — typed rows here,
`Segment` construction and dispatch over there — without sharing a file.
`driver` imports everything back and re-exports it, so the names it always
exposed are still importable from it; the explanatory comments below say
"this module" and mean the driver-and-gather pair, as they did before the
split.
"""
from __future__ import annotations

import typing
from typing import Any, Callable, Sequence

import numpy as np
from numba import njit

from decider2.types import FeatureKind, Step, feature_kind


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
# keep their own dtypes: `args` is a 5-tuple of per-kind row arrays in
# `FeatureKind` order, `params` the `(floats, ints, pat_bytes, pat_off,
# grp_off)` bundle. Same "homogeneous tuple indexed at a runtime position"
# mechanism as `_fill_array` above, applied once per kind instead of once —
# so there is still no width past which anything raises, and an Int64
# column is compared as an int64 (doc 03 §1.2) rather than rounded through
# float64 on the way in.
#
# A `bytes`-annotated (`FeatureKind.STR`) input's COLUMN is a span table:
# an `(n, 2)` int64 array of `(address, byte length)` per row, length -1
# for a null, produced by `boundary.extract` (the STR slot of the
# whole-frame gather, `_arrow.frame.materialize_columns`) through the compiled
# Arrow shim for `apply()` and by `runtime.invoke.score` from
# `str.encode("utf-8")` for a record (docs/BOUNDARY-REWORK.md §2.1). The
# addresses are only valid while whoever built the table keeps the bytes
# alive — the `ExtractedFrame` for a batch, the record's own `bytes` for a
# score — which every caller does for the duration of the run.
# ---------------------------------------------------------------------------

_TYPED_DTYPES: tuple[np.dtype, ...] = (
    np.dtype(np.float64),   # F64
    np.dtype(np.int64),     # I64
    np.dtype(np.bool_),     # BOOL
    np.dtype(np.int32),     # CODE
    np.dtype(np.int64),     # STR — the (n, 2) span table
)


def _readonly_empty(dtype: np.dtype, shape=(0,)) -> np.ndarray:
    arr = np.empty(shape, dtype=dtype)
    arr.flags.writeable = False
    return arr


# One inert, zero-length, read-only array per kind: the padding entry for a
# kind with no inputs (see `_fill_array`). Read-only so it is the SAME numba
# type as a real column would be (`_as_readonly` below); the STR dummy is
# 2-D like a real span table, or the tuple would not be homogeneous.
_TYPED_DUMMIES: tuple[np.ndarray, ...] = tuple(
    _readonly_empty(d, (0, 2) if kind is FeatureKind.STR else (0,))
    for kind, d in zip(FeatureKind, _TYPED_DTYPES)
)

def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


def _pattern_table(groups: Sequence[Sequence[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(pat_bytes, pat_off, grp_off)` for `groups`, each group the
    patterns of one `str`/`list[str]` param, in param order: pattern `p`
    is `pat_bytes[pat_off[p]:pat_off[p+1]]`, group `g` is patterns
    `grp_off[g]..grp_off[g+1]`. UTF-8 bytes, so a multi-byte pattern
    matches exactly its own byte sequence at the node.

    All three are 1-D, C-contiguous and READ-ONLY whether they hold zero
    patterns or a thousand (`np.frombuffer` over a `bytes` is read-only by
    construction; the offsets are made so), so the numba type of the bundle
    never depends on how many patterns there are or how long — the reason
    a pattern edit is a value change (`types.Step.typed_args`)."""
    encoded: list[bytes] = []
    grp_off = [0]
    for group in groups:
        for pattern in group:
            if not isinstance(pattern, str):
                raise TypeError(
                    f"a string pattern must be a str, got {type(pattern).__name__} {pattern!r}"
                )
            encoded.append(pattern.encode("utf-8"))
        grp_off.append(len(encoded))
    pat_off = np.zeros(len(encoded) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in encoded], out=pat_off[1:])
    pat_bytes = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    return pat_bytes, _readonly(pat_off), _readonly(np.array(grp_off, dtype=np.int64))


def _is_pattern_list(annotation: Any) -> bool:
    """A `list[str]`-annotated param: one GROUP of patterns (`decider2.
    trees.encode` declares a string-match node's literals this way, so the
    group's size is a value)."""
    return typing.get_origin(annotation) is list and typing.get_args(annotation) == (str,)


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
    """`(f64_cols, i64_cols, bool_cols, code_cols, span_tables)` — one
    homogeneous tuple of whole-column arrays per kind, each padded with
    its kind's dummy when empty. A `bytes`-annotated input reads
    `registry[name]` as its `(n, 2)` span table (see the section comment
    above); anything else there is a producer bug and is refused by name
    rather than handed to a kernel that would read it as addresses."""
    groups: list[list[np.ndarray]] = [[] for _ in FeatureKind]
    for inp, kind in zip(step.inputs, _typed_layout(step)):
        arr = registry[inp.name]
        if kind is FeatureKind.STR and not (arr.ndim == 2 and arr.shape[1] == 2):
            raise TypeError(
                f"typed step '{step.name}': input '{inp.name}' is declared bytes (a "
                f"string span), so its column must be an (n, 2) int64 span table; got "
                f"an array of shape {arr.shape} and dtype {arr.dtype}."
            )
        groups[kind].append(_as_readonly(arr, _TYPED_DTYPES[kind]))
    return tuple(
        tuple(g) if g else (_TYPED_DUMMIES[kind],) for kind, g in zip(FeatureKind, groups)
    )


def _typed_params(step: Step, values: Sequence[Any]) -> tuple:
    """`(floats, ints, pat_bytes, pat_off, grp_off)` — `values` (in
    `step.params` order) grouped by each `ParamDecl.annotation`, so the
    kernel can index each group at a runtime position (a homogeneous
    tuple) without slicing a mixed one (which numba cannot do from a
    closure constant — verified). Position within a group is first-
    appearance order in `step.params`, the rule the producer's own slot
    numbering follows. A `bool`-annotated param rides in the int group as
    0/1; a `str` param is one pattern GROUP of one pattern, a `list[str]`
    param one group of as many patterns as the list holds — both in the
    pattern table (`_pattern_table`), whose numba type does not depend on
    the count."""
    floats: list = []
    ints: list = []
    groups: list[Sequence[str]] = []
    for decl, value in zip(step.params, values):
        ann = decl.annotation if decl.annotation is not Any else type(decl.default)
        if ann is float:
            floats.append(float(value))
        elif ann in (int, bool):
            ints.append(int(value))
        elif ann is str:
            groups.append((value,))
        elif _is_pattern_list(ann):
            groups.append(tuple(value))
        else:
            raise ValueError(
                f"typed step '{step.name}': param '{decl.name}' is annotated "
                f"{ann!r}; a typed packed step takes float/int/bool/str/list[str] params only."
            )
    table = _pattern_table(groups) if groups else _EMPTY_PATTERNS
    return (tuple(floats), tuple(ints)) + table


# The pattern table of a step with no string patterns: no bytes, one
# pattern offset, one group offset — the same three types `_pattern_table`
# produces for a populated one (1-D, C, read-only), so the kernel sees one
# signature either way.
_EMPTY_PATTERNS: tuple[np.ndarray, np.ndarray, np.ndarray] = _pattern_table(())


@njit(cache=True)
def _fill_spans(tables, i, out):
    """The string slot's row gather: `out[2k], out[2k+1]` = the
    `(address, length)` span of string feature `k` on row `i`, read off
    its `(n, 2)` span table. Never dereferences the bytes — that is the
    walker's `STR_MATCH` node's job (`FeatureKind.STR`)."""
    for k in range(len(out) // 2):
        out[2 * k] = tables[k][i, 0]
        out[2 * k + 1] = tables[k][i, 1]
    return out


def _typed_row_args(arrays: tuple, counts: Sequence[int], i: int) -> tuple:
    """One row of a typed step as `_packed_row_args` needs it — the plain-
    Python (interpreted/stepped) counterpart of the kernel's own per-row
    gather in `build_packed_kernel`; same five-tuple, same slot rule.
    `counts[kind]` (from `_typed_counts`) says how many REAL columns each
    kind's tuple holds, so a padding dummy is never read. The spans come
    from the SAME table the fused kernel reads (BOUNDARY-REWORK.md §1.6:
    one import path, not two — the independent producer is `score()`)."""
    f_cols, i_cols, b_cols, c_cols, s_cols = arrays
    nf, ni, nb, nc, ns = counts
    return (
        np.array([f_cols[k][i] for k in range(nf)], dtype=np.float64),
        np.array([i_cols[k][i] for k in range(ni)], dtype=np.int64),
        np.array([b_cols[k][i] for k in range(nb)], dtype=np.bool_),
        np.array([c_cols[k][i] for k in range(nc)], dtype=np.int32),
        np.array([v for k in range(ns) for v in (s_cols[k][i, 0], s_cols[k][i, 1])], dtype=np.int64),
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
    """`build_packed_kernel` for a `typed_args` step: five row buffers
    hoisted above the loop (one per kind, the string span buffer holding
    two int64s per string feature), each refilled per row from its own
    homogeneous column tuple, then ONE call into `fn` with the five-tuple.
    Zero allocations per row, as before; the extra cost over the single
    float64 gather is the four (mostly empty) extra fill loops and the
    tuple build — measured in `evaluation/typed-features`."""
    n_f, n_i, n_b, n_c, n_s = _typed_counts(step)
    n_s2 = 2 * n_s
    fn = step.fn

    # The row five-tuple is built ONCE, above the loop (its members are the
    # hoisted buffers, refilled in place per row): building a tuple of
    # arrays per row measured ~+75 ns/row, the NRT bookkeeping on each
    # member. `fn` is expected to be `inline="always"` (a tree's `path_fn`
    # is) so the tuple never crosses a real call boundary either — see
    # `trees.interpreter.walk_tree` for the measured cost when it does.
    if step.reads_shared:
        @njit
        def kernel(arrays, params, shared, n, out):
            f_cols, i_cols, b_cols, c_cols, s_cols = arrays
            bf = np.empty(n_f, dtype=np.float64)
            bi = np.empty(n_i, dtype=np.int64)
            bb = np.empty(n_b, dtype=np.bool_)
            bc = np.empty(n_c, dtype=np.int32)
            bs = np.empty(n_s2, dtype=np.int64)
            row = (bf, bi, bb, bc, bs)
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
        f_cols, i_cols, b_cols, c_cols, s_cols = arrays
        bf = np.empty(n_f, dtype=np.float64)
        bi = np.empty(n_i, dtype=np.int64)
        bb = np.empty(n_b, dtype=np.bool_)
        bc = np.empty(n_c, dtype=np.int32)
        bs = np.empty(n_s2, dtype=np.int64)
        row = (bf, bi, bb, bc, bs)
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
        # Same five-tuple / params bundle the fused kernel builds
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


# `_packed_row_args` above calls `_scalar_arg` and is annotated with
# `ResolvedParams`, both of which belong to the generic per-row calling
# convention that stays in `decider2.compile.driver` — and `driver` imports
# this module. The import sits at the BOTTOM so the cycle resolves in either
# order: whichever module is executed first, every name the other one needs
# is already defined by the time the cross-import runs (`driver` imports this
# module mid-file, after `_scalar_arg`, for the same reason).
from decider2.compile.driver import ResolvedParams, _scalar_arg  # noqa: E402
