"""Getting data out of polars — the whole frame, ONCE, through Arrow.

docs/BOUNDARY-REWORK.md §1 (Stage 3). `extract_frame` is:

    route REQUIRED nulls (nulls.py)              whole frame, before export
      -> frame-tier casts (dtypes.py)           only the declared columns whose
                                                dtype the kind cannot read as is
      -> FrameView.bind(kernel_frame)           df.__arrow_c_stream__(): one
                                                capsule, polars rechunks in place
      -> FrameView.materialize_columns()        sm_resolve_all once, then
                                                sm_gather_row ONCE PER ROW, the
                                                typed row scattered into one
                                                contiguous column per input
      -> release                                polars gets its buffers back

Every declared column — Float32/64, every Int/UInt width, Boolean, the
temporal types, String, Categorical/Enum — is decoded by nanoarrow from
polars' own buffers; no `_get_buffers()`, no `to_numpy()`, no
`cast(pl.Categorical)` of every String. A null never has its value read: the
gather writes the column's fill (MISSING_AS / NOT_APPLICABLE_AS ride in the
`ColDesc`) or the default, and reports validity per column, which is what
an OPTIONAL input's `__valid__` array is.

Per-row cost is §1.3b's deliberate choice: one C call per row (measured
149 ns/row for 17 columns) instead of today's per-column numpy paths — a
~4.6x regression on the gather itself, accepted because batches run daily
and the metric is "less code from us, more from existing tooling". The
boundary is therefore O(rows x columns); it is also the same path for every
mode (§1.6), interpreted and stepped included.

What the caller must know:

    * the caller's frame IS rechunked in place by polars' export (§9) —
      views a caller took into a multi-chunk frame before `apply()` point at
      the old chunks afterwards;
    * a STR column's `values` is an `(n, 2)` int64 table of `(address,
      length)` into polars' memory: valid while `ExtractedFrame.kernel_frame`
      is alive (the export is zero-copy over the frame's own buffers —
      measured: the same address across two exports, readable after
      release), which is why that frame rides in the result;
    * `FrameView` structs are pooled per THREAD (`threading.local`): serving
      runs concurrent `apply()` calls on one pipeline, and a pooled row
      buffer shared across threads would answer one row with another's.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from decider2._arrow.frame import ROW_DTYPES, ColumnSpec, FramePlan, FrameView
from decider2.types import FeatureKind, Input, MissingInputPolicy, NullPolicy

from .dtypes import ColumnPlan, NeedsKernelSplit, cast_frame, kind_for, plan_column
from .nulls import FillInfo, NullRouting, fill_reason, route_required_nulls

__all__ = [
    "ExtractedColumn",
    "ExtractedFrame",
    "NeedsKernelSplit",
    "extract_frame",
]

STR = FeatureKind.STR


@dataclass(frozen=True)
class ExtractedColumn:
    """One column, ready for the calling convention (doc 05 §3): `values`
    already in the dtype the kernel is typed against (`ROW_DTYPES[kind]`,
    the same table `compile.driver.numpy_dtype` reads — float64 / int64 /
    bool / int32, or `(n, 2)` int64 spans for a `bytes` input), read-only,
    C-contiguous.

    `validity` is populated only for `NullPolicy.OPTIONAL` columns — the
    mask the compiled driver builds a numba `Optional` per row from (doc 05
    §2). For every other null policy the values array is already
    unconditionally safe to read: `REQUIRED` nulls were routed away before
    the export, and `MISSING_AS`/`NOT_APPLICABLE_AS` nulls were filled by
    the gather.

    `categories` is populated only for CODE columns that arrived with a
    dictionary (a Categorical/Enum, or a String cast to one): the exported
    dictionary in index order, which a code counts into. This is what lets
    a `str`-typed param be resolved to the matching code at param-
    resolution time (`runtime.invoke.resolve_params`) instead of the kernel
    ever seeing text.
    """

    name: str
    values: np.ndarray
    validity: np.ndarray | None
    plan: ColumnPlan
    fill: FillInfo | None = None
    categories: tuple[str | None, ...] | None = None


@dataclass(frozen=True)
class ExtractedFrame:
    """The whole-frame result of `extract_frame`: every input column, ready
    for the calling convention, plus the routing decision for rows a
    `REQUIRED` null pulled out before the kernel ever saw them.

    `categories` is the frame-level counterpart of `ExtractedColumn.
    categories` — every dictionary-bearing CODE column's name mapped to its
    category tuple, gathered here so a caller (`runtime.invoke.resolve_
    params`) can resolve a `str`-typed param's literal without walking
    `columns` itself. `kernel_frame` is the row subset the kernel sees,
    after routing and after the frame-tier casts; it keeps the exported
    buffers (a STR column's spans) alive.
    """

    columns: dict[str, ExtractedColumn]
    routing: NullRouting
    kernel_frame: pl.DataFrame
    categories: dict[str, tuple[str | None, ...]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The schema-invariant half: decided once per (declared inputs, frame
# schema), cached forever — the frame's column order and dtypes are the
# whole input to every decision here, and a pipeline sees a handful of
# schemas in its life.
# ---------------------------------------------------------------------------


class _SchemaPlan:
    __slots__ = ("present", "absent", "plans", "casts", "frame_plan")

    def __init__(self, inputs: Sequence[Input], columns: Sequence[str], dtypes: Sequence[pl.DataType]) -> None:
        position = {name: k for k, name in enumerate(columns)}
        self.present: tuple[Input, ...] = tuple(d for d in inputs if d.name in position)
        self.absent: tuple[Input, ...] = tuple(d for d in inputs if d.name not in position)
        plans: dict[str, ColumnPlan] = {}
        casts: list[tuple[int, ColumnPlan]] = []
        specs: list[ColumnSpec] = []
        for decl in self.present:
            kind = kind_for(decl.annotation)
            plan = plan_column(decl.name, dtypes[position[decl.name]], kind)   # NeedsKernelSplit for List/Struct/...
            plans[decl.name] = plan
            if plan.cast is not None:
                casts.append((position[decl.name], plan))
            specs.append(ColumnSpec(decl.name, kind, _fill_for(decl, kind)))
        self.plans = plans
        self.casts = tuple(casts)
        self.frame_plan = FramePlan(specs, columns) if specs else None


def _fill_for(decl: Input, kind: FeatureKind):
    """The value the gather writes under a null of a MISSING_AS /
    NOT_APPLICABLE_AS column (`ColumnSpec.fill`); None for the policies that
    do not fill, which get the kind's default (NaN / 0 / False / -1)."""
    if fill_reason(decl.null_policy) is None or decl.fill is None:
        return None
    if kind is FeatureKind.CODE and isinstance(decl.fill, str):
        raise ValueError(
            f"input '{decl.name}' is a `str` column with a string fill {decl.fill!r}; a `str` "
            "input enters the kernel as a dictionary code (doc 05 §1.5), so its fill must be an "
            "int code, not text."
        )
    return decl.fill


_SCHEMA_PLANS: dict[tuple, _SchemaPlan] = {}
_local = threading.local()


def _hashable(value):
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _schema_plan(inputs: Sequence[Input], columns: Sequence[str], dtypes: Sequence[pl.DataType]) -> _SchemaPlan:
    key = (
        tuple((d.name, _hashable(d.annotation), d.null_policy, _hashable(d.fill)) for d in inputs),
        tuple(columns), tuple(dtypes),
    )
    plan = _SCHEMA_PLANS.get(key)
    if plan is None:
        plan = _SCHEMA_PLANS[key] = _SchemaPlan(inputs, columns, dtypes)
    return plan


def _checkout(plan: FramePlan) -> FrameView:
    """This thread's pooled `FrameView` for `plan`, or a new one. Popped
    from the pool while in use, so a re-entrant call on the same thread
    gets its own; `_checkin` puts it back."""
    pool = getattr(_local, "views", None)
    if pool is None:
        pool = _local.views = {}
    view = pool.pop(plan, None)
    return view if view is not None else FrameView(plan)


def _checkin(plan: FramePlan, view: FrameView) -> None:
    view.release()
    _local.views[plan] = view


# ---------------------------------------------------------------------------
# The per-call half.
# ---------------------------------------------------------------------------


def _synthesize_absent_column(decl: Input, n: int) -> ExtractedColumn:
    """Review finding 4: a declared input entirely absent from the frame —
    not merely null on some rows — shares the same fill/route path a null
    does; there is no polars column to export, so this builds the column
    the gather would have produced had it existed and been null on every
    row, in the kind's own dtype.

    A REQUIRED `decl` reaching here has already had every one of `n` rows
    routed away by `route_required_nulls` (or the whole batch already
    raised, for a `raise_for` column) — `n` is 0 in that case, and this only
    has to produce a correctly-shaped EMPTY array so the compiled kernel's
    calling convention still has something to bind its `array` argument
    role against (without even an empty array in the registry, `fused` mode
    raises a bare `KeyError` building the kernel call, three frames deep in
    `modes._build_call_args` — the exact failure this finding names).
    MISSING_AS/NOT_APPLICABLE_AS/OPTIONAL never route rows away, so `n` here
    is the full `kernel_frame` height, filled/masked exactly as a genuinely
    all-null column of that tier would be.
    """
    kind = kind_for(decl.annotation)
    dtype = ROW_DTYPES[kind]
    plan = ColumnPlan(decl.name, pl.Null(), kind, None,
                      note="column absent from the input frame — synthesized placeholder (doc 03 §1)")
    validity = None
    fill = None
    reason = fill_reason(decl.null_policy)
    if reason is not None:
        values = np.full((n, 2) if kind is STR else n, _fill_for(decl, kind) if decl.fill is not None else 0, dtype)
        fill = FillInfo(reason=reason, filled_count=n, filled_mask=np.ones(n, dtype=bool))
    else:
        values = np.zeros((n, 2) if kind is STR else n, dtype)
        if kind is STR:
            values[:, 1] = -1
        if decl.null_policy is NullPolicy.OPTIONAL:
            validity = np.zeros(n, dtype=bool)
    values.flags.writeable = False
    return ExtractedColumn(decl.name, values, validity, plan, fill=fill)


def extract_frame(
    frame: pl.DataFrame,
    inputs: Sequence[Input],
    *,
    policy: MissingInputPolicy | None = None,
) -> ExtractedFrame:
    """The whole-frame extraction pass (doc 05 §1 + §2 tied together):

    1. route `REQUIRED` nulls over the *whole* frame — a `raise_for` column
       fails the whole batch here, before any export happens (doc 03 §1);
    2. filter to the rows that pass; insert the frame-tier casts the
       declared kinds need (`dtypes.plan_column`, decided once per schema);
    3. export the frame once, resolve every declared column, gather every
       row (§1.3b), and hand each input back as one typed column. A
       declared input with no flat form (`NeedsKernelSplit`, an
       `ArrowKindError`) fails here by name, before the export — the
       kernel-split escape is one layer up, not this function's job.

    The rows `routing.mask` selects are **not** dropped — they are excluded
    from `kernel_frame` and left for the caller to route to
    `routing.decision` (doc 03 §1: "the violation lands in the decision
    record with the column, the reason code and the count").
    """
    routing = route_required_nulls(frame, inputs, policy)
    kernel_frame = frame.filter(pl.Series(~routing.mask)) if routing.routed_count else frame
    n = kernel_frame.height

    sp = _schema_plan(tuple(inputs), tuple(kernel_frame.columns), tuple(kernel_frame.dtypes))
    if sp.casts:
        kernel_frame = cast_frame(kernel_frame, sp.casts)

    columns: dict[str, ExtractedColumn] = {}
    categories: dict[str, tuple[str | None, ...]] = {}
    if sp.frame_plan is not None:
        fp = sp.frame_plan
        view = _checkout(fp)
        try:
            # `ArrowKindError` from here means the table had no cast for
            # this (dtype, kind) pair and nanoarrow named the Arrow type;
            # a nested/object column was already refused by `plan_column`.
            view.bind(kernel_frame)
            cols = view.materialize_columns()
            by_kind = {
                FeatureKind.F64: cols.f64, FeatureKind.I64: cols.i64, FeatureKind.BOOL: cols.b8,
                FeatureKind.CODE: cols.i32, STR: cols.span,
            }
            for c, decl in enumerate(sp.present):
                kind = FeatureKind(int(fp.kinds[c]))
                values = by_kind[kind][int(fp.slots[c])]
                values.flags.writeable = False
                validity = None
                fill = None
                reason = fill_reason(decl.null_policy)
                if decl.null_policy is NullPolicy.OPTIONAL:
                    validity = cols.valid[c]
                elif reason is not None:
                    valid = cols.valid[c]
                    missing = n - int(np.count_nonzero(valid))
                    fill = FillInfo(reason=reason, filled_count=missing,
                                    filled_mask=(~valid) if missing else None)
                cats = None
                if kind is FeatureKind.CODE:
                    cats = view.dictionary(decl.name)
                    if cats is not None:
                        categories[decl.name] = cats
                columns[decl.name] = ExtractedColumn(
                    decl.name, values, validity, sp.plans[decl.name], fill=fill, categories=cats,
                )
        finally:
            _checkin(fp, view)

    for decl in sp.absent:
        columns[decl.name] = _synthesize_absent_column(decl, n)

    return ExtractedFrame(columns=columns, routing=routing, kernel_frame=kernel_frame, categories=categories)
