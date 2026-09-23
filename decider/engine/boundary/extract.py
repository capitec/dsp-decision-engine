from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
import polars as pl

from decider.engine.boundary._arrow.plan import ColumnSpec, FramePlan
from decider.engine.boundary._arrow.view import FrameView
from decider.engine.boundary.dtypes import ColumnPlan, cast_frame, plan_column
from decider.engine.boundary.nulls import check_required
from decider.engine.ir.decls import FeatureKind, Input, NullPolicy, feature_kind


@dataclass(frozen=True)
class ExtractedColumn:
    """One input column, read-only and C-contiguous, in the dtype the kernel is typed against.

    Args:
        values: float64 / int64 / bool / int32 codes, or an `(n, 2)` int64 table
            of `(address, byte length)` spans for a `bytes` input (length -1 for a null).
        validity: the per-row mask, set only for an OPTIONAL input.
        categories: the exported dictionary a `str` input's codes index, when it has one.
    """

    name: str
    values: np.ndarray
    validity: np.ndarray | None
    plan: ColumnPlan
    categories: tuple[str | None, ...] | None = None


@dataclass(frozen=True)
class ExtractedFrame:
    """Every declared input of `extract_frame`, plus the frame the kernel reads.

    `kernel_frame` is the frame after casts; it keeps the memory a `bytes`
    input's spans point into alive. `categories` maps each `str` input with a
    dictionary to its categories.
    """

    columns: dict[str, ExtractedColumn]
    kernel_frame: pl.DataFrame
    categories: dict[str, tuple[str | None, ...]] = field(default_factory=dict)


class _SchemaPlan:
    __slots__ = ("plans", "casts", "frame_plan")

    def __init__(self, inputs: Sequence[Input], columns: Sequence[str], dtypes: Sequence[pl.DataType]):
        position = {name: k for k, name in enumerate(columns)}
        self.plans: dict[str, ColumnPlan] = {}
        casts, specs = [], []
        for decl in inputs:
            kind = feature_kind(decl.annotation)
            plan = plan_column(decl.name, dtypes[position[decl.name]], kind)
            self.plans[decl.name] = plan
            if plan.cast is not None:
                casts.append((position[decl.name], plan))
            specs.append(ColumnSpec(decl.name, kind, _fill_for(decl, kind)))
        self.casts = tuple(casts)
        self.frame_plan = FramePlan(specs, columns) if specs else None


def _fill_for(decl: Input, kind: FeatureKind):
    if decl.null_policy is not NullPolicy.MISSING_AS:
        return None
    if kind is FeatureKind.CODE and isinstance(decl.fill, str):
        raise ValueError(
            f"input '{decl.name}' is a `str` column with a string fill {decl.fill!r}; a `str` "
            "input enters the kernel as a dictionary code, so its fill must be an int code."
        )
    return decl.fill


# Keyed by the whole schema: a pipeline sees a handful of schemas in its life.
# ponytail: unbounded, bound it if schemas ever vary per request.
_schema_plan = lru_cache(maxsize=None)(_SchemaPlan)

# FrameViews hold row buffers, so they're pooled per thread: serving runs
# concurrent calls on one pipeline.
_local = threading.local()


def _checkout(plan: FramePlan) -> FrameView:
    pool = _local.__dict__.setdefault("views", {})
    # Popped while in use, so a re-entrant call on the same thread gets its own view.
    return pool.pop(plan, None) or FrameView(plan)


def _checkin(plan: FramePlan, view: FrameView) -> None:
    view.release()
    _local.views[plan] = view


def extract_frame(frame: pl.DataFrame, inputs: Sequence[Input], *, path: str = "") -> ExtractedFrame:
    """Read every declared input out of a polars frame in one Arrow export.

    A null (or absent column) in a REQUIRED input raises `MissingInputError`
    naming the input, `path` and the null row count. MISSING_AS nulls are
    filled; OPTIONAL inputs get a validity mask. A column its kind can't read
    raises `ArrowKindError` (`NeedsKernelSplit` for nested columns) by name.
    Exporting rechunks `frame` in place when no cast was needed.

    Args:
        path: the step path, for error messages.

    Example::

        ex = extract_frame(df, [Input("income", float), Input("bonus", float | None, NullPolicy.OPTIONAL)])
        ex.columns["income"].values   # float64, read-only
    """
    check_required(frame, inputs, path)
    absent = [d.name for d in inputs if d.name not in frame.columns]
    if absent:
        # An absent column is an all-null one, so it takes the same fill/mask path.
        frame = frame.with_columns(pl.lit(None).alias(name) for name in absent)
    sp = _schema_plan(tuple(inputs), tuple(frame.columns), tuple(frame.dtypes))
    kernel_frame = cast_frame(frame, sp.casts) if sp.casts else frame
    columns: dict[str, ExtractedColumn] = {}
    categories: dict[str, tuple[str | None, ...]] = {}
    fp = sp.frame_plan
    if fp is None:
        return ExtractedFrame(columns, kernel_frame, categories)
    view = _checkout(fp)
    try:
        view.bind(kernel_frame)
        cols = view.materialize_columns()
        by_kind = (cols.f64, cols.i64, cols.b8, cols.i32, cols.span)
        for c, decl in enumerate(inputs):
            kind = FeatureKind(int(fp.kinds[c]))
            values = by_kind[kind][int(fp.slots[c])]
            values.flags.writeable = False
            validity = cols.valid[c] if decl.null_policy is NullPolicy.OPTIONAL else None
            cats = view.dictionary(decl.name) if kind is FeatureKind.CODE else None
            if cats is not None:
                categories[decl.name] = cats
            columns[decl.name] = ExtractedColumn(decl.name, values, validity, sp.plans[decl.name], cats)
    finally:
        _checkin(fp, view)
    return ExtractedFrame(columns, kernel_frame, categories)
