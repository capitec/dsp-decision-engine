"""The boundary: polars in, a compiled kernel's arrays out, polars back.

Doc 05 (Boundary and compilation) is the spec; doc 00-BUILD.md Layer 1 is the
build order. Four modules, in dependency order:

    dtypes.py    the dtype LADDER (§1.5) — classifies, never rejects
    nulls.py     doc 03 §1's four null situations — fill, route, or pass through
    extract.py   doc 05 §1 + §2 tied together — polars-native, never `to_arrow`
    writeback.py doc 05 §3.1 — dtype-grouped 2D arrays, never records

`marshal.py` (whole-row bulk request marshalling, doc 05 §3.1b) and
`chunk.py` (doc 05 §3.2) are a different slice of Layer 1 and are not part of
this package's surface.
"""
from __future__ import annotations

from .dtypes import ColumnPlan, DtypeTier, EntryMode, explain_boundary, plan_column, probe_column
from .extract import (
    ExtractedColumn,
    ExtractedFrame,
    NeedsKernelSplit,
    extract_column,
    extract_frame,
    is_clean,
    rechunk_once,
)
from .nulls import FillInfo, FillReason, NullRouting, fill_column, route_required_nulls, validity_mask
from .writeback import DtypeGroup, KernelOutputs, Layout, resolve_kept_input_columns, row_to_dict, to_series, write_back

__all__ = [
    # dtypes
    "DtypeTier", "EntryMode", "ColumnPlan", "plan_column", "probe_column", "explain_boundary",
    # nulls
    "FillReason", "FillInfo", "NullRouting", "fill_column", "route_required_nulls", "validity_mask",
    # extract
    "ExtractedColumn", "ExtractedFrame", "NeedsKernelSplit",
    "rechunk_once", "is_clean", "extract_column", "extract_frame",
    # writeback
    "Layout", "DtypeGroup", "KernelOutputs", "to_series", "write_back", "row_to_dict",
    "resolve_kept_input_columns",
]
