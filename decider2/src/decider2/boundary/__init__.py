"""The boundary: polars in, a compiled kernel's arrays out, polars back.

Doc 05 (Boundary and compilation) is the spec; docs/BOUNDARY-REWORK.md §1 is
the shape since Stage 3: the whole frame crosses ONCE through
`__arrow_c_stream__()`, vendored nanoarrow decodes it (`decider2._arrow`),
and `sm_gather_row` fills a typed row per row. Four modules, in dependency
order:

    dtypes.py    the Arrow TABLE (§1.2) — declared kind x polars dtype:
                 native, one frame-tier cast, or NeedsKernelSplit
    nulls.py     doc 03 §1's four situations — REQUIRED routes at the frame
                 level; the fill/mask themselves happen in the gather
    extract.py   the whole-frame import: route, cast, bind, gather, release
    writeback.py doc 05 §3.1 — dtype-grouped 2D arrays, never records

`marshal.py` (whole-row bulk request marshalling, doc 05 §3.1b) and
`chunk.py` (doc 05 §3.2) are a different slice of Layer 1 and are not part of
this package's surface.

This package's re-exports are narrowed to what `decider2.runtime.invoke` —
the only caller outside `boundary/` itself and its own tests (over-engineering
audit) — actually imports via `from decider2.boundary import ...`. Everything
else here (the Arrow table, null routing, `explain_boundary`, etc.) is still
public — import it from its own submodule
(`decider2.boundary.dtypes`/`.nulls`/`.extract`/`.writeback`), exactly as
every test in this package already does.
"""
from __future__ import annotations

from .extract import extract_frame
from .writeback import DtypeGroup, KernelOutputs, Layout, resolve_kept_input_columns, write_back

__all__ = [
    "extract_frame",
    "Layout", "DtypeGroup", "KernelOutputs", "write_back", "resolve_kept_input_columns",
]
