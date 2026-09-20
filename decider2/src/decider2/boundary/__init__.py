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

This package's re-exports are narrowed to what `decider2.runtime.invoke` —
the only caller outside `boundary/` itself and its own tests (over-engineering
audit) — actually imports via `from decider2.boundary import ...`. Everything
else here (the dtype ladder, null routing, per-column extraction, etc.) is
still public — import it from its own submodule
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
