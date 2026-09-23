"""decider2's compiled Arrow boundary: vendored nanoarrow 0.9.0 behind a
C shim, and the `FrameView` import layer over it (docs/BOUNDARY-REWORK.md
§1, §4).

`diagnose()` and `available()` work on any box — including one where the
extension is missing or broken — so they are the two names imported
eagerly. `FramePlan`, `FrameView`, the intrinsics and the addresses load
the extension on first access, and raise the one `ImportError`
(`_shim.missing_message`) if they cannot.

Nothing in `decider2/` imports this package yet (Stage 1 has no consumer).
"""
from __future__ import annotations

from decider2._arrow.doctor import available, diagnose

_LAZY = {
    "FramePlan": "frame", "FrameView": "frame", "ColumnSpec": "frame", "Rows": "frame",
    "ArrowKindError": "frame", "ArrowImportError": "frame", "ROW_DTYPES": "frame",
    "materialize_rows": "frame", "string_lengths": "frame",
    "GATHER_ADDR": "_shim", "GET_STRING_ADDR": "_shim", "NANOARROW_VERSION": "_shim",
    "call_gather": "_shim", "call_get_string": "_shim",
    "load_u8": "_shim", "load_i64": "_shim", "load_f64": "_shim", "lib": "_shim",
}

__all__ = ["available", "diagnose", *sorted(_LAZY)]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
