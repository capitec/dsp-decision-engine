"""Compiled-value representations shared by every runner, so `Raw[str]`/`Raw[bytes]`/`Rows[Item]`
give the same value regardless of `mode=`.
"""
from __future__ import annotations

import ctypes

import numpy as np
import polars as pl

from decider.engine.ir.decls import Input, NullPolicy
from decider.types import string_code

# A `bytes` object's own buffer, without the microsecond `np.ndarray.ctypes.data` costs.
_address = ctypes.pythonapi.PyBytes_AsString
_address.restype = ctypes.c_void_p
_address.argtypes = (ctypes.py_object,)


def codes(values: np.ndarray) -> np.ndarray:
    """Strings as int32 dictionary codes, the same ones `raw_str()` gives its constants."""
    return np.fromiter(map(string_code, values), np.int32, len(values))


def spans(x: np.ndarray, mask: np.ndarray | None, alive: list, source: pl.Series | None) -> np.ndarray:
    """Strings as `(address, byte length)` spans into Arrow memory, -1 for a null; zero-copy where it can."""
    if mask is not None:
        x = np.where(mask, x, None)
    if len(x) <= 32:
        # A few records (score): borrowing each value's own bytes beats building a frame to export.
        return _borrowed(x, alive)
    from decider.engine.boundary.extract import extract_frame

    # The input frame's own column when it holds these values, else a copy (an override, a row subset).
    if source is None or source.dtype != pl.String:
        source = pl.Series(x.tolist(), dtype=pl.String)
    extracted = extract_frame(source.to_frame("s"), [Input("s", bytes, NullPolicy.OPTIONAL)])
    alive.append(extracted.kernel_frame)
    return extracted.columns["s"].values


def _borrowed(x: np.ndarray, alive: list) -> np.ndarray:
    # The spans point into the `bytes` objects `kept` holds, so it goes in `alive`
    # with them: dropping it while a kernel still reads the spans is a dangling pointer.
    kept: list[bytes] = []
    flat: list[int] = []
    for s in x.tolist():
        if s is None:
            flat += (0, -1)
            continue
        b = str.encode(s)   # unbound: a non-str value is the TypeError the caller names
        kept.append(b)
        flat += (_address(b), len(b))
    alive.append(kept)
    return np.array(flat, np.int64).reshape(len(x), 2)
