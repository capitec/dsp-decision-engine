"""Compiled-value representations shared by every runner, so `Text`/`Columnar[Item]`
give the same value regardless of `mode=`.
"""
from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import polars as pl

from decikit._engine.ir.decls import Input, NullPolicy
from decikit.ops.text import Text

# A `bytes` object's own buffer, without the microsecond `np.ndarray.ctypes.data` costs.
_address = ctypes.pythonapi.PyBytes_AsString
_address.restype = ctypes.c_void_p
_address.argtypes = (ctypes.py_object,)


def despan(x: Any) -> Any:
    """Convert `Text` (or a list containing them) to `str | None` for user-facing output."""
    if isinstance(x, Text):
        return x.text
    if isinstance(x, list):
        return [v.text if isinstance(v, Text) else v for v in x]
    return x


def span_objects(values: np.ndarray, fill: str | None = None, missing: np.ndarray | None = None) -> np.ndarray:
    """`values` as `Text` objects, `fill` on the rows `missing` marks.

    Filled element by element: a `Text` has `__len__` and `__getitem__`, so numpy
    would read a bare one as a sequence.
    """
    out = np.empty(len(values), object)
    for i, value in enumerate(values):
        out[i] = Text(value)
    if missing is not None:
        filled = Text(fill)
        for i in np.flatnonzero(missing):
            out[i] = filled
    return out


def spans(x: np.ndarray, mask: np.ndarray | None, alive: list, source: pl.Series | None) -> np.ndarray:
    """Strings as `(address, byte length)` spans into Arrow memory, -1 for a null; zero-copy where it can."""
    if mask is not None:
        x = np.where(mask, x, None)
    if len(x) <= 32:
        # A few records (a record's run): borrowing each value's own bytes beats building a frame to export.
        return _borrowed(x, alive)
    from decikit._engine.boundary.extract import extract_frame

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
