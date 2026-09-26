"""Compiled-value representations shared by every runner, so `Raw[str]`/`Raw[bytes]`/`Rows[Item]`
give the same value regardless of `mode=`.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from decider.engine.ir.decls import Input, NullPolicy
from decider.types import string_code


def codes(values: np.ndarray) -> np.ndarray:
    """Strings as int32 dictionary codes, the same ones `raw_str()` gives its constants."""
    return np.fromiter(map(string_code, values), np.int32, len(values))


def spans(x: np.ndarray, mask: np.ndarray | None, alive: list, source: pl.Series | None) -> np.ndarray:
    """Strings as `(address, byte length)` spans into Arrow memory, -1 for a null; zero-copy where it can."""
    from decider.engine.boundary.extract import extract_frame

    if mask is not None:
        x = np.where(mask, x, None)
    if len(x) <= 32:
        # A few records (score): encoding them beats building a frame to export.
        raw = [None if s is None else str.encode(s) for s in x]
        buffer = np.frombuffer(b"".join(b for b in raw if b) or bytes(1), np.uint8)
        alive.append(buffer)
        lengths = np.array([-1 if b is None else len(b) for b in raw], np.int64)
        starts = np.cumsum(np.maximum(lengths, 0)) - np.maximum(lengths, 0)
        return np.stack([buffer.ctypes.data + starts, lengths], axis=1)
    # The input frame's own column when it holds these values, else a copy (an override, a row subset).
    if source is None or source.dtype != pl.String:
        source = pl.Series(x.tolist(), dtype=pl.String)
    extracted = extract_frame(source.to_frame("s"), [Input("s", bytes, NullPolicy.OPTIONAL)])
    alive.append(extracted.kernel_frame)
    return extracted.columns["s"].values
