"""Compiled-value representations shared by every runner, so `Raw[str]`/`Raw[bytes]`/`Rows[Item]`
give the same value regardless of `mode=`.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
import polars as pl

from decider.engine.ir.decls import Input, NullPolicy
from decider.types import Representation, raw_string_codes


class StringCodes:
    """Codes shared with `raw_str()`'s symbolic constants, plus any value seen at runtime."""

    def __init__(self) -> None:
        self._codes: dict[str, int] = dict(raw_string_codes())
        self._lock = threading.Lock()

    def encode(self, values: np.ndarray) -> np.ndarray:
        with self._lock:
            self._codes.update(raw_string_codes())
            codes = {s: self._codes.setdefault(s, len(self._codes)) for s in values}
        return np.fromiter((codes[s] for s in values), np.int32, len(values))

    def bundle(self, name: str, value: str) -> np.int32:
        with self._lock:
            self._codes.update(raw_string_codes())
            return np.int32(self._codes.setdefault(value, len(self._codes)))


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


def build_raw(kind: Representation, values: np.ndarray, mask: np.ndarray | None, alive: list,
              source: pl.Series | None, codes: StringCodes) -> np.ndarray:
    """`RAW_STRING`/`RAW_BYTES`: the same representation in every runner. Other kinds pass through."""
    if kind is Representation.RAW_BYTES:
        return spans(values, mask, alive, source)
    if kind is Representation.RAW_STRING:
        return codes.encode(values)
    return values
