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


def _utf8(value: object) -> bytes | None:
    if isinstance(value, Span):
        return value.bytes
    return value.encode() if isinstance(value, str) else None


class Span:
    """A `Raw[bytes]` value outside a kernel: a string by its operations, `(address, byte length)` by index.

    Answers what a compiled span answers, so a step annotated `Raw[bytes]` gives
    the same result in every mode. A null is not a string: it equals nothing,
    including another null, and holds no prefix, suffix or part.

        Span("private") == "private"        # True
        Span(None) == Span(None)            # False
        Span(None)[1]                       # -1
    """

    __slots__ = ("bytes", "text")

    def __init__(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            # `Raw[bytes]` is a string column read as its UTF-8 bytes, the same refusal a kernel gives.
            raise TypeError(f"{type(value).__name__} is not a string")
        self.text = value
        self.bytes = None if value is None else value.encode()

    def __repr__(self) -> str:
        return f"Span({self.text!r})"

    def __eq__(self, other: object) -> bool:
        other = _utf8(other)
        return self.bytes is not None and other is not None and self.bytes == other

    def __len__(self) -> int:
        # Code points, as CPython counts them. A null has no length a kernel can
        # report through `len`, which may not be negative, so it reads as empty.
        return 0 if self.text is None else len(self.text)

    def __getitem__(self, k: int) -> int:
        # The kernel's `(address, byte length)`; only the length means anything here.
        return (0, -1 if self.bytes is None else len(self.bytes))[k]

    def startswith(self, prefix: object) -> bool:
        prefix = _utf8(prefix)
        return self.bytes is not None and prefix is not None and self.bytes.startswith(prefix)

    def endswith(self, suffix: object) -> bool:
        suffix = _utf8(suffix)
        return self.bytes is not None and suffix is not None and self.bytes.endswith(suffix)

    def __contains__(self, part: object) -> bool:
        part = _utf8(part)
        return self.bytes is not None and part is not None and part in self.bytes


def span_objects(values: np.ndarray, fill: str | None = None, missing: np.ndarray | None = None) -> np.ndarray:
    """`values` as `Span` objects, `fill` on the rows `missing` marks.

    Filled element by element: a `Span` has `__len__` and `__getitem__`, so numpy
    would read a bare one as a sequence.
    """
    out = np.empty(len(values), object)
    for i, value in enumerate(values):
        out[i] = Span(value)
    if missing is not None:
        filled = Span(fill)
        for i in np.flatnonzero(missing):
            out[i] = filled
    return out


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
