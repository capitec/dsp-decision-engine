"""The cached numba loops that drive the shim's per-row gather over a whole frame.

Both shim addresses are arguments, so each specialisation is a genuine
disk-cache hit in a fresh process.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from numba import njit

from decider.engine.boundary._arrow.intrinsics import call_gather, call_get_string


@dataclass
class Rows:
    """`FrameView.materialize()`: `(n, slots)` per kind, `span` `(n, 2 * n_str)`, `valid` `(n, ncols)`."""

    f64: np.ndarray
    i64: np.ndarray
    b8: np.ndarray
    i32: np.ndarray
    span: np.ndarray
    valid: np.ndarray

    def strings(self, slot: int) -> list[bytes | None]:
        """Every row's bytes for STR slot `slot` (copies), `None` for a null."""
        col = self.span[:, 2 * slot:2 * slot + 2]
        return [None if ln < 0 else ctypes.string_at(int(a), int(ln)) for a, ln in col.tolist()]


@dataclass
class Columns:
    """`FrameView.materialize_columns()`: `(slots, n)` per kind, `span` `(slots, n, 2)`, `valid` `(ncols, n)`."""

    f64: np.ndarray
    i64: np.ndarray
    b8: np.ndarray
    i32: np.ndarray
    span: np.ndarray
    valid: np.ndarray


@njit(cache=True)
def materialize_rows(gather_addr, plan_addr, n, f64, i64, b8, i32, span, valid,
                     out_f64, out_i64, out_b8, out_i32, out_span, out_valid):
    for i in range(n):
        call_gather(gather_addr, plan_addr, i)
        out_f64[i, :] = f64
        out_i64[i, :] = i64
        out_b8[i, :] = b8
        out_i32[i, :] = i32
        out_span[i, :] = span
        out_valid[i, :] = valid


@njit(cache=True)
def materialize_columns(gather_addr, plan_addr, n, f64, i64, b8, i32, span, valid,
                        out_f64, out_i64, out_b8, out_i32, out_span, out_valid):
    # The out_* first axes are the true slot counts; the row buffers are padded to >= 1.
    nf, ni, nb, nc, ns, ncols = (
        out_f64.shape[0], out_i64.shape[0], out_b8.shape[0], out_i32.shape[0],
        out_span.shape[0], out_valid.shape[0],
    )
    for i in range(n):
        call_gather(gather_addr, plan_addr, i)
        for j in range(nf):
            out_f64[j, i] = f64[j]
        for j in range(ni):
            out_i64[j, i] = i64[j]
        for j in range(nb):
            out_b8[j, i] = b8[j]
        for j in range(nc):
            out_i32[j, i] = i32[j]
        for j in range(ns):
            out_span[j, i, 0] = span[2 * j]
            out_span[j, i, 1] = span[2 * j + 1]
        for c in range(ncols):
            out_valid[c, i] = valid[c] != 0


@njit(cache=True)
def string_lengths(get_string_addr, view_addr, n, out):
    """Every row's byte length through `sm_get_string`, -1 for a null."""
    for i in range(n):
        _, ln = call_get_string(get_string_addr, view_addr, i)
        out[i] = ln
