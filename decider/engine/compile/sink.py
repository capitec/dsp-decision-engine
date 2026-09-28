"""`Sink`: a growable buffer a compiled step's `Columnar[Item]` output drains into.

One row emits a variable number of items, which no fixed-stride kernel output
column can hold. `Sink` amortized-doubles a `(rows, n_fields)` float64 buffer
(every field widens to float64; a bool or int field narrows back once its row
is sliced out, in Python) so every schema shares one buffer shape instead of
one generated per `Item`.
"""
from __future__ import annotations

import numpy as np
from numba import float64, int64, literal_unroll
from numba.experimental import jitclass

_SPEC = [("data", float64[:, :]), ("length", int64)]


@jitclass(_SPEC)
class Sink:
    """A growable `(rows, n_fields)` float64 buffer; `push` amortized-doubles it.

    Example::

        sink = Sink(capacity=4, n_fields=2)
        sink.push((1.5, 2))
        sink.data[: sink.length]
    """

    def __init__(self, capacity, n_fields):
        self.data = np.empty((max(capacity, 1), n_fields), np.float64)
        self.length = 0

    def _grow(self):
        capacity = max(4, self.data.shape[0] * 2)
        grown = np.empty((capacity, self.data.shape[1]), np.float64)
        grown[: self.length] = self.data[: self.length]
        self.data = grown

    def push(self, item):
        if self.length == self.data.shape[0]:
            self._grow()
        j = 0
        # `item`'s fields are a mix of float/int/bool: `literal_unroll` reads a fixed-size
        # heterogeneous tuple positionally, since numba has no runtime index into one.
        for x in literal_unroll(item):
            self.data[self.length, j] = x
            j += 1
        self.length += 1
