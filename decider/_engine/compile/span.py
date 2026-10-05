# A `Raw[bytes]` value in a kernel is a `(address, byte length)` pair into Arrow
# memory. `SpanType` gives that pair its own numba type, so `==`, `len` and the
# rest below mean string things on a span and nothing else: overloading them on
# the bare `UniTuple(int64, 2)` would capture every unrelated pair of ints in a
# step, answering False for `(1, 2) == (1, 2)`. The type deliberately isn't a
# tuple either, because numba rewrites `len(t)` on a tuple *argument* to the
# constant 2 before type inference runs, where no overload can reach it.
from __future__ import annotations

import ctypes

import numpy as np
from numba.core import types
from numba.core.datamodel import models
from numba.extending import register_model, typeof_impl

# A `bytes` object's own buffer, without the microsecond `np.ndarray.ctypes.data` costs.
_address = ctypes.pythonapi.PyBytes_AsString
_address.restype = ctypes.c_void_p
_address.argtypes = (ctypes.py_object,)


class SpanType(types.Type):
    """A `Raw[bytes]` value in compiled code: UTF-8 bytes at an address, byte length -1 for a null."""

    # What UniTupleModel reads, so a span is the same `[2 x i64]` value the kernel builds.
    dtype = types.int64
    count = 2

    def __init__(self) -> None:
        super().__init__("span")

    def __len__(self) -> int:
        return 2


register_model(SpanType)(models.UniTupleModel)

SPAN = SpanType()

# One span as numpy stores it: the same 16 bytes `[2 x i64]` the kernel holds.
SPAN_DTYPE = np.dtype([("address", np.int64), ("length", np.int64)])


class SpanArray(np.ndarray):
    """A column of spans: one element is a span in a kernel and a `Span` in Python.

    The bytes live wherever they were read from, so `kept` holds whatever owns
    them (an Arrow import, a list of `bytes`) for as long as any view of this
    array survives.
    """

    def __getitem__(self, k):
        value = super().__getitem__(k)
        if not isinstance(k, (int, np.integer)):
            return value
        # The Python-side counterpart of the kernel's span, so a step that falls back to
        # Python answers `== "x"`, `len` and the rest identically. Imported here because
        # the representations live under `engine/run`, which reads this module.
        from decider._engine.run.representations import Span

        length = int(value["length"])
        return Span(None if length < 0 else ctypes.string_at(int(value["address"]), length).decode())

    def __iter__(self):
        return map(self.__getitem__, range(len(self)))


@typeof_impl.register(SpanArray)
def _typeof_span_array(val, c):
    return types.Array(SPAN, 1, "C", readonly=not val.flags.writeable)


def span_array(pairs: np.ndarray, kept: object = None) -> SpanArray:
    """An int64 array of `(address, byte length)` pairs as an array of spans.

    `kept` is whatever owns the bytes the addresses point into.

    Example::

        span_array(np.array([address, 5], np.int64), kept=[b"snoop"])
    """
    out = pairs.reshape(-1).view(SPAN_DTYPE).view(SpanArray)
    out.kept = kept
    return out


def spans_of(values, optional: bool, null) -> SpanArray:
    """`values`' UTF-8 bytes as an array of spans, holding the `bytes` the addresses point into.

    `null(k)` is raised for a `None` at index `k` unless `optional`.
    """
    kept: list[bytes] = []
    flat: list[int] = []
    for k, value in enumerate(values):
        if value is None:
            if not optional:
                raise null(k)
            flat += (0, -1)
            continue
        encoded = str.encode(value)   # unbound: a non-str value raises TypeError, as a kernel refuses one
        kept.append(encoded)
        flat += (_address(encoded), len(encoded))
    return span_array(np.array(flat, np.int64), kept)
