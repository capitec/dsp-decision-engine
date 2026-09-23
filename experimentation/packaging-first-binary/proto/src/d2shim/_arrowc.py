"""Consume a polars Series' Arrow C Data Interface export WITHOUT pyarrow.

`pl.Series.__arrow_c_stream__()` returns a PyCapsule wrapping an
`ArrowArrayStream` (https://arrow.apache.org/docs/format/CDataInterface.html).
We read it with ctypes: pull every chunk's `ArrowArray`, and record the raw
buffer addresses. For a polars 1.41 String column the exported layout is
Arrow's **Utf8View** ("binview"): format string `"vu"`, buffers

    [0] validity bitmap (may be NULL when no nulls)
    [1] views: 16 bytes per row
            u32 length
            if length <= 12: the bytes themselves, inline, zero-padded
            else:            u32 prefix(4 bytes), u32 buffer_index, u32 offset
    [2 .. n_buffers-2] variadic data buffers (long strings live here)
    [n_buffers-1] int64 sizes of those data buffers

Nothing here copies. The `ArrowArray` structs stay alive on the `StringView`
object; `release()` (or garbage collection) hands the buffers back.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

import numpy as np

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


class ArrowSchema(ctypes.Structure):
    pass


ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowSchema))),
    ("private_data", ctypes.c_void_p),
]


class ArrowArray(ctypes.Structure):
    pass


ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArray))),
    ("private_data", ctypes.c_void_p),
]


class ArrowArrayStream(ctypes.Structure):
    pass


ArrowArrayStream._fields_ = [
    ("get_schema", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ArrowArrayStream), ctypes.POINTER(ArrowSchema))),
    ("get_next", ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ArrowArrayStream), ctypes.POINTER(ArrowArray))),
    ("get_last_error", ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.POINTER(ArrowArrayStream))),
    ("release", ctypes.CFUNCTYPE(None, ctypes.POINTER(ArrowArrayStream))),
    ("private_data", ctypes.c_void_p),
]


@dataclass
class Chunk:
    length: int
    offset: int
    null_count: int
    validity: int          # address or 0
    views: int             # address of the 16-byte views buffer
    data: list[int]        # addresses of the variadic data buffers
    data_sizes: list[int]  # their byte lengths
    _arr: ArrowArray = field(repr=False, default=None)


@dataclass
class StringView:
    """Everything a kernel needs, per chunk, plus the numba-friendly tables."""

    format: str
    chunks: list[Chunk]
    _stream: ArrowArrayStream = field(repr=False, default=None)

    @property
    def n_rows(self) -> int:
        return sum(c.length for c in self.chunks)

    def release(self) -> None:
        for c in self.chunks:
            if c._arr is not None and c._arr.release:
                c._arr.release(ctypes.byref(c._arr))
                c._arr.release = ctypes.cast(None, ArrowArray._fields_[8][1])
        if self._stream is not None and self._stream.release:
            self._stream.release(ctypes.byref(self._stream))
            self._stream.release = ctypes.cast(None, ArrowArrayStream._fields_[3][1])

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.release()
        except Exception:
            pass


def export(series, requested_schema=None) -> StringView:
    """Walk the C stream and collect raw buffer addresses. O(chunks), not O(rows)."""
    capsule = series.__arrow_c_stream__(requested_schema)
    ptr = _PyCapsule_GetPointer(capsule, b"arrow_array_stream")
    stream = ctypes.cast(ptr, ctypes.POINTER(ArrowArrayStream)).contents
    schema = ArrowSchema()
    rc = stream.get_schema(ctypes.byref(stream), ctypes.byref(schema))
    if rc != 0:
        raise RuntimeError(stream.get_last_error(ctypes.byref(stream)))
    fmt = schema.format.decode()
    if schema.release:
        schema.release(ctypes.byref(schema))
    chunks = []
    while True:
        arr = ArrowArray()
        rc = stream.get_next(ctypes.byref(stream), ctypes.byref(arr))
        if rc != 0:
            raise RuntimeError(stream.get_last_error(ctypes.byref(stream)))
        if not arr.release:  # end of stream
            break
        nb = arr.n_buffers
        bufs = [arr.buffers[i] or 0 for i in range(nb)]
        if fmt == "vu":
            n_data = nb - 3
            sizes_ptr = bufs[-1]
            sizes = list(np.ctypeslib.as_array(
                ctypes.cast(sizes_ptr, ctypes.POINTER(ctypes.c_int64)), shape=(n_data,)
            )) if n_data else []
            chunks.append(Chunk(
                length=arr.length, offset=arr.offset, null_count=arr.null_count,
                validity=bufs[0], views=bufs[1], data=bufs[2:2 + n_data],
                data_sizes=[int(x) for x in sizes], _arr=arr,
            ))
        else:  # large_utf8 "U" or utf8 "u": [validity, offsets, values]
            chunks.append(Chunk(
                length=arr.length, offset=arr.offset, null_count=arr.null_count,
                validity=bufs[0], views=bufs[1], data=[bufs[2]], data_sizes=[-1], _arr=arr,
            ))
    # Keep the capsule alive with the view: the stream struct lives inside it.
    view = StringView(format=fmt, chunks=chunks, _stream=stream)
    view._capsule = capsule
    return view


# ---------------------------------------------------------------------------
# Tables for a numba kernel. All addresses are plain uint64 ARGUMENTS
# (EXPERIMENTS.md §W: never captured globals -- that kills the disk cache).
# ---------------------------------------------------------------------------
MAX_DATA_BUFFERS = 64


def chunk_tables(chunk: Chunk):
    """(views: uint8[n*16] numpy view, validity: uint8 numpy view or empty,
    data_addr: uint64[MAX], data_size: uint64[MAX], n_data)"""
    n = chunk.length + chunk.offset
    views = np.ctypeslib.as_array(ctypes.cast(chunk.views, ctypes.POINTER(ctypes.c_uint8)), shape=(n * 16,)) \
        if n else np.empty(0, dtype=np.uint8)
    if chunk.validity:
        validity = np.ctypeslib.as_array(
            ctypes.cast(chunk.validity, ctypes.POINTER(ctypes.c_uint8)), shape=((n + 7) // 8,))
    else:
        validity = np.empty(0, dtype=np.uint8)
    data_addr = np.zeros(MAX_DATA_BUFFERS, dtype=np.uint64)
    data_size = np.zeros(MAX_DATA_BUFFERS, dtype=np.uint64)
    nd = len(chunk.data)
    if nd > MAX_DATA_BUFFERS:
        raise ValueError(f"{nd} variadic data buffers > {MAX_DATA_BUFFERS}")
    data_addr[:nd] = chunk.data
    data_size[:nd] = chunk.data_sizes
    return views, validity, data_addr, data_size, nd
