"""Python side of the nanoarrow shim: load libnashim.so, import a polars
Series through the Arrow C stream in ONE ctypes call, and the numba
intrinsic that calls `sm_get_string` through a function pointer that is an
ARGUMENT of the kernel (EXPERIMENTS.md §W: never a captured global, or the
kernel will not disk-cache).

Nothing in this file knows how a string is laid out in memory.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

HERE = os.path.dirname(os.path.abspath(__file__))
A_DIR = os.path.normpath(os.path.join(HERE, "..", "arrow-strings-in-tree"))
if A_DIR not in sys.path:
    sys.path.append(A_DIR)  # after this directory, so our bench.py/results_io.py win
import arrowc  # noqa: E402  (the incumbent's ctypes structs; read-only import)

lib = ctypes.CDLL(os.path.join(HERE, "c", "libnashim.so"))
_vp, _i64, _int = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
for name in ("sm_sizeof_array_view", "sm_sizeof_schema", "sm_sizeof_array", "sm_sizeof_error"):
    getattr(lib, name).restype = ctypes.c_size_t
    getattr(lib, name).argtypes = []
lib.sm_nanoarrow_version.restype = ctypes.c_char_p
lib.sm_import_single.argtypes = [_vp, _vp, _vp, _vp, _vp, _vp]
lib.sm_import_single.restype = _int
lib.sm_view_set.argtypes = [_vp, _vp, _vp, _vp]
lib.sm_view_set.restype = _int
lib.sm_view_validate.argtypes = [_vp, _int, _vp]
lib.sm_view_validate.restype = _int
lib.sm_view_storage_type.argtypes = [_vp]
lib.sm_view_storage_type.restype = _int
lib.sm_view_length.argtypes = [_vp]
lib.sm_view_length.restype = _i64
lib.sm_view_offset.argtypes = [_vp]
lib.sm_view_offset.restype = _i64
for name in ("sm_view_reset", "sm_array_release", "sm_schema_release"):
    getattr(lib, name).argtypes = [_vp]
    getattr(lib, name).restype = None
lib.sm_error_message.argtypes = [_vp]
lib.sm_error_message.restype = ctypes.c_char_p
for name in ("sm_get_string", "sm_get_string_checked"):
    getattr(lib, name).argtypes = [_vp, _i64, ctypes.POINTER(_vp)]
    getattr(lib, name).restype = _i64

GET_STRING_ADDR = ctypes.cast(lib.sm_get_string, _vp).value
GET_STRING_CHECKED_ADDR = ctypes.cast(lib.sm_get_string_checked, _vp).value
NANOARROW_VERSION = lib.sm_nanoarrow_version().decode()

VIEW_SIZE = lib.sm_sizeof_array_view()
ERR_SIZE = lib.sm_sizeof_error()
assert lib.sm_sizeof_schema() == ctypes.sizeof(arrowc.ArrowSchema)
assert lib.sm_sizeof_array() == ctypes.sizeof(arrowc.ArrowArray)

# nanoarrow's enum ArrowType values we care about (nanoarrow.h)
# enum ArrowType (nanoarrow.h): NA=1, STRING=15, BINARY=16, LARGE_STRING=36, LARGE_BINARY=37,
# BINARY_VIEW=40, STRING_VIEW=41
STORAGE_NAMES = {1: "null (n)", 15: "utf8 (u)", 16: "binary", 36: "large_utf8 (U)", 37: "large_binary",
                 41: "utf8_view (vu)", 40: "binary_view"}
STRING_VIEW = 41
MINIMAL, DEFAULT, FULL = 1, 2, 3


class NanoError(RuntimeError):
    pass


class NanoChunk:
    __slots__ = ("array", "view", "view_addr", "arrowc_chunk")

    def __init__(self, array, view):
        self.array = array
        self.view = view
        self.view_addr = ctypes.addressof(view)
        self.arrowc_chunk = None


def _arrowc_chunk(arr: arrowc.ArrowArray, fmt: str) -> arrowc.Chunk:
    """The same raw addresses arrowc.export records, so approach C can feed
    A's kernel from a nanoarrow-imported array."""
    nb = arr.n_buffers
    bufs = [arr.buffers[i] or 0 for i in range(nb)]
    if fmt == "vu":
        n_data = nb - 3
        sizes = list(np.ctypeslib.as_array(
            ctypes.cast(bufs[-1], ctypes.POINTER(ctypes.c_int64)), shape=(n_data,))) if n_data else []
        return arrowc.Chunk(length=arr.length, offset=arr.offset, null_count=arr.null_count,
                            validity=bufs[0], views=bufs[1], data=bufs[2:2 + n_data],
                            data_sizes=[int(x) for x in sizes], _arr=None)
    return arrowc.Chunk(length=arr.length, offset=arr.offset, null_count=arr.null_count,
                        validity=bufs[0], views=bufs[1], data=[bufs[2]], data_sizes=[-1], _arr=None)


class NanoView:
    """A polars Series imported through nanoarrow. `chunks[k].view_addr` is
    the `struct ArrowArrayView*` the kernel receives as a uint64 argument."""

    def __init__(self, series=None, *, stream_ptr=None, capsule=None, structs=None):
        self.err = ctypes.create_string_buffer(ERR_SIZE)
        self._owned = structs is None
        if structs is not None:  # a hand-built (schema, array) pair, borrowed: never released here
            self.schema, arr = structs
            self._capsule = None
            self.format = self.schema.format.decode()
            view = ctypes.create_string_buffer(VIEW_SIZE)
            rc = lib.sm_view_set(view, ctypes.byref(self.schema), ctypes.byref(arr), self.err)
            if rc != 0:
                raise NanoError(f"sm_view_set rc={rc}: {lib.sm_error_message(self.err).decode()}")
            self.chunks = [NanoChunk(arr, view)]
            self.storage_type = lib.sm_view_storage_type(view)
            return
        if series is not None:
            capsule = series.__arrow_c_stream__()
            stream_ptr = arrowc._PyCapsule_GetPointer(capsule, b"arrow_array_stream")
        self._capsule = capsule
        self.schema = arrowc.ArrowSchema()
        arr = arrowc.ArrowArray()
        arr2 = arrowc.ArrowArray()
        view = ctypes.create_string_buffer(VIEW_SIZE)
        rc = lib.sm_import_single(stream_ptr, ctypes.byref(self.schema), ctypes.byref(arr),
                                  ctypes.byref(arr2), view, self.err)
        if rc < 0:
            raise NanoError(f"sm_import_single rc={rc}: {lib.sm_error_message(self.err).decode()}")
        self.format = self.schema.format.decode()
        if rc == 0:
            self.chunks = [NanoChunk(arr, view)]
        else:
            # multi-chunk: arr and arr2 are the first two, keep pulling in ctypes
            stream = ctypes.cast(stream_ptr, ctypes.POINTER(arrowc.ArrowArrayStream)).contents
            arrays = [arr, arr2]
            while True:
                a = arrowc.ArrowArray()
                if stream.get_next(ctypes.byref(stream), ctypes.byref(a)) != 0:
                    raise NanoError(stream.get_last_error(ctypes.byref(stream)))
                if not a.release:
                    break
                arrays.append(a)
            self.chunks = []
            for a in arrays:
                v = ctypes.create_string_buffer(VIEW_SIZE)
                rc = lib.sm_view_set(v, ctypes.byref(self.schema), ctypes.byref(a), self.err)
                if rc != 0:
                    raise NanoError(f"sm_view_set rc={rc}: {lib.sm_error_message(self.err).decode()}")
                self.chunks.append(NanoChunk(a, v))
        self.storage_type = lib.sm_view_storage_type(self.chunks[0].view)

    @property
    def n_rows(self):
        return sum(c.array.length for c in self.chunks)

    def validate(self, level: int) -> None:
        """nanoarrow's ArrowArrayViewValidate at `level` (1 minimal, 2 default, 3 full)."""
        for c in self.chunks:
            rc = lib.sm_view_validate(c.view, level, self.err)
            if rc != 0:
                raise NanoError(f"validate(level={level}) rc={rc}: {lib.sm_error_message(self.err).decode()}")

    def view_addrs(self) -> np.ndarray:
        return np.array([c.view_addr for c in self.chunks], dtype=np.uint64)

    def arrowc_chunks(self) -> list[arrowc.Chunk]:
        for c in self.chunks:
            if c.arrowc_chunk is None:
                c.arrowc_chunk = _arrowc_chunk(c.array, self.format)
        return [c.arrowc_chunk for c in self.chunks]

    def release(self):
        for c in self.chunks:
            lib.sm_view_reset(c.view)
            if self._owned:
                lib.sm_array_release(ctypes.byref(c.array))
        if self._owned:
            lib.sm_schema_release(ctypes.byref(self.schema))
        self.chunks = []

    def __del__(self):  # pragma: no cover
        try:
            if self.chunks:
                self.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The numba side: `int64 sm_get_string(void* view, int64 i, const uint8_t** data)`
# called through an address held in an ARGUMENT. Returns (data_addr, length);
# length -1 means null, -2 means the checked accessor refused the element.
# ---------------------------------------------------------------------------
@intrinsic
def call_get_string(typingctx, fn_t, view_t, i_t):
    ints = (types.uint64, types.int64, types.intp, types.uintp)
    if fn_t not in ints or view_t not in ints or i_t not in ints:
        return None
    ret = types.UniTuple(types.int64, 2)
    sig = ret(fn_t, view_t, i_t)

    def codegen(context, builder, signature, args):
        fn, view, i = args
        i8p = ir.IntType(8).as_pointer()
        i64 = ir.IntType(64)
        fnty = ir.FunctionType(i64, [i8p, i64, i8p.as_pointer()])
        fnptr = builder.inttoptr(fn, fnty.as_pointer())
        vp = builder.inttoptr(view, i8p)
        slot = cgutils.alloca_once(builder, i8p)
        ln = builder.call(fnptr, [vp, i, slot])
        data = builder.ptrtoint(builder.load(slot), i64)
        return context.make_tuple(builder, ret, [data, ln])

    return sig, codegen


def get_string_py(chunk: NanoChunk, i: int, checked=False):
    """ctypes call to the same accessor, for tests and the corruption probes."""
    p = _vp()
    fn = lib.sm_get_string_checked if checked else lib.sm_get_string
    ln = fn(chunk.view, i, ctypes.byref(p))
    if ln < 0:
        return ln, None
    return ln, ctypes.string_at(p.value, ln)
