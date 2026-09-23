"""The compiled backend: nashim.py from experimentation/nanoarrow-accessors,
re-pointed at the `_nashim` extension module instead of a hand-built .so.

The only change from the experiment's file: instead of `ctypes.CDLL(path)`
and `lib.sm_*`, each function is a `ctypes.CFUNCTYPE(...)(address)` built
from the addresses `_nashim` exports. Nothing in this file knows how a
string is laid out in memory.

Importing this module raises ImportError when the extension is absent;
`d2shim.strings` catches that and falls back to `d2shim.pure`.
"""
from __future__ import annotations

import ctypes

import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

from . import _arrowc as arrowc
from . import _nashim  # ImportError here == no compiled backend

EXPECTED_ABI = 1
if getattr(_nashim, "ABI", None) != EXPECTED_ABI:
    raise ImportError(f"d2shim._nashim ABI {getattr(_nashim, 'ABI', None)!r} != expected {EXPECTED_ABI}")

_vp, _i64, _int, _sz, _cp = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p


def _fn(name, restype, *argtypes):
    return ctypes.CFUNCTYPE(restype, *argtypes)(getattr(_nashim, name))


class _Lib:
    """Same attribute names as the experiment's `lib = ctypes.CDLL(...)`."""
    sm_sizeof_array_view = _fn("sm_sizeof_array_view", _sz)
    sm_sizeof_schema = _fn("sm_sizeof_schema", _sz)
    sm_sizeof_array = _fn("sm_sizeof_array", _sz)
    sm_sizeof_error = _fn("sm_sizeof_error", _sz)
    sm_import_single = _fn("sm_import_single", _int, _vp, _vp, _vp, _vp, _vp, _vp)
    sm_view_set = _fn("sm_view_set", _int, _vp, _vp, _vp, _vp)
    sm_view_validate = _fn("sm_view_validate", _int, _vp, _int, _vp)
    sm_view_storage_type = _fn("sm_view_storage_type", _int, _vp)
    sm_view_length = _fn("sm_view_length", _i64, _vp)
    sm_view_offset = _fn("sm_view_offset", _i64, _vp)
    sm_view_reset = _fn("sm_view_reset", None, _vp)
    sm_array_release = _fn("sm_array_release", None, _vp)
    sm_schema_release = _fn("sm_schema_release", None, _vp)
    sm_error_message = _fn("sm_error_message", _cp, _vp)
    sm_get_string = _fn("sm_get_string", _i64, _vp, _i64, ctypes.POINTER(_vp))
    sm_get_string_checked = _fn("sm_get_string_checked", _i64, _vp, _i64, ctypes.POINTER(_vp))


lib = _Lib
GET_STRING_ADDR = _nashim.sm_get_string
GET_STRING_CHECKED_ADDR = _nashim.sm_get_string_checked
NANOARROW_VERSION = _nashim.NANOARROW_VERSION
EXTENSION_FILE = _nashim.__file__

VIEW_SIZE = lib.sm_sizeof_array_view()
ERR_SIZE = lib.sm_sizeof_error()
assert lib.sm_sizeof_schema() == ctypes.sizeof(arrowc.ArrowSchema)
assert lib.sm_sizeof_array() == ctypes.sizeof(arrowc.ArrowArray)

STORAGE_NAMES = {1: "null (n)", 15: "utf8 (u)", 16: "binary", 36: "large_utf8 (U)", 37: "large_binary",
                 41: "utf8_view (vu)", 40: "binary_view"}
STRING_VIEW = 41
MINIMAL, DEFAULT, FULL = 1, 2, 3


class NanoError(RuntimeError):
    pass


class NanoChunk:
    __slots__ = ("array", "view", "view_addr")

    def __init__(self, array, view):
        self.array = array
        self.view = view
        self.view_addr = ctypes.addressof(view)


class NanoView:
    """A polars Series imported through nanoarrow. `chunks[k].view_addr` is
    the `struct ArrowArrayView*` the kernel receives as a uint64 argument."""

    def __init__(self, series):
        self.err = ctypes.create_string_buffer(ERR_SIZE)
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
        for c in self.chunks:
            rc = lib.sm_view_validate(c.view, level, self.err)
            if rc != 0:
                raise NanoError(f"validate(level={level}) rc={rc}: {lib.sm_error_message(self.err).decode()}")

    def release(self):
        for c in self.chunks:
            lib.sm_view_reset(c.view)
            lib.sm_array_release(ctypes.byref(c.array))
        lib.sm_schema_release(ctypes.byref(self.schema))
        self.chunks = []

    def __del__(self):  # pragma: no cover
        try:
            if self.chunks:
                self.release()
        except Exception:
            pass


@intrinsic
def call_get_string(typingctx, fn_t, view_t, i_t):
    """int64 sm_get_string(void* view, int64 i, const uint8_t** data), called
    through an address held in an ARGUMENT (never a captured global, or the
    kernel will not disk-cache). Returns (data_addr, length); -1 = null,
    -2 = the checked accessor refused the element."""
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
    p = _vp()
    fn = lib.sm_get_string_checked if checked else lib.sm_get_string
    ln = fn(chunk.view, i, ctypes.byref(p))
    if ln < 0:
        return ln, None
    return ln, ctypes.string_at(p.value, ln)
