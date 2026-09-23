"""ctypes loader for c/librowshim.so plus the two numba intrinsics the
probe kernels use: call rs_gather_row(plan*, i) and rs_get_string(view*, i)
through function-pointer ARGUMENTS (EXPERIMENTS.md SW rule)."""
from __future__ import annotations
import ctypes, os
import numpy as np
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(HERE, "c", "librowshim.so"))
_vp, _i64, _int = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
for n in ("rs_sizeof_view", "rs_sizeof_schema", "rs_sizeof_array", "rs_sizeof_error", "rs_sizeof_plan"):
    getattr(lib, n).restype = ctypes.c_size_t; getattr(lib, n).argtypes = []
lib.rs_import_frame.argtypes = [_vp] * 6; lib.rs_import_frame.restype = _int
lib.rs_set_array_full.argtypes = [_vp, _vp, _vp]; lib.rs_set_array_full.restype = _int
lib.rs_validate.argtypes = [_vp, _int, _vp]; lib.rs_validate.restype = _int
lib.rs_child.argtypes = [_vp, _int]; lib.rs_child.restype = _vp
lib.rs_dictionary.argtypes = [_vp]; lib.rs_dictionary.restype = _vp
lib.rs_storage_type.argtypes = [_vp]; lib.rs_storage_type.restype = _int
lib.rs_length.argtypes = [_vp]; lib.rs_length.restype = _i64
for n in ("rs_view_reset", "rs_array_release", "rs_schema_release"):
    getattr(lib, n).argtypes = [_vp]; getattr(lib, n).restype = None
lib.rs_error_message.argtypes = [_vp]; lib.rs_error_message.restype = ctypes.c_char_p
lib.rs_gather_row.argtypes = [_vp, _i64]; lib.rs_gather_row.restype = None
lib.rs_get_string.argtypes = [_vp, _i64, ctypes.POINTER(_vp)]; lib.rs_get_string.restype = _i64
lib.rs_get_f64.argtypes = [_vp, _i64]; lib.rs_get_f64.restype = ctypes.c_double
lib.rs_get_i64.argtypes = [_vp, _i64]; lib.rs_get_i64.restype = _i64
lib.rs_is_null.argtypes = [_vp, _i64]; lib.rs_is_null.restype = _int

GATHER_ADDR = ctypes.cast(lib.rs_gather_row, _vp).value
GET_STRING_ADDR = ctypes.cast(lib.rs_get_string, _vp).value
VIEW_SIZE, SCHEMA_SIZE, ARRAY_SIZE, ERR_SIZE, PLAN_SIZE = (
    lib.rs_sizeof_view(), lib.rs_sizeof_schema(), lib.rs_sizeof_array(), lib.rs_sizeof_error(), lib.rs_sizeof_plan())

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

F64, I64, BOOL, STR, CODE = 0, 1, 2, 3, 4


class RowPlan(ctypes.Structure):
    _fields_ = [("ncols", ctypes.c_int32), ("views", ctypes.POINTER(_vp)), ("kinds", ctypes.POINTER(ctypes.c_int8)),
                ("slots", ctypes.POINTER(ctypes.c_int32)), ("f64", ctypes.POINTER(ctypes.c_double)),
                ("i64", ctypes.POINTER(_i64)), ("b8", ctypes.POINTER(ctypes.c_uint8)),
                ("spans", ctypes.POINTER(_i64)), ("valid", ctypes.POINTER(ctypes.c_uint8))]


assert ctypes.sizeof(RowPlan) == PLAN_SIZE, (ctypes.sizeof(RowPlan), PLAN_SIZE)


class FrameImport:
    """Pooled structs: allocate once, reuse per call (the 'B pooled' shape)."""

    def __init__(self):
        self.schema = ctypes.create_string_buffer(SCHEMA_SIZE)
        self.array = ctypes.create_string_buffer(ARRAY_SIZE)
        self.array2 = ctypes.create_string_buffer(ARRAY_SIZE)
        self.view = ctypes.create_string_buffer(VIEW_SIZE)
        self.err = ctypes.create_string_buffer(ERR_SIZE)
        self._p = tuple(ctypes.cast(b, _vp) for b in (self.schema, self.array, self.array2, self.view, self.err))
        self.live = False

    def import_df(self, df):
        cap = df.__arrow_c_stream__()
        ptr = _PyCapsule_GetPointer(cap, b"arrow_array_stream")
        s, a, a2, v, e = self._p
        rc = lib.rs_import_frame(ptr, s, a, a2, v, e)
        if rc < 0:
            raise RuntimeError(f"rc={rc} {lib.rs_error_message(e)}")
        self.live = True
        self._cap = cap
        return rc

    def child(self, k):
        return lib.rs_child(self._p[3], k)

    def release(self):
        if self.live:
            s, a, a2, v, e = self._p
            lib.rs_view_reset(v); lib.rs_array_release(a); lib.rs_array_release(a2); lib.rs_schema_release(s)
            self.live = False


def make_plan(child_addrs, kinds, slots, nf, ni, nb, ns):
    """Row buffers + a RowPlan over them. Returns (plan, addr, bufs)."""
    n = len(child_addrs)
    views = (_vp * n)(*child_addrs)
    k = (ctypes.c_int8 * n)(*kinds)
    s = (ctypes.c_int32 * n)(*slots)
    bf = np.zeros(max(nf, 1), np.float64); bi = np.zeros(max(ni, 1), np.int64)
    bb = np.zeros(max(nb, 1), np.uint8); bs = np.zeros(max(2 * ns, 1), np.int64); bv = np.zeros(max(n, 1), np.uint8)
    plan = RowPlan(n, views, k, s, bf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   bi.ctypes.data_as(ctypes.POINTER(_i64)), bb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                   bs.ctypes.data_as(ctypes.POINTER(_i64)), bv.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    keep = (views, k, s)
    return plan, ctypes.addressof(plan), (bf, bi, bb, bs, bv), keep


@intrinsic
def call_gather(typingctx, fn_t, plan_t, i_t):
    ints = (types.uint64, types.int64, types.intp, types.uintp)
    if fn_t not in ints or plan_t not in ints or i_t not in ints:
        return None
    sig = types.none(fn_t, plan_t, i_t)

    def codegen(context, builder, signature, args):
        fn, plan, i = args
        i8p = ir.IntType(8).as_pointer()
        fnty = ir.FunctionType(ir.VoidType(), [i8p, ir.IntType(64)])
        builder.call(builder.inttoptr(fn, fnty.as_pointer()), [builder.inttoptr(plan, i8p), i])
        return context.get_dummy_value()
    return sig, codegen


@intrinsic
def call_get_string(typingctx, fn_t, view_t, i_t):
    ints = (types.uint64, types.int64, types.intp, types.uintp)
    if fn_t not in ints or view_t not in ints or i_t not in ints:
        return None
    ret = types.UniTuple(types.int64, 2)
    sig = ret(fn_t, view_t, i_t)

    def codegen(context, builder, signature, args):
        fn, view, i = args
        i8p = ir.IntType(8).as_pointer(); i64 = ir.IntType(64)
        fnty = ir.FunctionType(i64, [i8p, i64, i8p.as_pointer()])
        slot = cgutils.alloca_once(builder, i8p)
        ln = builder.call(builder.inttoptr(fn, fnty.as_pointer()), [builder.inttoptr(view, i8p), i, slot])
        data = builder.ptrtoint(builder.load(slot), i64)
        return context.make_tuple(builder, ret, [data, ln])
    return sig, codegen


@intrinsic
def load_u8(typingctx, addr):
    if not isinstance(addr, types.Integer):
        return None

    def codegen(context, builder, sig, args):
        return builder.load(builder.inttoptr(args[0], ir.IntType(8).as_pointer()))
    return types.uint8(addr), codegen
