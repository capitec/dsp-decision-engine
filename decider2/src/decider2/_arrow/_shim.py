"""The ctypes side of `decider2._arrow._nashim`, plus the numba intrinsics
that call into it.

Importing this module loads the compiled extension. There is no pure-Python
fallback, by decision (docs/BOUNDARY-REWORK.md §4.4 and the packaging
strand's "Decision taken"): a box where the extension is missing or will
not load gets an `ImportError` naming the platform, the interpreter and the
cause, from `import decider2._arrow._shim` onwards. `decider2._arrow.
diagnose()` is importable without the extension and reports the same.

Every `sm_*` function is a `ctypes.CFUNCTYPE(...)(address)` built from the
addresses `_nashim` exports; `GATHER_ADDR` and `GET_STRING_ADDR` are the
two the kernels take as `uint64` ARGUMENTS. The intrinsics are pure LLVM
(`inttoptr` + `call`/`load`) with no dynamic global, so a `cache=True`
kernel that uses them disk-caches (EXPERIMENTS.md §W; proven for this
module by `tests/test_shim.py`).
"""
from __future__ import annotations

import ctypes
import platform
import sys

from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import intrinsic

EXPECTED_ABI = 1
_EXT = "decider2._arrow._nashim"


def _platform_line() -> str:
    impl = platform.python_implementation()
    return f"{platform.platform()} / {impl} {platform.python_version()} ({platform.machine()})"


def missing_message(cause: BaseException) -> str:
    """The one message a broken or absent extension produces, wherever it is
    first noticed: platform, interpreter, cause, and what to do."""
    return (
        f"decider2's compiled Arrow shim ({_EXT}) could not be loaded on "
        f"{_platform_line()}: {type(cause).__name__}: {cause}. decider2 has no "
        "pure-Python path for this by design. Install a decider2 wheel built for "
        "this platform (cp310-abi3-<platform>), or build from the sdist with a C "
        "compiler and the CPython development headers present. "
        "`python -c \"import decider2._arrow, json; print(json.dumps(decider2._arrow.diagnose(), indent=1))\"` "
        "reports what is installed."
    )


def _load():
    try:
        from decider2._arrow import _nashim
    except ImportError as exc:  # no wheel for this platform, build skipped, truncated .so, ...
        raise ImportError(missing_message(exc)) from exc
    abi = getattr(_nashim, "ABI", None)
    if abi != EXPECTED_ABI:
        raise ImportError(missing_message(
            RuntimeError(f"extension ABI {abi!r} != expected {EXPECTED_ABI} (stale build?)")
        ))
    return _nashim


_nashim = _load()

_vp, _i64, _i32, _int, _sz, _cp, _dbl = (
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_double,
)


def _fn(name: str, restype, *argtypes):
    return ctypes.CFUNCTYPE(restype, *argtypes)(getattr(_nashim, name))


class lib:
    """The shim's functions as ctypes callables, one attribute per `sm_*`."""

    sm_sizeof_array_view = _fn("sm_sizeof_array_view", _sz)
    sm_sizeof_schema = _fn("sm_sizeof_schema", _sz)
    sm_sizeof_array = _fn("sm_sizeof_array", _sz)
    sm_sizeof_error = _fn("sm_sizeof_error", _sz)
    sm_sizeof_coldesc = _fn("sm_sizeof_coldesc", _sz)
    sm_sizeof_plan = _fn("sm_sizeof_plan", _sz)
    sm_error_message = _fn("sm_error_message", _cp, _vp)
    sm_import_frame = _fn("sm_import_frame", _int, _vp, _vp, _vp, _vp, _vp, _vp)
    sm_stream_get_next = _fn("sm_stream_get_next", _int, _vp, _vp, _vp)
    sm_array_is_valid = _fn("sm_array_is_valid", _int, _vp)
    sm_view_set = _fn("sm_view_set", _int, _vp, _vp, _vp, _vp)
    sm_view_validate = _fn("sm_view_validate", _int, _vp, _int, _vp)
    sm_view_storage_type = _fn("sm_view_storage_type", _int, _vp)
    sm_view_length = _fn("sm_view_length", _i64, _vp)
    sm_view_offset = _fn("sm_view_offset", _i64, _vp)
    sm_view_null_count = _fn("sm_view_null_count", _i64, _vp)
    sm_view_n_children = _fn("sm_view_n_children", _i64, _vp)
    sm_view_child = _fn("sm_view_child", _vp, _vp, _i64)
    sm_view_dictionary = _fn("sm_view_dictionary", _vp, _vp)
    sm_view_n_variadic_buffers = _fn("sm_view_n_variadic_buffers", _int, _vp)
    sm_view_has_validity = _fn("sm_view_has_validity", _int, _vp)
    sm_view_reset = _fn("sm_view_reset", None, _vp)
    sm_array_release = _fn("sm_array_release", None, _vp)
    sm_schema_release = _fn("sm_schema_release", None, _vp)
    sm_schema_n_children = _fn("sm_schema_n_children", _i64, _vp)
    sm_schema_format = _fn("sm_schema_format", _cp, _vp)
    sm_schema_child_name = _fn("sm_schema_child_name", _cp, _vp, _i64)
    sm_schema_child_format = _fn("sm_schema_child_format", _cp, _vp, _i64)
    sm_schema_child_to_string = _fn("sm_schema_child_to_string", _i64, _vp, _i64, _vp, _i64)
    sm_is_null = _fn("sm_is_null", _int, _vp, _i64)
    sm_get_f64 = _fn("sm_get_f64", _dbl, _vp, _i64)
    sm_get_i64 = _fn("sm_get_i64", _i64, _vp, _i64)
    sm_get_string = _fn("sm_get_string", _i64, _vp, _i64, ctypes.POINTER(_vp))
    sm_resolve_col = _fn("sm_resolve_col", _int, _vp, _i32, _i32, _dbl, _i64, _vp)
    sm_resolve_all = _fn("sm_resolve_all", _int, _vp, _vp, _vp, _vp, _vp, _vp, _i32, _vp, _vp)
    sm_gather_row = _fn("sm_gather_row", None, _vp, _i64)


# The two addresses a kernel receives as uint64 ARGUMENTS.
GATHER_ADDR: int = _nashim.sm_gather_row
GET_STRING_ADDR: int = _nashim.sm_get_string

NANOARROW_VERSION: str = _nashim.NANOARROW_VERSION
EXTENSION_FILE: str = _nashim.__file__
ABI: int = _nashim.ABI

VIEW_SIZE = lib.sm_sizeof_array_view()
SCHEMA_SIZE = lib.sm_sizeof_schema()
ARRAY_SIZE = lib.sm_sizeof_array()
ERR_SIZE = lib.sm_sizeof_error()
COLDESC_SIZE = lib.sm_sizeof_coldesc()
PLAN_SIZE = lib.sm_sizeof_plan()

# nanoarrow validation levels (diagnostics only; never on the request path)
VALIDATE_MINIMAL, VALIDATE_DEFAULT, VALIDATE_FULL = 1, 2, 3

# `enum ArrowType` (nanoarrow.h) by value, for messages. Enum VALUES are
# API, not layout.
STORAGE_TYPE_NAMES = {
    0: "uninitialized", 1: "null", 2: "bool", 3: "uint8", 4: "int8", 5: "uint16", 6: "int16",
    7: "uint32", 8: "int32", 9: "uint64", 10: "int64", 11: "half_float", 12: "float",
    13: "double", 14: "string", 15: "binary", 16: "fixed_size_binary", 17: "date32",
    18: "date64", 19: "timestamp", 20: "time32", 21: "time64", 22: "interval_months",
    23: "interval_day_time", 24: "decimal128", 25: "decimal256", 26: "list", 27: "struct",
    28: "sparse_union", 29: "dense_union", 30: "dictionary", 31: "map", 32: "extension",
    33: "fixed_size_list", 34: "duration", 35: "large_string", 36: "large_binary",
    37: "large_list", 38: "interval_month_day_nano", 39: "run_end_encoded",
    40: "binary_view", 41: "string_view", 42: "decimal32", 43: "decimal64", 44: "list_view",
    45: "large_list_view",
}

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def stream_pointer(capsule) -> int:
    """The `struct ArrowArrayStream*` inside a `__arrow_c_stream__()` capsule."""
    return _PyCapsule_GetPointer(capsule, b"arrow_array_stream")


# --- intrinsics: pointer-as-argument calls and raw loads --------------------
#
# Each takes the address as an integer VALUE (uint64/int64/intp) so the
# compiled code contains no process-specific constant (EXPERIMENTS.md §W).

_INT_TYPES = (types.uint64, types.int64, types.intp, types.uintp)


@intrinsic
def call_gather(typingctx, fn_t, plan_t, i_t):
    """`void sm_gather_row(const SmRowPlan*, int64 i)` through `fn_t`."""
    if fn_t not in _INT_TYPES or plan_t not in _INT_TYPES or i_t not in _INT_TYPES:
        return None
    sig = types.none(fn_t, plan_t, i_t)

    def codegen(context, builder, signature, args):
        fn, plan, i = args
        i8p = ir.IntType(8).as_pointer()
        i64 = ir.IntType(64)
        fnty = ir.FunctionType(ir.VoidType(), [i8p, i64])
        i = builder.sext(i, i64) if i.type.width < 64 else i
        builder.call(builder.inttoptr(fn, fnty.as_pointer()), [builder.inttoptr(plan, i8p), i])
        return context.get_dummy_value()

    return sig, codegen


@intrinsic
def call_get_string(typingctx, fn_t, view_t, i_t):
    """`int64 sm_get_string(const ArrowArrayView*, int64 i, const uint8_t**)`
    through `fn_t`. Returns `(data_addr, length)`; length -1 = null."""
    if fn_t not in _INT_TYPES or view_t not in _INT_TYPES or i_t not in _INT_TYPES:
        return None
    ret = types.UniTuple(types.int64, 2)
    sig = ret(fn_t, view_t, i_t)

    def codegen(context, builder, signature, args):
        fn, view, i = args
        i8p = ir.IntType(8).as_pointer()
        i64 = ir.IntType(64)
        fnty = ir.FunctionType(i64, [i8p, i64, i8p.as_pointer()])
        i = builder.sext(i, i64) if i.type.width < 64 else i
        slot = cgutils.alloca_once(builder, i8p)
        ln = builder.call(builder.inttoptr(fn, fnty.as_pointer()),
                          [builder.inttoptr(view, i8p), i, slot])
        data = builder.ptrtoint(builder.load(slot), i64)
        return context.make_tuple(builder, ret, [data, ln])

    return sig, codegen


def _load_intrinsic(llvm_type, numba_type):
    @intrinsic
    def load(typingctx, addr_t):
        if not isinstance(addr_t, types.Integer):
            return None

        def codegen(context, builder, sig, args):
            return builder.load(builder.inttoptr(args[0], llvm_type.as_pointer()))

        return numba_type(addr_t), codegen

    return load


load_u8 = _load_intrinsic(ir.IntType(8), types.uint8)
load_i64 = _load_intrinsic(ir.IntType(64), types.int64)
load_f64 = _load_intrinsic(ir.DoubleType(), types.float64)


def info() -> dict:
    """What loaded, for `diagnose()`."""
    return {
        "extension": EXTENSION_FILE,
        "nanoarrow": NANOARROW_VERSION,
        "abi": ABI,
        "sizes": {"array_view": VIEW_SIZE, "schema": SCHEMA_SIZE, "array": ARRAY_SIZE,
                  "error": ERR_SIZE, "coldesc": COLDESC_SIZE, "plan": PLAN_SIZE},
        "gather_addr": hex(GATHER_ADDR),
        "get_string_addr": hex(GET_STRING_ADDR),
        "python_executable": sys.executable,
    }
