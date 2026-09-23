from __future__ import annotations

import ctypes
import platform
import sys

from decider.engine.boundary._arrow._build import build


def missing_message(cause: BaseException) -> str:
    """The one message a shim that can't be built or loaded produces, wherever it is first noticed."""
    where = (f"{platform.platform()} / {platform.python_implementation()} "
             f"{platform.python_version()} ({platform.machine()})")
    return (
        f"decider's compiled Arrow shim could not be loaded on {where}: "
        f"{type(cause).__name__}: {cause}. There is no pure-Python path for the data boundary. "
        "Install a C compiler (set CC to choose one) so the shim is built on first import. "
        "`python -c \"import decider.engine.boundary._arrow as a, json; "
        "print(json.dumps(a.diagnose(), indent=1))\"` reports what is installed."
    )


def _load() -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(build()))
    except (OSError, RuntimeError) as exc:  # no compiler, a failed compile, a corrupt cached file
        raise ImportError(missing_message(exc)) from exc


lib = _load()

_vp, _i64, _i32, _int, _sz, _cp, _dbl = (
    ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32, ctypes.c_int, ctypes.c_size_t,
    ctypes.c_char_p, ctypes.c_double,
)
_SIGNATURES = {
    "sm_sizeof_array_view": (_sz,), "sm_sizeof_schema": (_sz,), "sm_sizeof_array": (_sz,),
    "sm_sizeof_error": (_sz,), "sm_sizeof_coldesc": (_sz,), "sm_sizeof_plan": (_sz,),
    "sm_nanoarrow_version": (_cp,),
    "sm_error_message": (_cp, _vp),
    "sm_import_frame": (_int, _vp, _vp, _vp, _vp, _vp, _vp),
    "sm_view_storage_type": (_int, _vp),
    "sm_view_length": (_i64, _vp),
    "sm_view_offset": (_i64, _vp),
    "sm_view_null_count": (_i64, _vp),
    "sm_view_child": (_vp, _vp, _i64),
    "sm_view_dictionary": (_vp, _vp),
    "sm_view_n_variadic_buffers": (_int, _vp),
    "sm_view_has_validity": (_int, _vp),
    "sm_array_release": (None, _vp),
    "sm_release": (None, _vp, _vp, _vp, _vp),
    "sm_schema_child_to_string": (_i64, _vp, _i64, _vp, _i64),
    "sm_get_string": (_i64, _vp, _i64, ctypes.POINTER(_vp)),
    "sm_resolve_all": (_int, _vp, _vp, _vp, _vp, _vp, _vp, _i32, _vp, _vp),
    "sm_gather_row": (None, _vp, _i64),
    "sm_columns": (None, _vp, _i64, _vp, _i32),
    "sm_array_move": (None, _vp, _vp),
}
for _name, (_restype, *_argtypes) in _SIGNATURES.items():
    _fn = getattr(lib, _name)
    _fn.restype, _fn.argtypes = _restype, _argtypes


def _address(name: str) -> int:
    return ctypes.cast(getattr(lib, name), ctypes.c_void_p).value


# The two addresses kernels receive as uint64 arguments.
GATHER_ADDR: int = _address("sm_gather_row")
GET_STRING_ADDR: int = _address("sm_get_string")

NANOARROW_VERSION: str = lib.sm_nanoarrow_version().decode()
VIEW_SIZE = lib.sm_sizeof_array_view()
SCHEMA_SIZE = lib.sm_sizeof_schema()
ARRAY_SIZE = lib.sm_sizeof_array()
ERR_SIZE = lib.sm_sizeof_error()
COLDESC_SIZE = lib.sm_sizeof_coldesc()
PLAN_SIZE = lib.sm_sizeof_plan()

# nanoarrow's `enum ArrowType` by value, for messages. Enum values are API, not layout.
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
    return _PyCapsule_GetPointer(capsule, b"arrow_array_stream")


def info() -> dict:
    return {
        "extension": lib._name,
        "nanoarrow": NANOARROW_VERSION,
        "sizes": {"array_view": VIEW_SIZE, "schema": SCHEMA_SIZE, "array": ARRAY_SIZE,
                  "error": ERR_SIZE, "coldesc": COLDESC_SIZE, "plan": PLAN_SIZE},
        "gather_addr": hex(GATHER_ADDR),
        "get_string_addr": hex(GET_STRING_ADDR),
        "python_executable": sys.executable,
    }
