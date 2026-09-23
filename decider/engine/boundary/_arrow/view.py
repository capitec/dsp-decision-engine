from __future__ import annotations

import ctypes

import numpy as np

from decider.engine.boundary._arrow import _shim
from decider.engine.boundary._arrow._shim import GATHER_ADDR, STORAGE_TYPE_NAMES, lib, stream_pointer
from decider.engine.boundary._arrow.kernels import Columns, Rows, materialize_columns, materialize_rows
from decider.engine.boundary._arrow.plan import (
    BOOL, CODE, F64, I64, ROW_DTYPES, STR, ArrowImportError, ArrowKindError, FramePlan,
)
from decider.engine.ir.decls import FeatureKind


class _RowPlan(ctypes.Structure):
    _fields_ = [
        ("ncols", ctypes.c_int32), ("_pad", ctypes.c_int32), ("cols", ctypes.c_void_p),
        ("f64", ctypes.c_void_p), ("i64", ctypes.c_void_p), ("b8", ctypes.c_void_p),
        ("i32", ctypes.c_void_p), ("span", ctypes.c_void_p), ("valid", ctypes.c_void_p),
    ]


assert ctypes.sizeof(_RowPlan) == _shim.PLAN_SIZE, (ctypes.sizeof(_RowPlan), _shim.PLAN_SIZE)


class FrameView:
    """Reusable import state for one `FramePlan`: binds one polars frame at a time.

    `bind(df)` exports the whole frame once and resolves every declared column;
    `gather(i)` fills the typed row buffers (`f64`, `i64`, `b8`, `i32`, `span`,
    `valid`) for row `i`. STR spans point into polars' memory and dangle after
    `release()`. Not thread-safe: use one view per thread.

    Example::

        with FrameView(FramePlan([("income", FeatureKind.F64)], df.columns)).bind(df) as fv:
            cols = fv.materialize_columns()
    """

    def __init__(self, plan: FramePlan) -> None:
        self.plan = plan
        n = plan.ncols
        self._schema = ctypes.create_string_buffer(_shim.SCHEMA_SIZE)
        self._array = ctypes.create_string_buffer(_shim.ARRAY_SIZE)
        self._array2 = ctypes.create_string_buffer(_shim.ARRAY_SIZE)
        self._view = ctypes.create_string_buffer(_shim.VIEW_SIZE)
        self._err = ctypes.create_string_buffer(_shim.ERR_SIZE)
        self._p = tuple(ctypes.addressof(b) for b in
                        (self._schema, self._array, self._array2, self._view, self._err))
        self._cols = ctypes.create_string_buffer(_shim.COLDESC_SIZE * max(n, 1))
        # One buffer per kind, never zero-length: numba types a 0-d and a 1-d array differently.
        nf, ni, nb, nc, ns = plan.counts
        self.f64 = np.zeros(max(nf, 1), dtype=ROW_DTYPES[F64])
        self.i64 = np.zeros(max(ni, 1), dtype=ROW_DTYPES[I64])
        self.b8 = np.zeros(max(nb, 1), dtype=ROW_DTYPES[BOOL])
        self.i32 = np.zeros(max(nc, 1), dtype=ROW_DTYPES[CODE])
        self.span = np.zeros(max(2 * ns, 2), dtype=ROW_DTYPES[STR])
        self.valid = np.zeros(max(n, 1), dtype=np.uint8)
        self._rowplan = _RowPlan(
            n, 0, ctypes.addressof(self._cols),
            self.f64.ctypes.data, self.i64.ctypes.data, self.b8.ctypes.data,
            self.i32.ctypes.data, self.span.ctypes.data, self.valid.ctypes.data,
        )
        # Per column: (data address advanced by offset, validity address, offset, child view).
        self.addrs = np.zeros((max(n, 1), 4), dtype=np.uint64)
        self.plan_addr: int = ctypes.addressof(self._rowplan)
        self.gather_addr: int = GATHER_ADDR
        self.n = 0
        self._capsule = None
        self.bound = False

    def bind(self, df) -> FrameView:
        """Export `df` and resolve every declared column; `ArrowKindError` names a column its kind can't read.

        polars rechunks the exported frame in place, so a multi-chunk frame arrives as one chunk.
        """
        if self.bound:
            self.release()
        plan = self.plan
        if tuple(df.columns) != plan.columns:
            raise ValueError(
                f"frame columns {list(df.columns)!r} differ from the plan's "
                f"{list(plan.columns)!r}; build a FramePlan for this frame's schema"
            )
        capsule = df.__arrow_c_stream__()
        s, a, a2, v, e = self._p
        rc = lib.sm_import_frame(stream_pointer(capsule), s, a, a2, v, e)
        if rc < 0:
            raise ArrowImportError(f"sm_import_frame rc={rc}: {lib.sm_error_message(e).decode()}")
        self._capsule = capsule
        self.bound = True
        if rc == 1:
            self.release()
            # ponytail: polars always sends one chunk; read each chunk if a polars upgrade stops that.
            raise ArrowImportError("the Arrow stream has more than one chunk; rechunk the frame first")
        self._resolve(v)
        self.n = lib.sm_view_length(v)
        return self

    def _resolve(self, view_addr: int) -> None:
        plan = self.plan
        rc = lib.sm_resolve_all(
            view_addr, plan.kinds.ctypes.data, plan.slots.ctypes.data, plan.child_idx.ctypes.data,
            plan.fill_f64.ctypes.data, plan.fill_i64.ctypes.data, plan.ncols,
            self.plan_addr, self.addrs.ctypes.data,
        )
        if rc == 0:
            return
        c = -rc - 1
        name, kind = plan.names[c], FeatureKind(int(plan.kinds[c]))
        arrow_type = self.arrow_type(name)
        self.release()
        err = ArrowKindError(_mismatch_message(name, kind, arrow_type))
        err.column, err.kind, err.arrow_type = name, kind, arrow_type
        raise err

    def release(self) -> None:
        """Hand polars its buffers back. Idempotent."""
        if not self.bound:
            return
        s, a, a2, v, _ = self._p
        lib.sm_view_reset(v)
        lib.sm_array_release(a)
        lib.sm_array_release(a2)
        lib.sm_schema_release(s)
        self._capsule = None
        self.bound = False
        self.n = 0

    def __enter__(self) -> FrameView:
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def arrow_type(self, name: str) -> str:
        """nanoarrow's rendering of a column's Arrow type, e.g. `"string_view"` or `"decimal128(38, 2)"`."""
        self._require_bound()
        buf = ctypes.create_string_buffer(256)
        lib.sm_schema_child_to_string(self._p[0], self.plan.columns.index(name), ctypes.addressof(buf), 256)
        return buf.value.decode()

    def child_view(self, name: str) -> int:
        """Address of a column's `ArrowArrayView`, the view argument of `call_get_string`."""
        self._require_bound()
        return lib.sm_view_child(self._p[3], self.plan.columns.index(name))

    def dictionary(self, name: str) -> tuple[str | None, ...] | None:
        """The exported dictionary of a Categorical/Enum column in index order, or `None`.

        A CODE value indexes this, not `Series.cat.get_categories()`: polars'
        physical codes are process-global but its Arrow export carries a
        per-batch dictionary.
        """
        self._require_bound()
        d = lib.sm_view_dictionary(self.child_view(name))
        if not d:
            return None
        p = ctypes.c_void_p()
        out = []
        for i in range(lib.sm_view_length(d)):
            ln = lib.sm_get_string(d, i, ctypes.byref(p))
            out.append(None if ln < 0 else ctypes.string_at(p.value, ln).decode("utf-8"))
        return tuple(out)

    def describe(self) -> list[dict]:
        """One record per declared column: Arrow and storage type, kind, slot, offset, nulls, dictionary."""
        self._require_bound()
        out = []
        for c, name in enumerate(self.plan.names):
            v = self.child_view(name)
            d = lib.sm_view_dictionary(v)
            out.append({
                "name": name,
                "arrow_type": self.arrow_type(name),
                "storage_type": STORAGE_TYPE_NAMES.get(lib.sm_view_storage_type(v), "?"),
                "kind": FeatureKind(int(self.plan.kinds[c])).name,
                "slot": int(self.plan.slots[c]),
                "offset": lib.sm_view_offset(v),
                "length": lib.sm_view_length(v),
                "null_count": lib.sm_view_null_count(v),
                "has_validity": bool(lib.sm_view_has_validity(v)),
                "n_variadic_buffers": lib.sm_view_n_variadic_buffers(v),
                "dictionary_length": lib.sm_view_length(d) if d else None,
                "dictionary_storage_type": (STORAGE_TYPE_NAMES.get(lib.sm_view_storage_type(d), "?")
                                            if d else None),
            })
        return out

    def gather(self, i: int) -> None:
        """Fill the row buffers with row `i`."""
        self._require_bound()
        if not 0 <= i < self.n:
            raise IndexError(f"row {i} of {self.n}")
        lib.sm_gather_row(self.plan_addr, i)

    def string(self, slot: int) -> bytes | None:
        """The bytes of STR slot `slot` in the last gathered row (a copy), or `None` for a null."""
        addr, ln = int(self.span[2 * slot]), int(self.span[2 * slot + 1])
        return None if ln < 0 else ctypes.string_at(addr, ln)

    def materialize(self) -> Rows:
        """Every row, as `(n, slots)` arrays per kind."""
        self._require_bound()
        n = self.n
        nf, ni, nb, nc, ns = self.plan.counts
        out = Rows(
            f64=np.empty((n, max(nf, 1)), np.float64), i64=np.empty((n, max(ni, 1)), np.int64),
            b8=np.empty((n, max(nb, 1)), np.bool_), i32=np.empty((n, max(nc, 1)), np.int32),
            span=np.empty((n, max(2 * ns, 2)), np.int64),
            valid=np.empty((n, max(self.plan.ncols, 1)), np.uint8),
        )
        materialize_rows(
            np.uint64(self.gather_addr), np.uint64(self.plan_addr), n,
            self.f64, self.i64, self.b8, self.i32, self.span, self.valid,
            out.f64, out.i64, out.b8, out.i32, out.span, out.valid,
        )
        return out

    def materialize_columns(self) -> Columns:
        """Every row, as one C-contiguous column per slot: `f64[j]` is F64 slot `j`, and so on."""
        self._require_bound()
        n = self.n
        nf, ni, nb, nc, ns = self.plan.counts
        out = Columns(
            f64=np.empty((nf, n), np.float64), i64=np.empty((ni, n), np.int64),
            b8=np.empty((nb, n), np.bool_), i32=np.empty((nc, n), np.int32),
            span=np.empty((ns, n, 2), np.int64), valid=np.empty((self.plan.ncols, n), np.bool_),
        )
        materialize_columns(
            np.uint64(self.gather_addr), np.uint64(self.plan_addr), n,
            self.f64, self.i64, self.b8, self.i32, self.span, self.valid,
            out.f64, out.i64, out.b8, out.i32, out.span, out.valid,
        )
        return out

    def strings(self, slot: int) -> list[bytes | None]:
        """Every row's bytes for STR slot `slot` (copies), `None` for a null."""
        return self.materialize().strings(slot)

    def _require_bound(self) -> None:
        if not self.bound:
            raise RuntimeError("FrameView is not bound to a frame; call bind(df) first")


def _mismatch_message(name: str, kind: FeatureKind, arrow_type: str) -> str:
    hint = ""
    if arrow_type.startswith("decimal"):
        hint = " Cast Decimal to a scaled integer in the frame tier before the boundary."
    elif arrow_type.startswith(("list", "struct", "fixed_size_list", "large_list", "map")):
        hint = " Nested columns do not enter the kernel; split the kernel around them."
    elif arrow_type.startswith("dictionary") and kind is STR:
        hint = " A Categorical/Enum column is read by its dictionary codes; declare it `str` (CODE)."
    elif arrow_type.startswith(("string", "large_string")) and kind is CODE:
        hint = " A String column is a STR feature; CODE is a dictionary index."
    return f"column {name!r} is Arrow {arrow_type} but is declared {kind.name}, which cannot read it.{hint}"
