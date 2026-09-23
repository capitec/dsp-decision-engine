"""`FramePlan` + `FrameView`: a polars DataFrame crosses the boundary ONCE.

docs/BOUNDARY-REWORK.md §1.1:

    df.__arrow_c_stream__()          the whole frame, one capsule
      -> sm_import_frame             schema + first chunk -> ArrowArrayView (children = columns)
      -> sm_resolve_all              ONE C call: a ColDesc per declared column
                                     {data*, validity*, offset, kind, slot, fill}
                                     + the flat address table
      -> sm_gather_row(plan, i)      ONE C call per row -> the typed row
         (f64[nf], i64[ni], b8[nb], i32[nc], span[2*ns], valid[ncols])

A `FramePlan` is decided once per (declared inputs, frame column order);
a `FrameView` is the pooled per-pipeline (per-thread) object that owns the
opaque nanoarrow structs, the `ColDesc` array, the `RowPlan` and the row
buffers, and binds one frame at a time. Every address a kernel sees —
`gather_addr`, `plan_addr`, `addrs`, the row buffers — is a value it
receives as an argument. Nothing here is a consumer yet: Stages 2–4 wire
trees, `apply()` and `score()` onto it.

Kinds are `decider2.types.FeatureKind` (F64=0, I64=1, BOOL=2, CODE=3,
STR=4); a STR feature's slot holds `(address, length)` with `length == -1`
for a null, pointing straight into polars' memory for as long as the view
is bound.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from numba import njit

from decider2._arrow import _shim
from decider2._arrow._shim import (
    GATHER_ADDR, GET_STRING_ADDR, STORAGE_TYPE_NAMES, call_gather, lib, stream_pointer,
)
from decider2.types import FeatureKind

F64, I64, BOOL, CODE, STR = (
    FeatureKind.F64, FeatureKind.I64, FeatureKind.BOOL, FeatureKind.CODE, FeatureKind.STR,
)

# The typed row's buffer dtype per kind — `compile.driver._TYPED_DTYPES`'
# values, restated here so this package does not import the driver.
ROW_DTYPES: dict[FeatureKind, np.dtype] = {
    F64: np.dtype(np.float64),
    I64: np.dtype(np.int64),
    BOOL: np.dtype(np.bool_),
    CODE: np.dtype(np.int32),
    STR: np.dtype(np.int64),     # 2 per slot: (address, length)
}

_vp, _i64 = ctypes.c_void_p, ctypes.c_int64


class ArrowKindError(TypeError):
    """A declared column's Arrow type does not match its declared kind, or
    is one the kernel cannot read at all (Decimal, List, Struct, ...)."""


class ArrowImportError(RuntimeError):
    """`__arrow_c_stream__()` handed over something nanoarrow refused."""


@dataclass(frozen=True)
class ColumnSpec:
    """One declared input: which frame column, which kind, and what a null
    becomes in the row (`fill=None` -> NaN / 0 / False / -1 / null span)."""

    name: str
    kind: FeatureKind
    fill: Any = None


class FramePlan:
    """The schema-invariant half: decided once per (inputs, frame columns).

    `inputs` is the pipeline's declared input set — `ColumnSpec`s or
    `(name, kind)` pairs. `columns` is the frame's column order
    (`tuple(df.columns)`); child indices are looked up here so `bind()`
    never touches column names on the hot path. Slot `j` of kind `K` is the
    `j`-th input of kind `K` in `inputs` order — the same rule as
    `compile.driver._typed_layout`."""

    __slots__ = (
        "specs", "columns", "ncols", "kinds", "slots", "child_idx", "fill_f64", "fill_i64",
        "counts", "names",
    )

    def __init__(self, inputs: Iterable[ColumnSpec | tuple[str, FeatureKind]],
                 columns: Sequence[str]) -> None:
        specs = tuple(s if isinstance(s, ColumnSpec) else ColumnSpec(s[0], FeatureKind(s[1]))
                      for s in inputs)
        self.specs = specs
        self.columns = tuple(columns)
        position = {name: k for k, name in enumerate(self.columns)}
        missing = [s.name for s in specs if s.name not in position]
        if missing:
            raise KeyError(f"frame has no column(s) {missing!r}; it has {list(self.columns)!r}")
        n = len(specs)
        self.ncols = n
        self.names = tuple(s.name for s in specs)
        self.kinds = np.array([int(s.kind) for s in specs], dtype=np.int32)
        self.child_idx = np.array([position[s.name] for s in specs], dtype=np.int32)
        counts = [0] * len(FeatureKind)
        slots = np.zeros(n, dtype=np.int32)
        fill_f64 = np.full(max(n, 1), np.nan, dtype=np.float64)
        fill_i64 = np.zeros(max(n, 1), dtype=np.int64)
        for c, s in enumerate(specs):
            slots[c] = counts[s.kind]
            counts[s.kind] += 1
            if s.kind is CODE:
                fill_i64[c] = -1
            if s.fill is not None:
                if s.kind is F64:
                    fill_f64[c] = float(s.fill)
                elif s.kind in (I64, CODE, BOOL):
                    fill_i64[c] = int(s.fill)
                else:
                    raise ValueError(f"{s.name}: a STR feature has no fill; a null is length -1")
        self.slots = slots
        self.fill_f64 = fill_f64
        self.fill_i64 = fill_i64
        self.counts = tuple(counts)

    def __repr__(self) -> str:
        return f"FramePlan({list(zip(self.names, [FeatureKind(k).name for k in self.kinds]))})"


class _RowPlan(ctypes.Structure):
    _fields_ = [
        ("ncols", ctypes.c_int32), ("_pad", ctypes.c_int32), ("cols", _vp),
        ("f64", _vp), ("i64", _vp), ("b8", _vp), ("i32", _vp), ("span", _vp), ("valid", _vp),
    ]


assert ctypes.sizeof(_RowPlan) == _shim.PLAN_SIZE, (ctypes.sizeof(_RowPlan), _shim.PLAN_SIZE)


class FrameView:
    """Pooled per-pipeline import state for one `FramePlan`.

    `bind(df)` imports the frame and resolves every declared column; the
    kernel then takes `gather_addr`, `plan_addr` (and/or `addrs`) plus the
    row buffers as arguments. `release()` (or leaving the `with` block)
    hands polars its buffers back; the spans in `span` are dangling after
    that, by construction."""

    def __init__(self, plan: FramePlan) -> None:
        self.plan = plan
        n = plan.ncols
        # opaque nanoarrow structs, sized by the shim
        self._schema = ctypes.create_string_buffer(_shim.SCHEMA_SIZE)
        self._array = ctypes.create_string_buffer(_shim.ARRAY_SIZE)
        self._array2 = ctypes.create_string_buffer(_shim.ARRAY_SIZE)
        self._view = ctypes.create_string_buffer(_shim.VIEW_SIZE)
        self._err = ctypes.create_string_buffer(_shim.ERR_SIZE)
        self._p = tuple(ctypes.addressof(b) for b in
                        (self._schema, self._array, self._array2, self._view, self._err))
        self._cols = ctypes.create_string_buffer(_shim.COLDESC_SIZE * max(n, 1))
        # the typed row: one buffer per kind (never zero-length: numba types
        # a 0-d and a 1-d array differently, and the row must be one type)
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
        # the flat address table (data, validity, offset, view) per column
        self.addrs = np.zeros((max(n, 1), 4), dtype=np.uint64)
        self.plan_addr: int = ctypes.addressof(self._rowplan)
        self.gather_addr: int = GATHER_ADDR
        self.get_string_addr: int = GET_STRING_ADDR
        self.n: int = 0
        self.n_chunks: int = 0
        self._capsule = None
        self._extra: list[tuple[Any, Any]] = []   # (array buf, view buf) per extra chunk
        self._chunk = 0
        self.bound = False

    # --- bind / release -----------------------------------------------------

    def bind(self, df) -> "FrameView":
        """Import `df` (the whole frame, one capsule) and resolve every
        declared column. Raises `ArrowKindError` when a column's Arrow type
        does not match its declared kind. Polars rechunks the frame in
        place while exporting, so a multi-chunk frame arrives as ONE struct
        chunk (BOUNDARY-REWORK.md §1.4); the multi-chunk path below exists
        for a future polars that stops doing that."""
        if self.bound:
            self.release()
        plan = self.plan
        if tuple(df.columns) != plan.columns:
            raise ValueError(
                f"frame columns {list(df.columns)!r} differ from the plan's "
                f"{list(plan.columns)!r}; build a FramePlan for this frame's schema"
            )
        capsule = df.__arrow_c_stream__()
        stream = stream_pointer(capsule)
        s, a, a2, v, e = self._p
        rc = lib.sm_import_frame(stream, s, a, a2, v, e)
        if rc < 0:
            raise ArrowImportError(f"sm_import_frame rc={rc}: {lib.sm_error_message(e).decode()}")
        self._capsule = capsule
        self.bound = True
        self.n_chunks = 1
        if rc == 1:
            self._pull_remaining_chunks(stream)
        self._resolve(v)
        self.n = lib.sm_view_length(v)
        return self

    def _pull_remaining_chunks(self, stream: int) -> None:
        s, _, a2, _, e = self._p
        arrays = [self._array2]
        while True:
            buf = ctypes.create_string_buffer(_shim.ARRAY_SIZE)
            rc = lib.sm_stream_get_next(stream, ctypes.addressof(buf), e)
            if rc != 0:
                raise ArrowImportError(f"get_next rc={rc}: {lib.sm_error_message(e).decode()}")
            if not lib.sm_array_is_valid(ctypes.addressof(buf)):
                break
            arrays.append(buf)
        for arr in arrays:
            view = ctypes.create_string_buffer(_shim.VIEW_SIZE)
            rc = lib.sm_view_set(ctypes.addressof(view), s, ctypes.addressof(arr), e)
            if rc != 0:
                raise ArrowImportError(f"sm_view_set rc={rc}: {lib.sm_error_message(e).decode()}")
            self._extra.append((arr, view))
        self.n_chunks = 1 + len(self._extra)

    def _resolve(self, view_addr: int) -> None:
        plan = self.plan
        rc = lib.sm_resolve_all(
            view_addr, plan.kinds.ctypes.data, plan.slots.ctypes.data, plan.child_idx.ctypes.data,
            plan.fill_f64.ctypes.data, plan.fill_i64.ctypes.data, plan.ncols,
            self.plan_addr, self.addrs.ctypes.data,
        )
        if rc != 0:
            message = self._mismatch_message(-rc - 1)   # needs the bound schema
            self.release()
            raise ArrowKindError(message)

    def _mismatch_message(self, c: int) -> str:
        plan = self.plan
        name, kind = plan.names[c], FeatureKind(int(plan.kinds[c]))
        arrow_type = self.arrow_type(name)
        hint = ""
        if arrow_type.startswith("decimal"):
            hint = " Cast Decimal to a scaled integer in the frame tier before the boundary (doc 03 §1.2)."
        elif arrow_type.startswith(("list", "struct", "fixed_size_list", "large_list", "map")):
            hint = " Nested columns do not enter the kernel (KernelSplitPlan, boundary/dtypes.py)."
        elif arrow_type.startswith("dictionary") and kind is STR:
            hint = " A Categorical/Enum column reaches a STR feature through its dictionary (Stage 6); until then declare it CODE."
        elif arrow_type.startswith(("string", "large_string")) and kind is CODE:
            hint = " A String column is a STR feature; CODE is a dictionary index."
        return (
            f"column {name!r} is Arrow {arrow_type} but is declared {kind.name}, which cannot "
            f"read it.{hint}"
        )

    def release(self) -> None:
        if not self.bound:
            return
        s, a, a2, v, e = self._p
        for arr, view in self._extra:
            lib.sm_view_reset(ctypes.addressof(view))
            lib.sm_array_release(ctypes.addressof(arr))
        self._extra = []
        lib.sm_view_reset(v)
        lib.sm_array_release(a)
        lib.sm_array_release(a2)
        lib.sm_schema_release(s)
        self._capsule = None
        self.bound = False
        self.n = 0
        self.n_chunks = 0

    def __enter__(self) -> "FrameView":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.release()
        except Exception:
            pass

    # --- introspection (plan time / diagnostics / tests) --------------------

    def arrow_type(self, name: str) -> str:
        """nanoarrow's rendering of the column's Arrow type ("string_view",
        "dictionary(uint32)<string_view>", "decimal128(38, 2)", ...)."""
        self._require_bound()
        k = self.plan.columns.index(name)
        buf = ctypes.create_string_buffer(256)
        lib.sm_schema_child_to_string(self._p[0], k, ctypes.addressof(buf), 256)
        return buf.value.decode()

    def child_view(self, name: str) -> int:
        """Address of the column's `ArrowArrayView` (a kernel argument for
        `call_get_string`)."""
        self._require_bound()
        k = self.plan.columns.index(name)
        return lib.sm_view_child(self._p[3], k)

    def describe(self) -> list[dict]:
        """One record per declared column: name, Arrow type, storage type,
        kind, slot, offset, null count, whether it has a validity bitmap,
        dictionary length (Categorical/Enum) and variadic buffer count."""
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

    def validate(self, level: int = _shim.VALIDATE_FULL) -> None:
        """nanoarrow's own validation of the bound frame (diagnostics; the
        request path never validates)."""
        self._require_bound()
        rc = lib.sm_view_validate(self._p[3], level, self._p[4])
        if rc != 0:
            raise ArrowImportError(
                f"validate(level={level}) rc={rc}: {lib.sm_error_message(self._p[4]).decode()}"
            )

    # --- row access (tests, single-record paths) ----------------------------

    def gather(self, i: int) -> None:
        """Gather row `i` into the row buffers (one C call)."""
        self._require_bound()
        if not 0 <= i < self.n:
            raise IndexError(f"row {i} of {self.n}")
        lib.sm_gather_row(self.plan_addr, i)

    def string(self, slot: int) -> bytes | None:
        """The bytes of STR slot `slot` of the last gathered row (a copy),
        or None for a null."""
        addr, ln = int(self.span[2 * slot]), int(self.span[2 * slot + 1])
        if ln < 0:
            return None
        return ctypes.string_at(addr, ln)

    def materialize(self) -> "Rows":
        """Every row through the C gather via the cached kernel: 2-D arrays
        (n × slots) per kind, the span table and the validity table."""
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

    def strings(self, slot: int) -> list[bytes | None]:
        """Every row's bytes for STR slot `slot` (copies), None for nulls."""
        rows = self.materialize()
        return rows.strings(slot)

    def _require_bound(self) -> None:
        if not self.bound:
            raise RuntimeError("FrameView is not bound to a frame; call bind(df) first")


@dataclass
class Rows:
    f64: np.ndarray
    i64: np.ndarray
    b8: np.ndarray
    i32: np.ndarray
    span: np.ndarray
    valid: np.ndarray

    def strings(self, slot: int) -> list[bytes | None]:
        col = self.span[:, 2 * slot:2 * slot + 2]
        return [None if ln < 0 else ctypes.string_at(int(a), int(ln)) for a, ln in col.tolist()]


@njit(cache=True)
def materialize_rows(gather_addr, plan_addr, n, f64, i64, b8, i32, span, valid,
                     out_f64, out_i64, out_b8, out_i32, out_span, out_valid):
    """Gather every row through the shim (one pointer call per row) and copy
    the typed row out. `cache=True`: the two addresses are ARGUMENTS, so this
    specialisation is a genuine disk-cache hit in a fresh process."""
    for i in range(n):
        call_gather(gather_addr, plan_addr, i)
        out_f64[i, :] = f64
        out_i64[i, :] = i64
        out_b8[i, :] = b8
        out_i32[i, :] = i32
        out_span[i, :] = span
        out_valid[i, :] = valid


@njit(cache=True)
def string_lengths(get_string_addr, view_addr, n, out):
    """Every row's byte length through `sm_get_string` (-1 for null): the
    lazy STR-node shape, one accessor call per row, also cached."""
    for i in range(n):
        _, ln = _shim.call_get_string(get_string_addr, view_addr, i)
        out[i] = ln
