from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from decider.engine.ir.decls import FeatureKind
from decider.exceptions import ArrowImportError, ArrowKindError  # noqa: F401  (re-exported)

F64, I64, BOOL, CODE, STR = (
    FeatureKind.F64, FeatureKind.I64, FeatureKind.BOOL, FeatureKind.CODE, FeatureKind.STR,
)


@dataclass(frozen=True)
class ColumnSpec:
    """One declared column: its name, kind, and the value a null becomes.

    `fill=None` means the kind's default: NaN, 0, False, -1 (CODE) or a null span (STR).
    """

    name: str
    kind: FeatureKind
    fill: Any = None


class FramePlan:
    """Everything about an import that depends only on the declared columns and the frame's column order.

    Slot `j` of kind `K` is the `j`-th input of kind `K`, in declaration order.

    Example::

        plan = FramePlan([ColumnSpec("income", FeatureKind.F64, fill=0.0)], df.columns)
    """

    __slots__ = ("specs", "columns", "ncols", "names", "kinds", "slots", "child_idx",
                 "fill_f64", "fill_i64", "counts", "places", "addresses", "row_bytes")

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
        self.slots = np.zeros(n, dtype=np.int32)
        # Never zero-length, so the address handed to C is a real allocation.
        self.fill_f64 = np.full(max(n, 1), np.nan, dtype=np.float64)
        self.fill_i64 = np.zeros(max(n, 1), dtype=np.int64)
        for c, s in enumerate(specs):
            self.slots[c] = counts[s.kind]
            counts[s.kind] += 1
            if s.kind is CODE:
                self.fill_i64[c] = -1
            if s.fill is None:
                continue
            if s.kind is F64:
                self.fill_f64[c] = float(s.fill)
            elif s.kind is STR:
                raise ValueError(f"{s.name}: a STR feature has no fill; a null is length -1")
            else:
                self.fill_i64[c] = int(s.fill)
        self.counts = tuple(counts)
        # The bytes a row takes in `FrameView.columns()`' buffer: values, then validity.
        self.row_bytes = 8 * counts[F64] + 8 * counts[I64] + 16 * counts[STR] + 4 * counts[CODE] + counts[BOOL] + n
        self.places = tuple((int(s.kind), int(self.slots[c])) for c, s in enumerate(specs))
        # Taken once: `.ctypes.data` costs about a microsecond a call.
        self.addresses = tuple(a.ctypes.data for a in (self.kinds, self.slots, self.child_idx,
                                                      self.fill_f64, self.fill_i64))

    def __repr__(self) -> str:
        return f"FramePlan({list(zip(self.names, [FeatureKind(k).name for k in self.kinds]))})"
