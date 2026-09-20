"""The dtype ladder: classify every incoming column, reject nothing.

Doc 05 §1.5 — **a ladder, not a gate.** An earlier draft called this "the
admissible-dtype contract" and marked `Decimal`/`List` inadmissible; that was
withdrawn as the wrong default (doc 00-BUILD.md §2b: "flexible by default,
tighten for speed"). Real credit/fraud inputs are a mix of flat and deeply
nested, mixed types, and everything should work — some things just cost more.
Tightening a dtype is a performance choice an author makes deliberately, never
a precondition for using the library.

Three tiers (doc 05 §1.5, EXPERIMENTS.md §A):

    1. zero-copy       — clean Float64/Int64/Int32/UInt8/Datetime/Duration
    2. copies, compiled — nullable numeric, Boolean, Date/Datetime/Duration
                          as integers, Categorical/Enum as codes
    3. converted        — Utf8 (codes), Decimal (scaled int64 cents); anything
                          else (List, Struct, ...) has no flat-array form yet
                          and is reported for the kernel-split escape instead
                          (compile/numba/fallback.py, never a per-row
                          `objmode` — EXPERIMENTS.md §B)

This is **not** the same "tier" word `decider2.types.NullPolicy` uses. That
one is about which *values* are missing; this one is about which *dtypes* a
compiled kernel can hold. A column has one of each, independently.

Two probing rules survive from the stricter draft because they are
correctness, not performance (doc 05 §1.5, EXPERIMENTS.md §A, doc 00-BUILD.md
§O11):

    * probe with `except BaseException`, never `except Exception` — `Decimal`
      fails as `pyo3_runtime.PanicException`, a Rust panic that inherits
      `BaseException` directly, not `Exception`;
    * the panic does not fire on `_get_buffers()` — that call succeeds and
      hands back a usable `Int128` series. It fires one call deeper, on
      `.to_numpy()`/an unchecked `.cast()` of that buffer. So the gate belongs
      at the conversion, not at extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import polars as pl

__all__ = [
    "DtypeTier",
    "EntryMode",
    "ColumnPlan",
    "plan_column",
    "probe_column",
    "explain_boundary",
]


class DtypeTier(Enum):
    """Doc 05 §1.5's three tiers — a cost story, not a rejection list."""

    ZERO_COPY = 1
    COPY = 2
    CONVERT = 3


class EntryMode(Enum):
    """How a column actually reaches the compiled kernel."""

    NATIVE = "native"                 # zero-copy, no conversion at all
    COPY_VALIDITY = "copy_validity"   # nullable numeric: copy + a boolean mask
    BUFFER_COPY = "buffer_copy"       # Boolean: arrow bitpacks it, never zero-copy
    AS_INTEGER = "as_integer"         # Date/Datetime/Duration as their physical int
    CODES = "codes"                   # Categorical/Enum/Utf8 as integer codes
    SCALED_INT64 = "scaled_int64"     # Decimal -> money-scaled int64 cents
    KERNEL_SPLIT = "kernel_split"     # no flat-array form here; escape one layer up


@dataclass(frozen=True)
class ColumnPlan:
    """One column's spot on the ladder (doc 05 §1.5's table, row by row).

    `nullable` is supplied by the caller, never inferred from `dtype` alone:
    a clean and a null-bearing column of the same dtype produce `==`-equal
    `Schema` objects — polars has no `nullable` flag on a `DataType` at all
    (doc 00-BUILD.md §O11) — so nullability is a governance fact the input
    schema hand-authors, the same way the params/structure boundary is.
    """

    name: str
    dtype: pl.DataType
    tier: DtypeTier
    entry_mode: EntryMode
    nullable: bool
    note: str = ""


# The four tier-1/2 base types that are zero-copy when clean (doc 05 §1.2:
# "Only 6 of 26 dtype/nullability combinations are zero-copy — clean Float64,
# Int64, Int32, UInt8, Datetime and Duration").
_CLEAN_ZERO_COPY_NUMERIC: tuple[type, ...] = (pl.Float64, pl.Int64, pl.Int32, pl.UInt8)
_TEMPORAL: tuple[type, ...] = (pl.Date, pl.Datetime, pl.Duration)
_CATEGORY_LIKE: tuple[type, ...] = (pl.Categorical, pl.Enum)


def plan_column(name: str, dtype: pl.DataType, *, nullable: bool) -> ColumnPlan:
    """Classify one column by dtype and declared nullability alone — no data
    is touched. Never raises: the ladder rejects nothing (§1.5).

    This is the declarative half. `probe_column` is the other half: it
    actually attempts the conversion this function predicts, on real data,
    and downgrades to `KERNEL_SPLIT` if that attempt fails.
    """
    base = dtype.base_type()

    if base in _CLEAN_ZERO_COPY_NUMERIC:
        if nullable:
            return ColumnPlan(
                name, dtype, DtypeTier.COPY, EntryMode.COPY_VALIDITY, nullable,
                "nullable numeric — copies at extraction, ~550-630us/100k rows (doc 05 §1.2)",
            )
        return ColumnPlan(
            name, dtype, DtypeTier.ZERO_COPY, EntryMode.NATIVE, nullable,
            "clean — one of the six zero-copy dtype/nullability combinations (doc 05 §1.2)",
        )

    if base is pl.Boolean:
        return ColumnPlan(
            name, dtype, DtypeTier.COPY, EntryMode.BUFFER_COPY, nullable,
            "arrow bitpacks Boolean; allow_copy=False always fails (doc 05 §1.4)",
        )

    if base in _TEMPORAL:
        return ColumnPlan(
            name, dtype, DtypeTier.COPY, EntryMode.AS_INTEGER, nullable,
            "enters as its physical int; datetime+datetime does not compile (doc 05 §1.5)",
        )

    if base in _CATEGORY_LIKE:
        return ColumnPlan(
            name, dtype, DtypeTier.COPY, EntryMode.CODES, nullable,
            "enters as codes; cross-frame code stability must be declared (doc 05 §1.5)",
        )

    if base is pl.String:  # pl.Utf8 is pl.String in this polars version
        return ColumnPlan(
            name, dtype, DtypeTier.CONVERT, EntryMode.CODES, nullable,
            "dictionary-encoded to codes — a string never enters a kernel as a string (doc 05 §1.5)",
        )

    if isinstance(dtype, pl.Decimal) or base is pl.Decimal:
        return ColumnPlan(
            name, dtype, DtypeTier.CONVERT, EntryMode.SCALED_INT64, nullable,
            "converted to scaled int64 cents — the money answer anyway (doc 03 §1.2)",
        )

    # Everything else (List, Struct, Array, Object, Binary, and anything
    # future) is not rejected either — there is simply no flat-array
    # conversion for it here yet. Reported for the kernel-split escape
    # (compile/numba/fallback.py splits the KERNEL around the offending
    # column; a per-row `objmode` escape is 77x a pure kernel and is refused
    # — EXPERIMENTS.md §B) rather than attempted and failed loudly.
    return ColumnPlan(
        name, dtype, DtypeTier.CONVERT, EntryMode.KERNEL_SPLIT, nullable,
        f"{dtype} has no flat-array conversion yet; the kernel splits around it (doc 05 §1.5)",
    )


def _probe_extract(series: pl.Series, plan: ColumnPlan) -> None:
    """Actually attempt the conversion `plan` predicts, discarding the
    result. This is the call that must survive under `except BaseException`
    in `probe_column` — `Decimal`'s `.cast(Int64)` is the one that can panic.
    """
    bufs = series._get_buffers()
    values = bufs["values"]

    if plan.entry_mode in (EntryMode.COPY_VALIDITY, EntryMode.BUFFER_COPY, EntryMode.AS_INTEGER):
        values.to_numpy()
        return
    if plan.entry_mode is EntryMode.CODES:
        if plan.dtype.base_type() in _CATEGORY_LIKE:
            values.to_numpy(allow_copy=False)
        else:
            series.cast(pl.Categorical)._get_buffers()["values"].to_numpy(allow_copy=False)
        return
    if plan.entry_mode is EntryMode.SCALED_INT64:
        values.cast(pl.Int64, strict=True)
        return
    # NATIVE and KERNEL_SPLIT never reach here — probe_column short-circuits
    # both (nothing to probe; already known to need the escape).


def probe_column(series: pl.Series, *, nullable: bool | None = None) -> ColumnPlan:
    """`plan_column`, verified against real data — the actual admissibility
    gate doc 00-BUILD.md's build order means by "`dtypes.py` (admissibility
    gate)". Downgrades to `EntryMode.KERNEL_SPLIT` if the predicted conversion
    fails on this column's real values.

    Must catch `BaseException`: `Decimal`'s failure is
    `pyo3_runtime.PanicException`, which does **not** inherit `Exception`
    (EXPERIMENTS.md §A) — a narrower `except Exception` here lets the panic
    escape and takes down the whole extraction, rather than reporting one
    column's real tier.
    """
    if nullable is None:
        nullable = series.null_count() > 0
    plan = plan_column(series.name, series.dtype, nullable=nullable)

    if plan.entry_mode in (EntryMode.NATIVE, EntryMode.KERNEL_SPLIT):
        return plan  # zero-copy needs no probe; already-split has nothing to try

    try:
        _probe_extract(series, plan)
    except BaseException as exc:  # noqa: BLE001 — doc 05 §1.5's hard requirement
        return ColumnPlan(
            plan.name, plan.dtype, DtypeTier.CONVERT, EntryMode.KERNEL_SPLIT, plan.nullable,
            f"planned {plan.entry_mode.value} failed to probe ({exc!r}); kernel splits around it",
        )
    return plan


def explain_boundary(
    frame: pl.DataFrame,
    *,
    nullable: Mapping[str, bool] | None = None,
    probe: bool = True,
) -> list[ColumnPlan]:
    """One row per column: which tier it lands on and why (doc 05 §1.5:
    "what the framework owes the author, since nothing is rejected"). Reading
    this answers "why is my batch slow" from a table instead of a guess.

    `nullable` overrides the declared-schema nullability per column (doc
    00-BUILD.md §O11: it cannot be read off the polars dtype). Columns not
    named there fall back to this frame's own `null_count()` — a convenience
    for ad-hoc inspection, not a substitute for the hand-authored schema a
    real pipeline runs against.

    `probe=True` (the default) actually attempts each conversion (§ above);
    `probe=False` gives the cheaper, purely declarative `plan_column` view.
    """
    nullable = nullable or {}
    plans = []
    for name in frame.columns:
        series = frame[name]
        is_nullable = nullable.get(name, series.null_count() > 0)
        if probe:
            plans.append(probe_column(series, nullable=is_nullable))
        else:
            plans.append(plan_column(name, series.dtype, nullable=is_nullable))
    return plans
