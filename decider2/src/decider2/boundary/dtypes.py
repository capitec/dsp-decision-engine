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

**A pydantic discriminated union, one class per row of the table.** Each
`EntryMode` used to be a tag two other modules (`extract.py`'s
`_extract_by_plan`, this module's own `_probe_extract`) switched on with an
`if`/`elif` chain. Now a `ColumnPlan` is one of five variant classes —
`ZeroCopyPlan`, `CopyPlan`, `CodesPlan`, `ScaledInt64Plan`, `KernelSplitPlan`
— discriminated on `entry_mode`, and each one owns `.extract()`: how *that*
variant gets its own column out of polars. Adding a sixth row to the ladder
means writing one new class here and adding it to the `ColumnPlan` union —
nothing downstream has a switch left to update.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Mapping, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DtypeTier",
    "EntryMode",
    "ColumnPlan",
    "ZeroCopyPlan",
    "CopyPlan",
    "CodesPlan",
    "ScaledInt64Plan",
    "KernelSplitPlan",
    "NeedsKernelSplit",
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
    """How a column actually reaches the compiled kernel. Doubles as the
    discriminator tag on `ColumnPlan`'s union — each member names exactly
    the variant class that knows how to act on it (`CopyPlan` claims three
    of them at once: `COPY_VALIDITY`/`BUFFER_COPY`/`AS_INTEGER` all extract
    identically — "get the values buffer as a numpy array" — and differ only
    in *why* a copy was unavoidable, which is what `note` is for)."""

    NATIVE = "native"                 # zero-copy, no conversion at all
    COPY_VALIDITY = "copy_validity"   # nullable numeric: copy + a boolean mask
    BUFFER_COPY = "buffer_copy"       # Boolean: arrow bitpacks it, never zero-copy
    AS_INTEGER = "as_integer"         # Date/Datetime/Duration as their physical int
    CODES = "codes"                   # Categorical/Enum/Utf8 as integer codes
    SCALED_INT64 = "scaled_int64"     # Decimal -> money-scaled int64 cents
    KERNEL_SPLIT = "kernel_split"     # no flat-array form here; escape one layer up


class NeedsKernelSplit(Exception):
    """Raised by a `KernelSplitPlan` — and by `ScaledInt64Plan` when a real
    Decimal conversion fails — when a dtype has no flat-array representation
    here (doc 05 §1.5: `List`, `Struct`, and anything else not yet on the
    ladder). **Not a rejection** — the ladder rejects nothing — this is the
    boundary handing the column to the escape mechanism one layer up:
    `compile/numba/fallback.py` splits the *kernel* around the column, never
    a per-row `objmode` escape (EXPERIMENTS.md §B measured that at 77x a pure
    kernel, worse than falling back to plain Python for the whole driver).
    """

    def __init__(self, name: str, dtype: pl.DataType, reason: str = ""):
        self.column = name
        self.dtype = dtype
        message = f"column '{name}' ({dtype}) has no flat-array extraction; needs the kernel-split escape"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


def _raw_values(series: pl.Series) -> pl.Series:
    """The values buffer, always — garbage under any null slot (doc 05 §2),
    never returned to a caller that hasn't already handled validity.
    """
    return series._get_buffers()["values"]


def _extract_numeric_like(series: pl.Series) -> tuple[np.ndarray, bool]:
    """Zero-copy when polars allows it, an ordinary copy otherwise — tried
    directly rather than assumed from the dtype/nullability table, because
    which combinations are actually zero-copy is a fact about the installed
    polars/numpy build (EXPERIMENTS.md §A pins it to this environment).
    Shared by `ZeroCopyPlan` and `CopyPlan`: "get the values buffer as a
    numpy array" is the same operation either way; they differ only in
    whether that succeeds without a copy.
    """
    values = _raw_values(series)
    try:
        return values.to_numpy(allow_copy=False), True
    except RuntimeError:
        return values.to_numpy(), False


def _extract_codes(series: pl.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    """Categorical/Enum/Utf8 all enter as integer codes (doc 05 §1.5) — a
    string never enters a kernel as a string. Categorical/Enum already store
    codes; Utf8 is dictionary-encoded first (`cast(pl.Categorical)`), then
    the same code extraction applies.

    Returns `(codes, categories)` — the category list is the dictionary a
    code indexes into (doc 05 §1.5 "Strings in detail", EXPERIMENTS.md §O):
    a `str`-typed param's literal is resolved against it at param-resolution
    time, rather than discarded here as it used to be. `CodesPlan.extract`
    is the only caller, and it returns the pair TOGETHER always: a codes
    array without its matching category list is meaningless (and dangerous
    — see the module docstring).
    """
    base = series.dtype.base_type()
    if base in (pl.Categorical, pl.Enum):
        codes = _raw_values(series).to_numpy(allow_copy=False)
        categories = tuple(series.cat.get_categories().to_list())
        return codes, categories
    encoded = series.cast(pl.Categorical)
    codes = _raw_values(encoded).to_numpy(allow_copy=False)
    categories = tuple(encoded.cat.get_categories().to_list())
    return codes, categories


def _decimal_as_cents(series: pl.Series, *, money_scale: int = 2) -> np.ndarray:
    """Doc 03 §1.2: `Decimal` converts to a money-scaled int64 at the
    boundary — never crosses as `Decimal` itself.

    Must run under `except BaseException`: `_get_buffers()` on a `Decimal`
    column succeeds and hands back a usable `Int128` series (doc 00-BUILD.md
    §O11) — the panic fires one call deeper, on `.to_numpy()`/an unchecked
    `.cast()` of *that* buffer, as `pyo3_runtime.PanicException`, which does
    not inherit `Exception` (EXPERIMENTS.md §A). The catch itself lives in
    `ScaledInt64Plan.extract`, the one caller — this is the pure conversion.
    """
    dtype = series.dtype
    if not isinstance(dtype, pl.Decimal):
        raise TypeError(f"_decimal_as_cents expects a Decimal column, got {dtype}")

    raw = _raw_values(series)  # Int128 series — the unscaled mantissa at `dtype.scale`
    scale = dtype.scale if dtype.scale is not None else 0
    cents = raw.cast(pl.Int64, strict=True)
    shift = money_scale - scale
    if shift > 0:
        cents = cents * (10**shift)
    elif shift < 0:
        cents = cents // (10 ** (-shift))
    return cents.to_numpy(allow_copy=False)


class _PlanBase(BaseModel):
    """One column's spot on the ladder (doc 05 §1.5's table, row by row) —
    shared shape; each subclass adds only its own `entry_mode` tag and
    `.extract()`.

    `nullable` is supplied by the caller, never inferred from `dtype` alone:
    a clean and a null-bearing column of the same dtype produce `==`-equal
    `Schema` objects — polars has no `nullable` flag on a `DataType` at all
    (doc 00-BUILD.md §O11) — so nullability is a governance fact the input
    schema hand-authors, the same way the params/structure boundary is.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    dtype: pl.DataType
    tier: DtypeTier
    nullable: bool
    note: str = ""

    def guard(self) -> None:
        """Run before any null-policy-specific handling (`extract.py`'s
        `extract_column`, before it even looks at the column's
        `NullPolicy`): a no-op for every variant except `KernelSplitPlan`,
        which raises here instead of letting a null-tier-specific path
        (`fill_column`, say) try to make sense of a dtype it can't touch.
        """
        return None

    def extract(self, series: pl.Series) -> tuple[np.ndarray, tuple[str, ...] | None]:
        """Values, plus the category list for a `CodesPlan` (`None`
        otherwise). Every variant implements this; nothing outside this
        module switches on `entry_mode` to decide how to call it.
        """
        raise NotImplementedError


class ZeroCopyPlan(_PlanBase):
    """Clean Float64/Int64/Int32/UInt8/Datetime/Duration — one of the six
    zero-copy dtype/nullability combinations (doc 05 §1.2)."""

    entry_mode: Literal[EntryMode.NATIVE] = EntryMode.NATIVE

    def extract(self, series: pl.Series) -> tuple[np.ndarray, None]:
        values, _zero_copy = _extract_numeric_like(series)
        return values, None


class CopyPlan(_PlanBase):
    """Nullable numeric, Boolean, or Date/Datetime/Duration: three different
    reasons a column can't be zero-copy, one identical extraction — "get the
    values buffer as a numpy array" (`_extract_numeric_like` tries zero-copy
    first regardless; these three just never win that bet in practice)."""

    entry_mode: Literal[EntryMode.COPY_VALIDITY, EntryMode.BUFFER_COPY, EntryMode.AS_INTEGER]

    def extract(self, series: pl.Series) -> tuple[np.ndarray, None]:
        values, _zero_copy = _extract_numeric_like(series)
        return values, None


class CodesPlan(_PlanBase):
    """Categorical/Enum/Utf8 as integer codes (doc 05 §1.5). `tier` varies
    by source dtype (Categorical/Enum are already `DtypeTier.COPY`; Utf8 is
    `DtypeTier.CONVERT` — it has to be dictionary-encoded first), which is
    why `tier` is a real field here rather than a fixed literal."""

    entry_mode: Literal[EntryMode.CODES] = EntryMode.CODES

    def extract(self, series: pl.Series) -> tuple[np.ndarray, tuple[str, ...]]:
        return _extract_codes(series)


class ScaledInt64Plan(_PlanBase):
    """Decimal -> money-scaled int64 cents (doc 03 §1.2) — the money answer
    anyway. The one variant whose `.extract()` can still fail on real data
    even though `plan_column` already predicted it (a Rust-panic overflow on
    `.cast(Int64)`), so it owns the `except BaseException` that downgrades
    that failure into the kernel-split escape, exactly as `probe_column`
    would have predicted had it run first."""

    entry_mode: Literal[EntryMode.SCALED_INT64] = EntryMode.SCALED_INT64

    def extract(self, series: pl.Series) -> tuple[np.ndarray, None]:
        try:
            return _decimal_as_cents(series), None
        except BaseException as exc:  # noqa: BLE001 — doc 05 §1.5's hard requirement
            raise NeedsKernelSplit(
                self.name, self.dtype,
                reason=f"Decimal->int64 cents conversion failed: {exc!r}",
            ) from exc


class KernelSplitPlan(_PlanBase):
    """No flat-array form here yet (`List`, `Struct`, anything future) —
    reported for the kernel-split escape rather than attempted and failed
    loudly (doc 05 §1.5: the ladder rejects nothing)."""

    entry_mode: Literal[EntryMode.KERNEL_SPLIT] = EntryMode.KERNEL_SPLIT

    def guard(self) -> None:
        raise NeedsKernelSplit(self.name, self.dtype, reason=self.note)

    def extract(self, series: pl.Series) -> tuple[np.ndarray, None]:
        raise NeedsKernelSplit(self.name, self.dtype, reason=self.note)


ColumnPlan = Annotated[
    Union[ZeroCopyPlan, CopyPlan, CodesPlan, ScaledInt64Plan, KernelSplitPlan],
    Field(discriminator="entry_mode"),
]


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
    and downgrades to `KernelSplitPlan` if that attempt fails.
    """
    base = dtype.base_type()

    if base in _CLEAN_ZERO_COPY_NUMERIC:
        if nullable:
            return CopyPlan(
                name=name, dtype=dtype, tier=DtypeTier.COPY, entry_mode=EntryMode.COPY_VALIDITY,
                nullable=nullable,
                note="nullable numeric — copies at extraction, ~550-630us/100k rows (doc 05 §1.2)",
            )
        return ZeroCopyPlan(
            name=name, dtype=dtype, tier=DtypeTier.ZERO_COPY, nullable=nullable,
            note="clean — one of the six zero-copy dtype/nullability combinations (doc 05 §1.2)",
        )

    if base is pl.Boolean:
        return CopyPlan(
            name=name, dtype=dtype, tier=DtypeTier.COPY, entry_mode=EntryMode.BUFFER_COPY,
            nullable=nullable,
            note="arrow bitpacks Boolean; allow_copy=False always fails (doc 05 §1.4)",
        )

    if base in _TEMPORAL:
        return CopyPlan(
            name=name, dtype=dtype, tier=DtypeTier.COPY, entry_mode=EntryMode.AS_INTEGER,
            nullable=nullable,
            note="enters as its physical int; datetime+datetime does not compile (doc 05 §1.5)",
        )

    if base in _CATEGORY_LIKE:
        return CodesPlan(
            name=name, dtype=dtype, tier=DtypeTier.COPY, nullable=nullable,
            note="enters as codes; cross-frame code stability must be declared (doc 05 §1.5)",
        )

    if base is pl.String:  # pl.Utf8 is pl.String in this polars version
        return CodesPlan(
            name=name, dtype=dtype, tier=DtypeTier.CONVERT, nullable=nullable,
            note="dictionary-encoded to codes — a string never enters a kernel as a string (doc 05 §1.5)",
        )

    if isinstance(dtype, pl.Decimal) or base is pl.Decimal:
        return ScaledInt64Plan(
            name=name, dtype=dtype, tier=DtypeTier.CONVERT, nullable=nullable,
            note="converted to scaled int64 cents — the money answer anyway (doc 03 §1.2)",
        )

    # Everything else (List, Struct, Array, Object, Binary, and anything
    # future) is not rejected either — there is simply no flat-array
    # conversion for it here yet. Reported for the kernel-split escape
    # (compile/numba/fallback.py splits the KERNEL around the offending
    # column; a per-row `objmode` escape is 77x a pure kernel and is refused
    # — EXPERIMENTS.md §B) rather than attempted and failed loudly.
    return KernelSplitPlan(
        name=name, dtype=dtype, tier=DtypeTier.CONVERT, nullable=nullable,
        note=f"{dtype} has no flat-array conversion yet; the kernel splits around it (doc 05 §1.5)",
    )


def _probe_extract(series: pl.Series, plan: ColumnPlan) -> None:
    """Actually attempt the conversion `plan` predicts, discarding the
    result. This is the call that must survive under `except BaseException`
    in `probe_column` — `Decimal`'s `.cast(Int64)` is the one that can panic.

    Delegates straight to `plan.extract()`: probing and extracting run the
    identical conversion (previously an `if`/`elif` over `plan.entry_mode`,
    duplicating `_extract_by_plan`'s own switch one module over) — only
    whether the result is kept differs, and that's the one line below.
    """
    plan.extract(series)


def probe_column(series: pl.Series, *, nullable: bool | None = None) -> ColumnPlan:
    """`plan_column`, verified against real data — the actual admissibility
    gate doc 00-BUILD.md's build order means by "`dtypes.py` (admissibility
    gate)". Downgrades to a `KernelSplitPlan` if the predicted conversion
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
        return KernelSplitPlan(
            name=plan.name, dtype=plan.dtype, tier=DtypeTier.CONVERT, nullable=plan.nullable,
            note=f"planned {plan.entry_mode.value} failed to probe ({exc!r}); kernel splits around it",
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
