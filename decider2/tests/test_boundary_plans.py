"""Tests for the pydantic discriminated-union / class-execution refactor of
`decider2.boundary`'s two dispatch axes:

    * `EntryMode` -> which `ColumnPlan` variant a column gets
      (`ZeroCopyPlan`, `CopyPlan`, `CodesPlan`, `ScaledInt64Plan`,
      `KernelSplitPlan`), each owning its own `.extract()`;
    * `NullPolicy` -> which `NullTierStrategy` a column's null handling
      goes through (`NULL_TIER_STRATEGIES`, keyed on the fixed
      `decider2.types.NullPolicy` enum).

Behaviour must be identical to the pre-refactor `if`/`elif` chains this
replaces — `test_boundary_dtypes.py`/`test_boundary_extract.py`/
`test_boundary_nulls.py` already pin the observable behaviour end to end;
this file is about the new seam itself: that each variant/strategy is the
one place its own behaviour lives, that the union actually discriminates,
and the two things the task brief calls load-bearing — CodesPlan's
(codes, categories) pair never drifting apart, and MISSING_AS/
NOT_APPLICABLE_AS staying distinct reason codes — plus the zero-copy
guarantee surviving the refactor at all.
"""
from datetime import date
from decimal import Decimal

import numpy as np
import polars as pl
import pytest
from pydantic import TypeAdapter

from decider2.boundary.dtypes import (
    ColumnPlan,
    CodesPlan,
    CopyPlan,
    DtypeTier,
    EntryMode,
    KernelSplitPlan,
    NeedsKernelSplit,
    ScaledInt64Plan,
    ZeroCopyPlan,
    plan_column,
)
from decider2.boundary.extract import extract_column, extract_frame
from decider2.boundary.nulls import (
    NULL_TIER_STRATEGIES,
    FillReason,
    NullTierStrategy,
    fill_column,
)
from decider2.types import Input, NullPolicy


# ---------------------------------------------------------------------------
# The union actually discriminates (pydantic, not just five same-shaped
# classes with a shared field).
# ---------------------------------------------------------------------------

def test_column_plan_is_a_working_discriminated_union():
    """Round-tripping an already-built plan through a `TypeAdapter` for the
    `ColumnPlan` union must land back on the exact same concrete class —
    confirming `entry_mode` really is wired as pydantic's discriminator,
    not just a same-named field five unrelated classes happen to share."""
    plan = plan_column("x", pl.Float64(), nullable=False)
    revalidated = TypeAdapter(ColumnPlan).validate_python(plan)
    assert type(revalidated) is ZeroCopyPlan


def test_each_entry_mode_maps_to_exactly_one_variant_class():
    assert isinstance(plan_column("x", pl.Float64(), nullable=False), ZeroCopyPlan)
    assert isinstance(plan_column("x", pl.Float64(), nullable=True), CopyPlan)
    assert isinstance(plan_column("b", pl.Boolean(), nullable=False), CopyPlan)
    assert isinstance(plan_column("t", pl.Datetime(), nullable=False), CopyPlan)
    assert isinstance(plan_column("c", pl.Categorical(), nullable=False), CodesPlan)
    assert isinstance(plan_column("s", pl.String(), nullable=False), CodesPlan)
    assert isinstance(plan_column("m", pl.Decimal(18, 2), nullable=False), ScaledInt64Plan)
    assert isinstance(plan_column("l", pl.List(pl.Float64()), nullable=False), KernelSplitPlan)


# ---------------------------------------------------------------------------
# Each plan variant extracts its own column correctly (`.extract()`, no
# switch at the call site — `_extract_by_plan` is now a one-line delegate).
# ---------------------------------------------------------------------------

def test_zero_copy_plan_extracts_a_clean_numeric_column():
    s = pl.Series("x", [1.0, 2.0, 3.0])
    plan = plan_column("x", s.dtype, nullable=False)
    values, categories = plan.extract(s)
    assert values.tolist() == [1.0, 2.0, 3.0]
    assert categories is None


def test_copy_plan_extracts_nullable_numeric_with_garbage_under_the_null():
    s = pl.Series("x", [1.0, None, 3.0])
    plan = plan_column("x", s.dtype, nullable=True)
    values, categories = plan.extract(s)
    assert categories is None
    assert values[0] == 1.0
    assert values[2] == 3.0
    # index 1 is unspecified garbage, not asserted here — validity is a
    # separate concern (NULL_TIER_STRATEGIES' OPTIONAL tier, tested below)


def test_copy_plan_extracts_boolean_via_buffer_copy():
    s = pl.Series("flag", [True, False, True])
    plan = plan_column("flag", s.dtype, nullable=False)
    assert plan.entry_mode is EntryMode.BUFFER_COPY
    values, categories = plan.extract(s)
    assert values.tolist() == [True, False, True]
    assert categories is None


def test_copy_plan_extracts_temporal_as_integer():
    s = pl.Series("d", [date(2024, 1, 1), date(2024, 1, 2)])
    plan = plan_column("d", s.dtype, nullable=False)
    assert plan.entry_mode is EntryMode.AS_INTEGER
    values, categories = plan.extract(s)
    assert categories is None
    assert values.dtype.kind in ("i", "u")


def test_codes_plan_extracts_categorical_codes_and_categories_together():
    s = pl.Series("segment", ["retail", "sme", "retail"]).cast(pl.Categorical)
    plan = plan_column("segment", s.dtype, nullable=False)
    values, categories = plan.extract(s)
    assert values[0] == values[2]
    assert values[0] != values[1]
    assert categories is not None
    assert categories[values[0]] == "retail"
    assert categories[values[1]] == "sme"


def test_codes_plan_extracts_utf8_via_dictionary_encoding():
    s = pl.Series("segment", ["retail", "sme", "retail"])
    plan = plan_column("segment", s.dtype, nullable=False)
    assert plan.tier is DtypeTier.CONVERT  # Utf8's own row of the ladder
    values, categories = plan.extract(s)
    assert values.dtype.kind in ("u", "i")
    assert categories is not None
    assert categories[values[0]] == "retail"
    assert categories[values[0]] == categories[values[2]]


def test_codes_plan_categories_never_drift_from_their_codes():
    """EXPERIMENTS.md §O: a `str` param's literal is resolved against this
    exact category list. Every code must be a valid index into it, and the
    text it names must round-trip through the original values."""
    raw = ["alpha", "beta", "alpha", "gamma", "beta"]
    s = pl.Series("x", raw)
    plan = plan_column("x", s.dtype, nullable=False)
    values, categories = plan.extract(s)
    assert [categories[c] for c in values] == raw


def test_scaled_int64_plan_extracts_decimal_as_cents():
    s = pl.Series("amount", [Decimal("19.99"), Decimal("5.00")], dtype=pl.Decimal(18, 2))
    plan = plan_column("amount", s.dtype, nullable=False)
    values, categories = plan.extract(s)
    assert categories is None
    assert values.tolist() == [1999, 500]
    assert values.dtype == np.int64


def test_kernel_split_plan_extract_always_raises():
    s = pl.Series("amounts", [[1.0, 2.0], [3.0]])
    plan = plan_column("amounts", s.dtype, nullable=False)
    with pytest.raises(NeedsKernelSplit) as exc_info:
        plan.extract(s)
    assert exc_info.value.column == "amounts"


def test_kernel_split_plan_guard_raises_before_any_null_handling():
    """`extract_column` calls `plan.guard()` unconditionally, before it even
    looks at the column's `NullPolicy` — a `KernelSplitPlan` column must
    fail the same way regardless of which null tier declared it."""
    s = pl.Series("amounts", [[1.0, 2.0], [3.0]])
    plan = plan_column("amounts", s.dtype, nullable=False)
    with pytest.raises(NeedsKernelSplit):
        plan.guard()


def test_every_other_variants_guard_is_a_no_op():
    for dtype in (pl.Float64(), pl.Boolean(), pl.Categorical(), pl.Decimal(18, 2)):
        plan = plan_column("x", dtype, nullable=False)
        assert plan.guard() is None


# ---------------------------------------------------------------------------
# Zero-copy survives the refactor.
# ---------------------------------------------------------------------------

def test_zero_copy_plan_still_shares_memory_with_its_polars_source():
    """The load-bearing performance property (doc 05 §1.2/§1.3): a clean
    zero-copy column's extracted array must share the same underlying
    buffer as polars' own — not a `.copy()` introduced by the refactor.
    `np.shares_memory` is the direct check; `OWNDATA`/`WRITEABLE` are the
    two flags that would otherwise let a silent copy slip through unnoticed.
    """
    s = pl.Series("x", np.arange(1000, dtype=np.float64))
    source = s._get_buffers()["values"].to_numpy(allow_copy=False)

    plan = plan_column("x", s.dtype, nullable=False)
    assert isinstance(plan, ZeroCopyPlan)
    values, _categories = plan.extract(s)

    assert np.shares_memory(values, source)
    assert values.flags["OWNDATA"] is False
    assert values.flags["WRITEABLE"] is False
    assert values.tolist() == source.tolist()


def test_extract_column_end_to_end_is_still_zero_copy_for_a_clean_required_column():
    s = pl.Series("net_income", [9200.0, 4100.0, 15000.0])
    source = s._get_buffers()["values"].to_numpy(allow_copy=False)
    col = extract_column(s)
    assert np.shares_memory(col.values, source)
    assert col.plan.entry_mode is EntryMode.NATIVE


# ---------------------------------------------------------------------------
# Each null tier behaves correctly (NULL_TIER_STRATEGIES, keyed on the
# fixed decider2.types.NullPolicy enum).
# ---------------------------------------------------------------------------

def test_null_tier_strategies_cover_all_four_members_of_the_fixed_enum():
    assert set(NULL_TIER_STRATEGIES) == set(NullPolicy)
    for strategy in NULL_TIER_STRATEGIES.values():
        assert isinstance(strategy, NullTierStrategy)


def test_only_required_routes_at_frame_level():
    for policy, strategy in NULL_TIER_STRATEGIES.items():
        expected = policy is NullPolicy.REQUIRED
        assert strategy.routes_at_frame_level is expected


def test_required_tier_extracts_through_when_clean():
    s = pl.Series("x", [1.0, 2.0])
    plan = plan_column("x", s.dtype, nullable=False)
    strategy = NULL_TIER_STRATEGIES[NullPolicy.REQUIRED]
    values, validity, fill, categories = strategy.extract_column(s, None, plan)
    assert values.tolist() == [1.0, 2.0]
    assert validity is None
    assert fill is None
    assert categories is None


def test_required_tier_refuses_a_column_with_remaining_nulls():
    s = pl.Series("x", [1.0, None])
    plan = plan_column("x", s.dtype, nullable=True)
    strategy = NULL_TIER_STRATEGIES[NullPolicy.REQUIRED]
    with pytest.raises(ValueError, match="route it with"):
        strategy.extract_column(s, None, plan)


def test_optional_tier_builds_a_validity_mask_and_never_fills():
    s = pl.Series("x", [1.0, None, 3.0])
    plan = plan_column("x", s.dtype, nullable=True)
    strategy = NULL_TIER_STRATEGIES[NullPolicy.OPTIONAL]
    decl = Input(name="x", annotation=float | None, null_policy=NullPolicy.OPTIONAL)
    values, validity, fill, categories = strategy.extract_column(s, decl, plan)
    assert validity.tolist() == [True, False, True]
    assert fill is None  # tier 3 is never filled — doc 03 §1


def test_fill_tiers_share_behaviour_and_differ_only_in_reason():
    """The load-bearing distinction the task brief calls out: MISSING_AS and
    NOT_APPLICABLE_AS must fill identically and tag DIFFERENT reason codes."""
    missing_decl = Input(name="x", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=-1.0)
    not_applicable_decl = Input(
        name="x", annotation=float, null_policy=NullPolicy.NOT_APPLICABLE_AS, fill=-1.0
    )
    s = pl.Series("x", [1.0, None, 3.0])
    plan = plan_column("x", s.dtype, nullable=True)

    missing_strategy = NULL_TIER_STRATEGIES[NullPolicy.MISSING_AS]
    na_strategy = NULL_TIER_STRATEGIES[NullPolicy.NOT_APPLICABLE_AS]

    m_values, m_validity, m_fill, _ = missing_strategy.extract_column(s, missing_decl, plan)
    n_values, n_validity, n_fill, _ = na_strategy.extract_column(s, not_applicable_decl, plan)

    # Identical filling:
    assert m_values.tolist() == n_values.tolist() == [1.0, -1.0, 3.0]
    assert m_validity is None and n_validity is None
    assert m_fill.filled_count == n_fill.filled_count == 1

    # DIFFERENT reason codes — this is the tier-2/tier-4 distinction:
    assert m_fill.reason is FillReason.MISSING
    assert n_fill.reason is FillReason.NOT_APPLICABLE
    assert m_fill.reason is not n_fill.reason


def test_the_two_fill_reasons_stay_distinct_through_fill_column_directly():
    s = pl.Series("x", [1.0, None])
    missing_decl = Input(name="x", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=0.0)
    na_decl = Input(name="x", annotation=float, null_policy=NullPolicy.NOT_APPLICABLE_AS, fill=0.0)
    _, missing_info = fill_column(s, missing_decl)
    _, na_info = fill_column(s, na_decl)
    assert missing_info.reason is FillReason.MISSING
    assert na_info.reason is FillReason.NOT_APPLICABLE


def test_the_two_fill_reasons_stay_distinct_when_the_column_is_absent():
    """`_synthesize_absent_column`'s path (review finding 4) — a declared
    input with no matching frame column at all must still tag the correct,
    distinct reason code for its tier, not silently collapse to one."""
    frame = pl.DataFrame({"id": [1, 2, 3]})
    missing_inputs = [
        Input(name="id", annotation=int, null_policy=NullPolicy.REQUIRED),
        Input(name="bureau_score", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=0.0),
    ]
    na_inputs = [
        Input(name="id", annotation=int, null_policy=NullPolicy.REQUIRED),
        Input(name="spouse_income", annotation=float, null_policy=NullPolicy.NOT_APPLICABLE_AS, fill=0.0),
    ]
    missing_result = extract_frame(frame, missing_inputs)
    na_result = extract_frame(frame, na_inputs)

    assert missing_result.columns["bureau_score"].fill.reason is FillReason.MISSING
    assert na_result.columns["spouse_income"].fill.reason is FillReason.NOT_APPLICABLE
    assert missing_result.columns["bureau_score"].fill.filled_count == 3
    assert na_result.columns["spouse_income"].fill.filled_count == 3


def test_synthesize_absent_matches_extract_column_shape_per_tier():
    """`synthesize_absent` (no real Series) and `extract_column` (a real,
    all-null Series) must agree on what each tier produces, since one
    stands in for the other when a declared input is missing entirely."""
    n = 3
    all_null = pl.Series("x", [None, None, None], dtype=pl.Float64)
    plan = plan_column("x", all_null.dtype, nullable=True)

    optional_decl = Input(name="x", annotation=float | None, null_policy=NullPolicy.OPTIONAL)
    optional_strategy = NULL_TIER_STRATEGIES[NullPolicy.OPTIONAL]
    via_column = optional_strategy.extract_column(all_null, optional_decl, plan)
    via_absent = optional_strategy.synthesize_absent(optional_decl, n)
    assert via_column.validity.tolist() == via_absent.validity.tolist() == [False, False, False]

    fill_decl = Input(name="x", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=7.0)
    fill_strategy = NULL_TIER_STRATEGIES[NullPolicy.MISSING_AS]
    via_column = fill_strategy.extract_column(all_null, fill_decl, plan)
    via_absent = fill_strategy.synthesize_absent(fill_decl, n)
    assert via_column.values.tolist() == via_absent.values.tolist() == [7.0, 7.0, 7.0]
    assert via_column.fill.reason is via_absent.fill.reason is FillReason.MISSING


def test_extract_frame_end_to_end_keeps_all_four_tiers_correct_at_once():
    """One frame, one column per tier — the whole registry exercised
    together, mirroring how a real pipeline declares its inputs."""
    frame = pl.DataFrame({
        "id": [1, 2, 3],
        "score": [600.0, None, 700.0],
        "backup_score": [None, 550.0, None],
        "income": [5000.0, None, 7000.0],
    })
    inputs = [
        Input(name="id", annotation=int, null_policy=NullPolicy.REQUIRED),
        Input(name="score", annotation=float, null_policy=NullPolicy.MISSING_AS, fill=-1.0),
        Input(name="backup_score", annotation=float, null_policy=NullPolicy.NOT_APPLICABLE_AS, fill=-2.0),
        Input(name="income", annotation=float | None, null_policy=NullPolicy.OPTIONAL),
    ]
    result = extract_frame(frame, inputs)
    assert result.routing.routed_count == 0  # only 'id' is REQUIRED, and it's clean

    assert result.columns["score"].values.tolist() == [600.0, -1.0, 700.0]
    assert result.columns["score"].fill.reason is FillReason.MISSING

    assert result.columns["backup_score"].values.tolist() == [-2.0, 550.0, -2.0]
    assert result.columns["backup_score"].fill.reason is FillReason.NOT_APPLICABLE

    assert result.columns["income"].validity.tolist() == [True, False, True]
    assert result.columns["income"].fill is None
