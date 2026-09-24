from limit_mgmt.caps import policy_proposed_limit, round_down_500
from limit_mgmt.vocab import Cap


def test_round_down_never_up():
    """§5.5: "rounded down... without exception... a cap that can be breached by
    rounding is not a cap." """
    assert round_down_500(14_999.0) == 14_500.0
    assert round_down_500(15_000.0) == 15_000.0
    assert round_down_500(500.0) == 500.0
    assert round_down_500(499.0) == 0.0


def test_lowest_cap_wins_and_is_named():
    # `cap_observed_spend` (C5) is a genuine binding cap here only because it is above
    # `current_limit`, matching its own invariant (C5 = max(current_limit, ...)) -- a cap
    # below current_limit would never be C5's own output (see `caps.cap_observed_spend`).
    rounded, binding = policy_proposed_limit(
        uncapped_target_limit=30_000.0, current_limit=5_000.0,
        cap_product_max=300_000.0, cap_income_multiple=22_000.0, cap_total_exposure=140_000.0,
        cap_observed_spend=9_600.0, matrix_max_increase=20_000.0, matrix_min_increment_unadjusted=500.0,
    )
    assert binding == Cap.C5_OBSERVED_SPEND
    assert rounded == 9_500.0  # 9600 rounded down to the nearest 500


def test_deterministic_tie_break_by_table_order():
    """§5.5: "Where two caps bind at the same value, both are recorded and the tie is
    broken by the order in the table above, deterministically." C1 precedes C2 in the
    table, so an exact tie resolves to C1."""
    rounded, binding = policy_proposed_limit(
        uncapped_target_limit=50_000.0, current_limit=10_000.0,
        cap_product_max=20_000.0, cap_income_multiple=20_000.0, cap_total_exposure=90_000.0,
        cap_observed_spend=90_000.0, matrix_max_increase=90_000.0, matrix_min_increment_unadjusted=500.0,
    )
    assert binding == Cap.C1_PRODUCT_MAX


def test_below_minimum_increment_yields_no_increase():
    """§5.4: "Below this the cell yields no increase at all."""
    rounded, binding = policy_proposed_limit(
        uncapped_target_limit=15_300.0, current_limit=15_000.0,
        cap_product_max=300_000.0, cap_income_multiple=300_000.0, cap_total_exposure=300_000.0,
        cap_observed_spend=300_000.0, matrix_max_increase=1_000.0, matrix_min_increment_unadjusted=2_000.0,
    )
    assert rounded == 15_000.0  # the 300-500 increase falls below the cell's own R2000 floor


def test_never_proposes_below_current_limit():
    rounded, binding = policy_proposed_limit(
        uncapped_target_limit=8_000.0, current_limit=15_000.0,
        cap_product_max=300_000.0, cap_income_multiple=9_000.0, cap_total_exposure=300_000.0,
        cap_observed_spend=300_000.0, matrix_max_increase=0.0, matrix_min_increment_unadjusted=500.0,
    )
    assert rounded == 15_000.0
