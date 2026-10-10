from datetime import date

import pytest
from decikit import flow

from credit_core.fees import (
    build_fee_caps_table, fee_caps_version_step, initiation_fee, initiation_fee_capped, initiation_fee_capped_step,
    initiation_fee_step, monthly_service_fee, monthly_service_fee_step,
)
from credit_core.rounding import round_advance, round_instalment, round_rate

FEES = flow(fee_caps_version_step, build_fee_caps_table(), initiation_fee_step, initiation_fee_capped_step,
            monthly_service_fee_step, name="fees").emit("initiation_fee_cap", "monthly_service_fee_cap", "initiation_fee")


def test_instalments_round_half_up_to_the_cent():
    assert round_instalment(0.125) == 0.13  # Python's round(0.125, 2) gives 0.12
    assert round_instalment(2355.4949) == 2355.49


def test_advances_round_half_up_to_the_nearest_hundred():
    assert round_advance(250.0) == 300.0
    assert round_advance(149.99) == 100.0
    assert round_advance(65049.0) == 65000.0


def test_rates_round_to_four_places():
    assert round_rate(0.123456) == 0.1235


def test_the_initiation_fee_is_a_base_plus_a_share_of_the_excess_up_to_the_cap():
    assert initiation_fee(10000.0) == 1050.0
    assert initiation_fee(20000.0) == 2050.0
    assert initiation_fee(65000.0) == 6500.0  # 6 550 before the cap
    assert initiation_fee_capped(6500.0) and not initiation_fee_capped(2050.0)


def test_the_service_fee_is_flat_up_to_its_cap():
    assert monthly_service_fee() == 69.0
    assert monthly_service_fee(flat_fee=80.0, monthly_service_fee_cap=69.0) == 69.0


def test_the_caps_in_force_on_the_decision_date_bind(score):
    now = score(FEES, {"decision_date": date(2026, 9, 1), "offered_amount": 65000.0})
    assert (now["initiation_fee"], now["monthly_service_fee"], now["initiation_fee_capped"]) == (6500.0, 69.0, True)
    earlier = score(FEES, {"decision_date": date(2025, 6, 1), "offered_amount": 65000.0})
    assert (earlier["initiation_fee"], earlier["monthly_service_fee"]) == (6200.0, 65.0)  # the 2025 caps
    assert earlier["fee_caps_cell_id"] != now["fee_caps_cell_id"]


def test_fees_below_the_cap_are_not_flagged_as_capped(score):
    out = score(FEES, {"decision_date": date(2026, 9, 1), "offered_amount": 20000.0})
    assert (out["initiation_fee"], out["initiation_fee_capped"]) == (2050.0, False)


def test_a_date_before_any_gazetted_caps_cannot_be_resolved(score):
    with pytest.raises(LookupError):
        score(FEES, {"decision_date": date(2024, 6, 1), "offered_amount": 1000.0})
