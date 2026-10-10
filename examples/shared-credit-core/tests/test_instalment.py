import pytest

from credit_core.instalment import (
    effective_annual_rate, instalment, instalment_before_fees, solve_advance_for_instalment, total_cost_of_credit,
    total_interest,
)


def test_instalment_is_the_annuity_payment():
    # R12 000 at 12% a year over 12 months: 1% a month, the standard annuity of R1 066.19.
    assert instalment_before_fees(12000.0, 12, 0.12) == pytest.approx(1066.1855, abs=1e-4)


def test_a_zero_rate_is_a_plain_division():
    assert instalment_before_fees(12000.0, 12, 0.0) == 1000.0


def test_fees_and_premium_are_added_to_the_instalment():
    assert instalment(1000.0, 69.0, 30.0) == 1099.0
    assert instalment(1000.0) == 1000.0


def test_total_cost_and_interest_include_the_initiation_fee():
    assert total_cost_of_credit(1099.0, 12, 500.0) == 13688.0
    assert total_interest(13688.0, 12000.0) == 1688.0


def test_effective_rate_compounds_the_monthly_rate():
    assert effective_annual_rate(0.12) == pytest.approx(1.01 ** 12 - 1)


@pytest.mark.parametrize("amount,term,rate", [(65000.0, 36, 0.1604), (10000.0, 12, 0.0), (500000.0, 84, 0.28)])
def test_the_inverse_recovers_the_advance(amount, term, rate):
    monthly = instalment_before_fees(amount, term, rate)
    assert solve_advance_for_instalment(monthly, term, rate) == pytest.approx(amount, abs=0.01)


def test_the_inverse_gives_the_advance_a_budget_can_carry():
    assert solve_advance_for_instalment(1066.1855, 12, 0.12) == pytest.approx(12000.0, abs=0.01)
