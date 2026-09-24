"""`core.instalment`: amortisation and its inverse are the same unit, run differently (00 §6.6, §13 Q7)."""
import pytest

from credit_core.instalment import instalment_before_fees, solve_advance_for_instalment


@pytest.mark.parametrize("amount,term,rate", [
    (65000.0, 36, 0.1604),
    (10000.0, 12, 0.0),        # zero-rate edge case
    (500000.0, 84, 0.28),
])
def test_inverse_recovers_the_advance_to_the_cent(amount, term, rate):
    monthly = instalment_before_fees(amount, term, rate)
    recovered = solve_advance_for_instalment(monthly, term, rate)
    assert recovered == pytest.approx(amount, abs=0.01)


def test_zero_rate_is_a_plain_division():
    assert instalment_before_fees(12000.0, 12, 0.0) == pytest.approx(1000.0)
