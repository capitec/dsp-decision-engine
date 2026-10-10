import pytest
from decikit import dag

from credit_core.affordability import (
    FAIL, INDETERMINATE, MARGINAL, PASS, affordability_buffer_applied_step, affordability_verdict_code_step,
    discretionary_income_step, max_affordable_instalment_step,
)

UNIT = dag(discretionary_income_step, affordability_buffer_applied_step, max_affordable_instalment_step,
           affordability_verdict_code_step, name="afford").emit(
    "discretionary_income", "affordability_buffer_applied", "max_affordable_instalment")
INCOME = {"net_monthly_income": 16000.0, "living_expenses": 6000.0, "existing_obligations": 1500.0}


def afford(score, params=None, **changes):
    return score(UNIT, {**INCOME, **changes}, params)


def test_capacity_is_discretionary_income_less_the_buffer(score):
    out = afford(score)
    assert out["discretionary_income"] == 8500.0
    assert out["max_affordable_instalment"] == 7480.0  # 8 500 * (1 - 12%)


def test_court_ordered_deductions_come_off_discretionary_income(score):
    assert afford(score, court_ordered_deductions=1000.0)["discretionary_income"] == 7500.0


def test_negative_discretionary_income_gives_no_capacity(score):
    out = afford(score, existing_obligations=12000.0)
    assert (out["discretionary_income"], out["max_affordable_instalment"]) == (-2000.0, 0.0)
    assert out["affordability_verdict_code"] == FAIL


@pytest.mark.parametrize("proposed,verdict", [
    (7000.0, PASS), (7480.0, PASS),           # up to the maximum
    (7481.0, MARGINAL), (7854.0, MARGINAL),   # within 5% over it
    (7854.01, FAIL), (20000.0, FAIL),
])
def test_the_verdict_compares_a_proposal_with_the_maximum(score, proposed, verdict):
    assert afford(score, proposed_instalment=proposed)["affordability_verdict_code"] == verdict


def test_a_larger_proposal_never_gets_a_better_verdict(score):
    verdicts = [afford(score, proposed_instalment=p)["affordability_verdict_code"] for p in range(0, 12000, 250)]
    assert verdicts == sorted(verdicts)


def test_without_a_proposal_the_verdict_says_whether_there_is_any_capacity(score):
    assert afford(score)["affordability_verdict_code"] == PASS
    assert afford(score, existing_obligations=12000.0)["affordability_verdict_code"] == FAIL


def test_no_net_income_cannot_be_assessed(score):
    assert afford(score, net_monthly_income=0.0)["affordability_verdict_code"] == INDETERMINATE


def test_a_consumer_sets_its_own_buffer_and_marginal_band_in_params(score):
    params = {"afford": {"affordability_buffer_applied": {"affordability_buffer": 0.18},
                         "affordability_verdict_code": {"marginal_band": 0.0}}}
    out = afford(score, params, proposed_instalment=7000.0)
    assert out["max_affordable_instalment"] == 6970.0  # 8 500 * 82%
    assert out["affordability_verdict_code"] == FAIL    # 7 000 is over, and no band is allowed


def test_one_step_can_run_twice_with_different_settings(score):
    default = affordability_buffer_applied_step.named("default_buffer").relabel(writes={"affordability_buffer_applied": "a"})
    strict = affordability_buffer_applied_step.named("strict_buffer").relabel(writes={"affordability_buffer_applied": "b"})
    out = score(dag(default, strict, name="p"), {}, {"p": {"strict_buffer": {"affordability_buffer": 0.18}}})
    assert (out["a"], out["b"]) == (0.12, 0.18)
