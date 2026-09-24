"""`core.affordability`: the unit only (00 §6.5; project 02 owns the assessment)."""
from decider import Engine, dag

from credit_core.affordability import (
    FAIL, INDETERMINATE, MARGINAL, PASS, affordability_buffer_applied_step, affordability_verdict_code_step,
    discretionary_income_step, max_affordable_instalment_step,
)


def _pipeline():
    return dag(discretionary_income_step, affordability_buffer_applied_step, max_affordable_instalment_step,
               affordability_verdict_code_step, name="afford").emit(
        "discretionary_income", "affordability_buffer_applied", "max_affordable_instalment",
        "affordability_verdict_code")


def _score(**record):
    exe = Engine().bind(_pipeline(), mode="interpreted")
    base = {"net_monthly_income": 16000.0, "living_expenses": 6000.0, "existing_obligations": 1500.0}
    return exe.score({**base, **record})


def test_discretionary_income_subtracts_court_ordered_deductions_outside_statutory():
    """Addendum item 10: `court_ordered_deductions` is a separate subtraction from `core.deductions`."""
    without = _score()["discretionary_income"]
    with_court_order = _score(court_ordered_deductions=1000.0)["discretionary_income"]
    assert without - with_court_order == 1000.0


def test_verdict_is_monotone_in_the_proposed_instalment():
    """02 §5.7.2(c): a larger proposed instalment can never produce a better verdict."""
    verdicts = [_score(proposed_instalment=i)["affordability_verdict_code"]
                for i in (1000.0, 5000.0, 8000.0, 8500.0, 20000.0)]
    rank = {PASS: 0, MARGINAL: 1, FAIL: 2}
    ranks = [rank[v] for v in verdicts]
    assert ranks == sorted(ranks)


def test_no_proposed_instalment_still_yields_pass_or_fail():
    out = _score()
    assert out["affordability_verdict_code"] in (PASS, FAIL)


def test_indeterminate_net_income_cannot_be_assessed():
    exe = Engine().bind(affordability_verdict_code_step, mode="interpreted")
    out = exe.score({"max_affordable_instalment": 0.0, "net_monthly_income": 0.0})
    assert out["affordability_verdict_code"] == INDETERMINATE


def test_same_capability_twice_with_different_settings_in_one_flow():
    """Acceptance §10 item 4: 07 supplies its own buffer (18%) against this module's default (12%)."""
    default_buffer = affordability_buffer_applied_step.named("buffer_default").relabel(
        writes={"affordability_buffer_applied": "buffer_default_value"})
    buffer_07 = affordability_buffer_applied_step.named("buffer_07").relabel(
        writes={"affordability_buffer_applied": "buffer_07_value"})
    named_twice = dag(default_buffer, buffer_07, name="afford")
    exe = Engine().bind(named_twice, mode="interpreted")
    params = {"afford": {"buffer_07": {"affordability_buffer": 0.18}}}
    out = exe.score({}, params=params)
    assert out["buffer_default_value"] == 0.12
    assert out["buffer_07_value"] == 0.18
