import math

import pytest

from limit_mgmt import affordability
from limit_mgmt.vocab import EvidenceTier, IncreasePath


def test_the_notional_instalment_is_the_minimum_payment_on_the_new_limit():
    assert affordability.notional_instalment(25_000.0, 20.0) == 875.0       # 3.5% on the card
    assert affordability.notional_instalment(25_000.0, 21.0) == 1_250.0     # 5% on the facility
    assert affordability.notional_instalment(2_000.0, 20.0) == 100.0        # the card's R100 minimum
    assert affordability.notional_instalment(500.0, 21.0) == 50.0           # the facility's R50 minimum
    assert affordability.notional_instalment(0.0, 20.0) == 0.0


@pytest.mark.parametrize("salary, variability, declared_age, tier", [
    (True, 0.06, 9_999, EvidenceTier.A_VERIFIED_SALARY), (True, 0.15, 9_999, EvidenceTier.A_VERIFIED_SALARY),
    (True, 0.30, 9_999, EvidenceTier.B_IRREGULAR_DEPOSITS), (False, 1.0, 200, EvidenceTier.C_DECLARED_REFRESHED),
    (False, 1.0, 365, EvidenceTier.C_DECLARED_REFRESHED), (False, 1.0, 500, EvidenceTier.D_DECLARED_STALE),
    (False, 1.0, 9_999, EvidenceTier.E_NO_EVIDENCE)])
def test_evidence_tiers_follow_the_waterfall(salary, variability, declared_age, tier):
    assert affordability.evidence_tier_code(salary, variability, declared_age, 0.15, 365) == tier


def test_a_refreshed_declared_income_is_indexed_for_inflation_for_up_to_a_year():
    def declared(age):
        return affordability.programme_income_evidence(EvidenceTier.C_DECLARED_REFRESHED, 0.0, 21_000.0, age, 0.052, 0.95, 0.4)[2]
    assert declared(200) == 21_539.08            # six months
    assert declared(360) == declared(365) == 22_092.0  # twelve months, the most it is indexed


def test_what_is_not_evidenced_is_absent_not_zero():
    statement, confidence, declared = affordability.programme_income_evidence(EvidenceTier.A_VERIFIED_SALARY, 23_500.0)
    assert (statement, confidence) == (23_500.0, 0.95) and math.isnan(declared)
    assert all(math.isnan(x) for x in affordability.programme_income_evidence(EvidenceTier.E_NO_EVIDENCE))


def test_a_steady_salary_goes_to_the_assessment_without_a_haircut(decide):
    out = decide()
    assert (out["evidence_tier_code"], out["gross_monthly_income"], out["increase_path_code"]) == (1, 23_500.0, IncreasePath.AUTOMATIC)


def test_irregular_deposits_carry_a_15_percent_haircut_and_are_still_automatic_when_fresh(decide):
    out = decide(salary_deposit_variability_pct=0.30)
    assert (out["evidence_tier_code"], out["gross_monthly_income"], out["increase_path_code"]) == (2, 19_975.0, IncreasePath.AUTOMATIC)


def test_salary_evidence_older_than_45_days_makes_the_increase_conditional(decide):
    out = decide(salary_deposit_as_of_date="2026-08-10")
    assert (out["income_staleness_days"], out["increase_path_code"]) == (45, IncreasePath.AUTOMATIC)
    out = decide(salary_deposit_as_of_date="2026-08-09")
    assert (out["income_staleness_days"], out["increase_path_code"]) == (46, IncreasePath.CONDITIONAL)


def test_a_bureau_view_older_than_35_days_is_conditional_and_older_than_60_is_not_offered(decide):
    assert decide(bureau_as_of_date="2026-08-20")["increase_path_code"] == IncreasePath.AUTOMATIC       # 35 days
    assert decide(bureau_as_of_date="2026-08-19")["increase_path_code"] == IncreasePath.CONDITIONAL     # 36 days
    stale = decide(bureau_as_of_date="2026-07-15")                                                      # 71 days
    assert (stale["increase_path_code"], stale["decline_reason_codes"]) == (IncreasePath.FAIL, [7010])


def test_growing_external_debt_or_stale_expenses_make_the_increase_conditional(decide):
    assert decide(obligations_growth_pct=0.40)["increase_path_code"] == IncreasePath.CONDITIONAL
    assert decide(expenses_refreshed_or_normed=False)["increase_path_code"] == IncreasePath.CONDITIONAL


def test_declared_income_alone_is_conditional_however_fresh(decide):
    for age, tier in ((200, 3), (500, 4)):
        out = decide(verified_salary_present_3m=False, declared_income_age_days=age)
        assert (out["evidence_tier_code"], out["increase_path_code"]) == (tier, IncreasePath.CONDITIONAL)


def test_no_income_evidence_at_all_is_not_offered_and_says_the_income_was_unestablished(decide):
    out = decide(verified_salary_present_3m=False, declared_income_age_days=9_999)
    assert (out["increase_path_code"], out["evidence_sufficiency_code"], out["decline_reason_codes"]) == (IncreasePath.FAIL, 1, [7010])
    assert out["final_proposed_limit"] == out["current_limit"]


def test_a_client_who_cannot_carry_any_of_the_increase_is_not_offered_and_keeps_the_limit(decide):
    out = decide(verified_salary_amount=4_000.0)
    assert (out["increase_path_code"], out["decline_reason_codes"], out["final_proposed_limit"]) == (IncreasePath.FAIL, [7003], 15_000.0)


def test_affordability_caps_an_increase_it_can_partly_carry_and_names_itself_the_binding_cap(decide):
    out = decide(verified_salary_amount=11_000.0)
    assert out["policy_proposed_limit"] == 20_000.0
    assert (out["max_affordable_instalment"], out["final_proposed_limit"], out["binding_cap_code"]) == (672.0, 19_000.0, 7)
    assert out["final_proposed_limit"] % 500 == 0 and out["final_proposed_limit"] * 0.035 <= out["max_affordable_instalment"]
    assert out["increase_path_code"] == IncreasePath.AUTOMATIC


def test_the_programme_buffer_is_an_overlay_on_the_shared_grid_and_the_unadjusted_buffer_survives(decide):
    out = decide()
    assert (out["affordability_buffer_unadjusted"], out["affordability_buffer_applied"]) == (0.1227, 0.1827)
    assert out["buffer_adjustments_applied"] == ["ADJ-07-BUFFER-BASE"]


def test_the_assessment_runs_in_limit_increase_mode_and_records_the_notional_instalment(decide):
    out = decide()
    assert out["notional_instalment"] == 700.0 and out["affordability_verdict_code"] == 1
