"""Offer-set construction and the final validation of the recommended offer."""
import copy
from dataclasses import asdict

import pytest
from decikit import Engine, flow

from loan_granting.offers import BEST_EXPECTED_VALUE, LARGEST_AMOUNT, LOWEST_TOTAL_COST, build_offer_set, expected_value, steps
from loan_granting.pricing import PricingResult
from loan_granting.solve import BIND_AFF, BIND_CAP, BIND_EXH

RULES = {"minimum_offer_amount": 2_000.0, "minimum_instalment": 150.0, "total_cost_ratio_threshold": 1.85,
         "effective_rate_ceiling": 0.60, "dedup_tolerance": 0.02, "lgd": 0.72, "average_ead": 0.55, "funding_cost": 0.0875}


def solution(term, amount, instalment, cost, ear, binding=BIND_AFF):
    pricing = PricingResult(amount, term, 7, True, 0.19, f"cell-{term}", "v1", instalment=instalment,
                            total_cost_of_credit=cost, effective_annual_rate=ear)
    return {"term_months": term, "amount": amount, "binding_constraint_code": binding, "evaluation_count": 5,
            "pricing": asdict(pricing)}


# A client who can afford about R4 350 a month, capped at R95 000, at nine permitted terms.
CANDIDATES = [
    solution(6, 22_100.0, 4_333.71, 26_002.26, 0.766), solution(12, 43_200.0, 4_347.89, 52_174.68, 0.431),
    solution(18, 61_900.0, 4_347.05, 78_246.90, 0.360), solution(24, 77_900.0, 4_348.64, 104_367.36, 0.343),
    solution(36, 95_000.0, 3_897.20, 140_299.20, 0.311, BIND_CAP), solution(48, 95_000.0, 3_246.67, 155_840.16, 0.304, BIND_CAP),
    solution(60, 95_000.0, 2_860.25, 171_615.00, 0.297, BIND_CAP), solution(72, 95_000.0, 2_663.33, 191_759.76, 0.303, BIND_CAP),
    {"term_months": 84, "excluded": "above the term cap"},
]


def recommended(offers):
    return next(o["term_months"] for o in offers if o["is_recommended"])


def test_each_minimum_viable_offer_rule_suppresses_the_terms_it_should_with_every_reason_recorded():
    offers, suppressed = build_offer_set(CANDIDATES, LARGEST_AMOUNT, 0.058, RULES)
    assert [o["term_months"] for o in offers] == [12, 18, 24, 36, 48, 60]
    assert {s["term_months"]: s["reason_codes"] for s in suppressed} == {6: [1605], 72: [1603, 1604], 84: []}
    assert next(s for s in suppressed if s["term_months"] == 84)["reason"] == "above the term cap"


def test_the_objective_picks_the_recommendation_from_the_same_offer_set():
    assert recommended(build_offer_set(CANDIDATES, LARGEST_AMOUNT, 0.058, RULES)[0]) == 36  # ties at R95 000, cheapest
    assert recommended(build_offer_set(CANDIDATES, LOWEST_TOTAL_COST, 0.058, RULES)[0]) == 12
    assert recommended(build_offer_set(CANDIDATES, BEST_EXPECTED_VALUE, 0.058, RULES)[0]) == 60


def test_exactly_one_offer_is_recommended():
    offers, _ = build_offer_set(CANDIDATES, LARGEST_AMOUNT, 0.058, RULES)
    assert sum(o["is_recommended"] for o in offers) == 1


def test_expected_value_falls_as_the_probability_of_default_rises():
    offer = next(o for o in build_offer_set(CANDIDATES, LARGEST_AMOUNT, 0.0, RULES)[0] if o["term_months"] == 60)
    values = [expected_value(offer, pd, 0.72, 0.55, 0.0875) for pd in (0.0, 0.05, 0.2)]
    assert values[0] == round(171_615.0 - 95_000.0 - 95_000.0 * 0.0875 * 5, 2)  # no default: the margin after funding
    assert values[0] > values[1] > values[2]


def test_a_riskier_client_is_recommended_a_shorter_term_when_the_objective_is_expected_value():
    patient = recommended(build_offer_set(CANDIDATES, BEST_EXPECTED_VALUE, 0.01, RULES)[0])
    cautious = recommended(build_offer_set(CANDIDATES, BEST_EXPECTED_VALUE, 0.35, RULES)[0])
    assert cautious < patient


def test_equal_amounts_with_instalments_within_the_tolerance_collapse_to_the_cheaper():
    close = [solution(36, 50_000.0, 2_000.0, 72_000.0, 0.3), solution(48, 50_000.0, 1_980.0, 95_000.0, 0.3)]
    offers, _ = build_offer_set(close, LARGEST_AMOUNT, 0.05, RULES)
    assert [o["term_months"] for o in offers] == [36]
    apart = [solution(36, 50_000.0, 2_000.0, 72_000.0, 0.3), solution(48, 50_000.0, 1_500.0, 72_000.0, 0.3)]
    assert len(build_offer_set(apart, LARGEST_AMOUNT, 0.05, RULES)[0]) == 2


def test_the_tie_break_prefers_the_lower_cost_and_then_the_shorter_term():
    same = [solution(48, 50_000.0, 1_500.0, 72_000.0, 0.3), solution(36, 50_000.0, 2_000.0, 72_000.0, 0.3)]
    offers, _ = build_offer_set(same, LARGEST_AMOUNT, 0.05, RULES)
    assert recommended(offers) == 36


def test_a_term_that_hit_the_evaluation_ceiling_is_suppressed_with_that_reason_not_as_unaffordable():
    exhausted = {"term_months": 60, "amount": None, "binding_constraint_code": BIND_EXH, "evaluation_count": 24, "pricing": None}
    none_affordable = {"term_months": 72, "amount": None, "binding_constraint_code": BIND_AFF, "evaluation_count": 3, "pricing": None}
    _, suppressed = build_offer_set([exhausted, none_affordable], LARGEST_AMOUNT, 0.05, RULES)
    assert [(s["term_months"], s["reason_codes"]) for s in suppressed] == [(60, [1420]), (72, [])]


def test_no_surviving_offer_is_an_empty_set_not_an_error():
    assert build_offer_set([], LARGEST_AMOUNT, 0.05, RULES) == ([], [])


@pytest.fixture(scope="module")
def validator(rate_card):
    *_, final_validation = steps(rate_card)
    return Engine().bind(flow(final_validation, name="check"), mode="interpreted")


VALIDATION_INPUTS = (
    "recommended_offer", "risk_grade", "applicant_age_years", "is_joint_application", "amount_cap", "term_cap",
    "worst_acceptable_grade", "max_affordable_instalment", "statutory_ceiling", "initiation_fee_cap",
    "monthly_service_fee_cap", "total_cost_ratio_threshold", "credit_life_substitution_declared")


@pytest.fixture
def approved(decide):
    out = decide()
    return {name: out[name] for name in VALIDATION_INPUTS}


def validate(validator, params, approved, **changes):
    out = validator.score({**approved, **changes}, {"shared": params["shared"]})
    return out["final_validation_passed"], out["final_validation_failures"]


def validate_offer(validator, params, approved, **offer_changes):
    return validate(validator, params, approved, recommended_offer={**approved["recommended_offer"], **offer_changes})


def test_the_recommended_offer_re_derives_from_its_amount_and_term_alone(validator, params, approved):
    assert validate(validator, params, approved) == (True, [])


@pytest.mark.parametrize("change, failed_check", [
    ({"instalment": 4_969.48}, "instalment_recomputes_to_the_cent"),
    ({"rate_cell_id": "rate_card.flex_loan@rc-2026.09#0|0|1"}, "rate_matches_the_card_cell"),
    ({"initiation_fee": 6_400.0}, "initiation_fee_recomputes"),
    ({"credit_life_premium": 300.0}, "credit_life_premium_recomputes"),
    ({"monthly_service_fee": 50.0}, "service_fee_recomputes"),
    ({"offered_amount": 120_050.0}, "amount_is_a_multiple_of_the_grid"),
    ({"term_months": 90}, "term_permitted"),
    ({"rate_card_version": "rc-old"}, "rate_card_version_matches"),
])
def test_a_tampered_offer_fails_the_check_that_covers_it(validator, params, approved, change, failed_check):
    passed, failures = validate_offer(validator, params, approved, **change)
    assert not passed and failed_check in failures


def test_an_offer_above_what_the_client_can_afford_fails_even_if_it_prices_correctly(validator, params, approved):
    affordable = approved["recommended_offer"]["instalment"] - 0.01
    assert validate(validator, params, approved, max_affordable_instalment=affordable) == (
        False, ["instalment_within_what_is_affordable"])


def test_an_amount_over_the_cap_fails(validator, params, approved):
    assert "amount_within_every_ceiling" in validate(validator, params, approved, amount_cap=100_000.0)[1]


def test_a_grade_worse_than_the_worst_acceptable_fails(validator, params, approved):
    assert "grade_within_the_worst_acceptable" in validate(validator, params, approved, worst_acceptable_grade=5.0)[1]


def test_with_no_recommended_offer_there_is_nothing_to_fail(validator, params, approved):
    assert validate(validator, params, approved, recommended_offer={}) == (True, [])
