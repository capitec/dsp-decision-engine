"""EP-3, the annual review: causes of a grade move, the weaker bases, re-pricing, the limit decision and exit."""
import copy
from datetime import date

import pytest

from business_credit_e2e import history, review, vocab
from business_nested.records import entity

NEXT_YEAR = date(2027, 9, 24)
STRONG = dict(ebitda=3_000_000.0, finance_charges=100_000.0, current_assets=4_000_000.0, interest_bearing_debt=500_000.0,
              tangible_net_worth=6_000_000.0)
WEAK = dict(ebitda=100_000.0, finance_charges=400_000.0, current_assets=800_000.0, current_liabilities=1_500_000.0,
            interest_bearing_debt=3_000_000.0, tangible_net_worth=300_000.0)
# The limit programme's inputs for a small revolving facility: its conduct over six cycles and the income on file.
LIMIT_ACCOUNT = {
    "cycle_balances": [8200.0, 7900.0, 8600.0, 7100.0, 6800.0, 7400.0], "cycle_limits": [15000.0] * 6,
    "cycle_purchase_values": [7200.0, 6800.0, 8100.0, 6600.0, 7900.0, 8400.0], "months_on_book": 42,
    "consent_automatic_increase": True, "worst_arrears_months_24": 0, "payment_to_balance_ratio_6m": 0.55,
    "declared_gross_income_on_file": 22000.0, "declared_net_income_on_file": 17500.0, "declared_income_age_days": 540,
    "declared_income_raw": 21000.0, "applicant1_employment_type_code": 1,
}


def owners(bureau_score, months=100.0, delinquency=0.0):
    """Two owners with the same bureau record and no adverse events."""
    return [entity(1, pct=60.0, is_controlling=True, bureau_score=bureau_score, months_on_record=months,
                   worst_delinquency_months=delinquency),
            entity(2, pct=40.0, bureau_score=bureau_score, months_on_record=months, worst_delinquency_months=delinquency)]


@pytest.fixture
def run_review(executable, params, request_for, originate):
    """`run_review(origination=..., review=..., **kwargs)` originates a facility, reviews it a year on, and returns
    `(origination record, review record)`; each argument is the changes to the sample request for that decision."""

    def run(origination=None, changes=None, *, run_params=None, **kwargs):
        store = history.DecisionStore()
        first = originate(store, request_for(**(origination or {})))
        base = {**(origination or {}), "decision_date": NEXT_YEAR, "knowledge_date": NEXT_YEAR, **(changes or {})}
        kwargs.setdefault("statement_age_months", 6.0)
        kwargs.setdefault("limit_account", LIMIT_ACCOUNT)
        return first, review.annual_review(executable, run_params or params, store, request_for(**base), **kwargs)

    return run


def causes(record):
    return {name: round(value, 6) for name, value in record["grade_migration"]["causes"].items()}


def test_a_review_links_to_its_predecessor_and_adds_up_the_movement(run_review):
    first, reviewed = run_review()
    assert reviewed["assessment_kind_code"] == vocab.EP3_ANNUAL_REVIEW
    assert reviewed["predecessor_decision_of_record_id"] == first["decision_of_record_id"]
    migration = reviewed["grade_migration"]
    assert sum(migration["causes"].values()) == pytest.approx(migration["observed_pd_movement"])
    assert reviewed["comparison_basis_code"] == vocab.COMPARISON_AS_GRADED


def test_a_year_with_nothing_changed_moves_only_where_an_overlay_lapsed(run_review):
    first, reviewed = run_review()
    # the term-loan overlay ended on 2026-12-31, so the probability falls by exactly what it added
    effect = first["probability_of_default"] - first["probability_of_default_unadjusted"]
    assert causes(reviewed) == {"model": 0.0, "business_data": 0.0, "entity_structure": 0.0, "entity_data": 0.0,
                                "overlay": pytest.approx(-effect, abs=1e-6)}
    assert (first["risk_grade"], reviewed["risk_grade"]) == (7, 6)


def test_better_financials_are_a_business_data_cause_and_nothing_else(run_review):
    _, reviewed = run_review(origination={"product_code": 51, **WEAK, "entities": owners(780.0)}, changes=STRONG)
    moved = {name: value for name, value in causes(reviewed).items() if value}
    assert list(moved) == ["business_data"] and moved["business_data"] < 0


def test_a_better_bureau_record_is_an_entity_data_cause(request_for, run_review):
    improved = copy.deepcopy(request_for()["entities"])
    improved[3]["bureau_score"] = 700.0  # Van Wyk, who was 540
    _, reviewed = run_review(origination={"product_code": 51}, changes={"product_code": 51, "entities": improved})
    moved = {name: value for name, value in causes(reviewed).items() if value}
    assert list(moved) == ["entity_data"] and moved["entity_data"] < 0


def test_a_new_ownership_split_is_an_entity_structure_cause(request_for, run_review):
    restructured = copy.deepcopy(request_for()["entities"])
    restructured[0]["effective_ownership_pct"], restructured[3]["effective_ownership_pct"] = 13.0, 32.0
    _, reviewed = run_review(origination={"product_code": 51}, changes={"product_code": 51, "entities": restructured})
    moved = {name: value for name, value in causes(reviewed).items() if value}
    assert list(moved) == ["entity_structure"]


def test_a_change_of_policy_since_the_last_decision_is_the_model_cause(params, run_review):
    new_policy = copy.deepcopy(params)
    new_policy["business_credit_e2e"]["business_credit"]["decision"]["business_grade"]["combined_pd_before_overlay"][
        "financial_weight"] = 0.6  # the financials count for more
    _, reviewed = run_review(origination={"product_code": 51}, changes={"product_code": 51}, run_params=new_policy)
    moved = {name: value for name, value in causes(reviewed).items() if value}
    assert list(moved) == ["model"]


@pytest.mark.parametrize("age, accounts, turnover, basis", [
    (9, False, 0, vocab.REVIEW_COMPLETE), (9.5, False, 0, vocab.REVIEW_LATE), (15, False, 0, vocab.REVIEW_LATE),
    (18, True, 0, vocab.REVIEW_STALE), (18, False, 8, vocab.REVIEW_TURNOVER_ONLY), (30, True, 6, vocab.REVIEW_TURNOVER_ONLY),
    (30, True, 5, vocab.REVIEW_NOT_PERFORMED), (18, False, 0, vocab.REVIEW_NOT_PERFORMED)])
def test_the_review_basis_follows_the_age_of_the_statements_and_the_fallback_evidence(age, accounts, turnover, basis):
    assert review.review_basis_code(age, accounts, turnover) == basis


@pytest.mark.parametrize("decision_date, basis, due", [
    (date(2027, 9, 24), vocab.REVIEW_COMPLETE, date(2028, 9, 24)), (date(2027, 9, 24), vocab.REVIEW_STALE, date(2028, 3, 24)),
    (date(2027, 9, 24), vocab.REVIEW_TURNOVER_ONLY, date(2027, 12, 24)), (date(2027, 12, 31), vocab.REVIEW_NOT_PERFORMED, date(2028, 2, 29)),
    (date(2027, 8, 31), vocab.REVIEW_STALE, date(2028, 2, 29))])
def test_a_weaker_review_buys_less_time_and_month_ends_clamp(decision_date, basis, due):
    assert review.next_review_date(decision_date, basis) == due


def test_a_late_review_trusts_95_percent_of_ebitda(run_review):
    _, complete = run_review()
    _, late = run_review(statement_age_months=12.0)
    assert late["request"]["ebitda"] == 1_045_000.0 and complete["request"]["ebitda"] == 1_100_000.0
    assert late["review_basis_code"] == vocab.REVIEW_LATE


def test_a_stale_review_holds_a_strong_client_to_the_grade_floor(run_review):
    strong = {"product_code": 51, **STRONG, "entities": owners(780.0)}
    _, complete = run_review(strong, statement_age_months=6.0)
    _, stale = run_review(strong, statement_age_months=18.0, management_accounts_present=True)
    assert complete["risk_grade"] < 7  # without the floor the client is better than 7
    assert stale["risk_grade"] == 7
    assert stale["next_review_date"] == date(2028, 3, 24)


def test_a_stale_review_never_flatters_a_client_who_is_already_worse_than_the_floor(run_review):
    weak = {**WEAK, "entities": owners(620.0, 40.0, 3.0)}
    _, complete = run_review(weak, statement_age_months=6.0)
    _, stale = run_review(weak, statement_age_months=18.0, management_accounts_present=True)
    assert complete["risk_grade"] == 10
    assert stale["risk_grade"] == 10  # a floor of 7 must not lift a grade 10 to 7


def test_a_turnover_only_review_floors_the_grade_at_6(run_review):
    strong = {"product_code": 51, **STRONG, "entities": owners(780.0)}
    _, reviewed = run_review(strong, statement_age_months=24.0, turnover_months=8)
    assert (reviewed["review_basis_code"], reviewed["risk_grade"]) == (vocab.REVIEW_TURNOVER_ONLY, 6)


def test_a_review_with_no_evidence_completes_with_the_grade_unchanged_and_flagged(run_review):
    first, reviewed = run_review(statement_age_months=30.0)
    assert reviewed["review_basis_code"] == vocab.REVIEW_NOT_PERFORMED
    assert (reviewed["risk_grade"], reviewed["grade_stale"]) == (first["risk_grade"], True)
    assert reviewed["next_review_date"] == date(2027, 11, 24)


def test_a_facility_priced_above_todays_rate_reports_a_negative_shortfall_and_no_action(run_review):
    _, reviewed = run_review()  # the overlay lapsed, so today's grade 6 prices at 18.0% against the contract's 19.8%
    reprice = reviewed["reprice"]
    assert (reprice["contractual_rate"], reprice["indicated_rate"]) == (0.198, 0.18)
    assert reprice["margin_shortfall_bps"] == -180.0
    assert reprice["margin_shortfall_per_year"] == -18_000.0
    assert reprice["action"] == vocab.REPRICE_NONE


def test_a_term_loan_underpriced_for_its_risk_is_flagged_for_renegotiation_not_repriced(run_review):
    _, reviewed = run_review(origination={**STRONG, "entities": owners(780.0)}, changes={**WEAK, "entities": owners(780.0)})
    assert reviewed["reprice"]["margin_shortfall_bps"] > 0
    assert reviewed["reprice"]["action"] == vocab.REPRICE_FLAG_RENEGOTIATE  # a fixed margin cannot move mid-term


def test_a_revolving_facility_can_be_repriced_on_notice(run_review):
    _, reviewed = run_review(origination={"product_code": 51, **STRONG, "entities": owners(780.0)},
                             changes={"product_code": 51, **WEAK, "entities": owners(780.0)})
    assert reviewed["reprice"]["margin_shortfall_bps"] > 0
    assert reviewed["reprice"]["action"] == vocab.REPRICE_RESET_ON_NOTICE


def test_an_unchanged_price_needs_no_action(run_review):
    _, reviewed = run_review(origination={"product_code": 51}, changes={"product_code": 51})
    assert (reviewed["reprice"]["margin_shortfall_bps"], reviewed["reprice"]["action"]) == (0.0, vocab.REPRICE_NONE)


def test_the_contract_terms_carry_forward_so_a_second_review_compares_with_the_contract(executable, params, request_for, originate):
    store = history.DecisionStore()
    first = originate(store, request_for())
    for year in (2027, 2028):
        reviewed = review.annual_review(executable, params, store, request_for(decision_date=f"{year}-09-24"), statement_age_months=6.0)
    assert reviewed["reprice"]["contractual_rate"] == first["nominal_annual_rate"] == 0.198
    assert [r["decision_of_record_id"] for r in store.history(500001)] == [1, 2, 3]


def test_a_revolving_facility_gets_a_limit_decision_from_the_limit_programme(run_review):
    small = {"product_code": 51, "requested_amount": 15_000.0}
    first, reviewed = run_review(small)
    assert first["offered_amount"] == 15_000.0
    decision = reviewed["limit_decision"]
    assert (decision["current_limit"], decision["proposed_limit"]) == (15_000.0, 22_000.0)
    assert decision["change_type_code"] == 2  # an increase that waits for the client to confirm their income


def test_a_revolving_review_without_the_limit_programmes_inputs_is_refused(run_review):
    with pytest.raises(ValueError, match="limit_account"):
        run_review({"product_code": 51}, limit_account=None)


def test_a_term_facility_has_no_limit_decision(run_review):
    assert run_review()[1]["limit_decision"] is None


def test_two_reviews_at_grade_10_or_worse_recommend_an_exit(run_review):
    weak = {**WEAK, "entities": owners(620.0, 40.0, 3.0)}
    first, reviewed = run_review(weak)
    assert (first["risk_grade"], first["outcome_code"]) == (10, 3)  # referred, then approved by a person
    assert reviewed["risk_grade"] == 10
    assert reviewed["exit_triggers"] == [vocab.EXIT_TWO_REVIEWS_AT_GRADE_10]


def test_one_review_at_grade_10_after_a_better_one_is_not_yet_an_exit(run_review):
    _, reviewed = run_review(changes={**WEAK, "entities": owners(620.0, 40.0, 3.0)})
    assert reviewed["risk_grade"] == 10 and reviewed["exit_triggers"] == []


def test_a_declined_application_has_no_facility_to_review(executable, params, request_for, originate):
    store = history.DecisionStore()
    declined = originate(store, request_for(**WEAK, entities=owners(480.0, 20.0, 9.0)))
    assert declined["outcome_code"] == 4
    with pytest.raises(KeyError, match="no approved decision"):
        review.annual_review(executable, params, store, request_for(), statement_age_months=6.0)


def test_a_facility_never_originated_has_nothing_to_review(executable, params, request_for):
    with pytest.raises(KeyError, match="no approved decision"):
        review.annual_review(executable, params, history.DecisionStore(), request_for(), statement_age_months=6.0)
