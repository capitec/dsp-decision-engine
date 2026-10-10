"""The do-nothing position, and the short-circuit to plain granting when consolidation is not needed."""
import pytest

CLEAN = [
    {"account_ref": 1, "account_type_code": 1, "provider_code": 0, "is_internal": True, "balance": 20_000.0, "instalment": 900.0,
     "nominal_annual_rate": 0.14, "remaining_term_months": 24.0, "months_in_arrears": 0, "opened_date": "2022-03-10"},
    {"account_ref": 2, "account_type_code": 10, "provider_code": 0, "is_internal": True, "balance": 3_000.0,
     "credit_limit": 8_000.0, "instalment": 250.0, "nominal_annual_rate": 0.16, "months_in_arrears": 0,
     "opened_date": "2021-06-01"},
]


def test_the_baseline_measures_the_client_as_they_stand(assess):
    base = assess()["baseline"]
    assert (base["account_count"], base["provider_count"], base["worst_arrears_months"]) == (9, 8, 1)
    assert base["total_instalment"] == 12_123.0
    assert 0.18 < base["weighted_average_rate"] < 0.21
    net = assess()["invariants"]["net_monthly_income"]
    assert base["debt_service_ratio"] == pytest.approx(base["existing_obligations"] / net, abs=1e-4)


def test_a_revolving_account_has_no_term_so_its_remaining_cost_assumes_the_declared_horizon(assess, params, sample):
    longer = {**params, "consolidation": {**params["consolidation"], "baseline_measures": {"paydown_horizon_months": 60}}}
    assert assess(params_document=longer)["baseline"]["total_remaining_cost"] > assess()["baseline"]["total_remaining_cost"]


def test_the_sample_client_needs_consolidation_and_the_reasons_are_on_file(assess):
    out = assess()
    assert out["short_circuit_applies"] is False
    assert out["short_circuit_verdicts"] == {
        "request_affordable_as_made": True, "debt_service_ratio_within_ceiling": True, "no_account_in_arrears": False,
        "rate_gap_within_threshold": False, "fewer_than_six_active_accounts": False, "no_settleable_account_at_high_rate": False}


def test_a_clean_client_with_an_affordable_request_is_granted_plainly_and_the_search_is_skipped(assess):
    out = assess(accounts=CLEAN, requested_amount=3_000.0)
    assert out["short_circuit_applies"] is True and all(out["short_circuit_verdicts"].values())
    assert (out["outcome_code"], out["chosen_offer"]["offered_amount"], out["chosen_offer"]["term_months"]) == (1, 3_000.0, 36)
    assert out["scenarios"] == [] and out["search_record"]["termination_cause"] == "short_circuited"


@pytest.mark.parametrize("change, condition", [
    ({"months_in_arrears": 1}, "no_account_in_arrears"),
    ({"nominal_annual_rate": 0.289, "balance": 3_000.0}, "no_settleable_account_at_high_rate"),
    ({"nominal_annual_rate": 0.27, "balance": 9_000.0, "credit_limit": 10_000.0}, "rate_gap_within_threshold"),
])
def test_each_short_circuit_condition_can_be_the_one_that_sends_the_client_to_the_search(assess, change, condition):
    accounts = [{**CLEAN[0], "balance": 5_000.0}, {**CLEAN[1], **change}]
    out = assess(accounts=accounts, requested_amount=3_000.0)
    assert out["short_circuit_verdicts"][condition] is False and out["short_circuit_applies"] is False
    assert out["search_record"]["termination_cause"] != "short_circuited"


def test_six_active_accounts_send_the_client_to_the_search(assess):
    accounts = [{**CLEAN[1], "account_ref": 10 + n, "opened_date": f"20{15 + n}-01-01", "balance": 1_000.0 + n} for n in range(6)]
    out = assess(accounts=accounts, requested_amount=3_000.0)
    assert out["short_circuit_verdicts"]["fewer_than_six_active_accounts"] is False


def test_a_request_the_client_cannot_afford_as_made_always_gets_the_search(assess):
    out = assess(requested_amount=500_000.0)
    assert out["short_circuit_verdicts"]["request_affordable_as_made"] is False and out["short_circuit_applies"] is False
    assert out["plain_offer"]["offered_amount"] < 500_000.0  # what they could have, plainly, is recorded beside it


def test_the_plain_offer_is_the_largest_affordable_flex_loan(assess):
    offer = assess()["plain_offer"]
    assert (offer["offered_amount"], offer["term_months"], offer["binding_constraint_code"]) == (20_000.0, 36, "BIND-REQ")
    assert offer["rate_cell_id"].startswith("rate_card.flex_loan@")
