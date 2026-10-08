"""The gates before the search: every gate recorded, and a client under debt review never declined."""
import pytest

from consolidation import reasons


def test_every_gate_is_recorded_not_only_the_one_that_fired(assess):
    fired = assess(client_under_administration=True, bureau_unobtainable=True)["eligibility_gate_verdicts"]
    assert list(fired) == [f"CON-ELIG-0{n}" for n in range(1, 8)]
    assert [g for g, v in fired.items() if v] == ["CON-ELIG-02", "CON-ELIG-06"]


def test_the_sample_client_proceeds_to_the_search(assess):
    out = assess()
    assert out["eligibility_route"] == "proceed" and not any(out["eligibility_gate_verdicts"].values())


def test_a_client_under_debt_review_is_referred_never_declined(assess):
    out = assess(client_under_debt_review=True)
    assert (out["eligibility_route"], out["outcome_code"], out["decline_reason_codes"]) == ("debt_review_path", 3, [reasons.DEBT_REVIEW])
    assert out["scenarios"] == [] and out["search_record"]["termination_cause"] == "not_run"


def test_debt_review_outranks_every_other_gate(assess):
    out = assess(client_under_debt_review=True, client_under_administration=True)
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [reasons.DEBT_REVIEW])


@pytest.mark.parametrize("change, reason", [
    ({"client_under_administration": True}, reasons.ADMINISTRATION),
    ({"bureau_unobtainable": True}, reasons.NO_BUREAU_VIEW),
])
def test_a_terminal_gate_declines_with_its_reason(assess, change, reason):
    out = assess(**change)
    assert (out["outcome_code"], out["decline_reason_codes"]) == (4, [reason])


def test_a_client_with_no_income_evidence_is_declined(assess):
    out = assess(applicant1_payslip_income=None, applicant1_employment_type_code=0)
    assert out["eligibility_gate_verdicts"]["CON-ELIG-05"] and (out["outcome_code"], reasons.NO_INCOME in out["decline_reason_codes"]) == (4, True)


def test_a_consolidation_in_the_last_six_months_is_referred(assess):
    out = assess(last_consolidation_date="2026-06-15")
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [reasons.SERIAL_CONSOLIDATION])
    assert assess(last_consolidation_date="2026-02-01")["eligibility_route"] == "proceed"


def test_an_active_reckless_lending_allegation_is_referred_to_compliance(assess):
    out = assess(active_reckless_lending_allegation=True)
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [reasons.RECKLESS_ALLEGATION])


def test_fewer_than_two_settleable_accounts_means_consolidation_does_not_apply(assess, sample):
    out = assess(accounts=sample["accounts"][:1])
    assert (out["eligibility_route"], out["eligibility_gate_verdicts"]["CON-ELIG-04"]) == ("not_applicable", True)
    assert out["scenarios"] == []


def test_when_consolidation_does_not_apply_the_request_is_granted_plainly(assess, sample):
    out = assess(accounts=sample["accounts"][:1])
    assert out["outcome_code"] == 1
    assert (out["chosen_offer"]["offered_amount"], out["chosen_offer"]["term_months"]) == (20_000.0, 36)


def test_the_minimum_settleable_accounts_is_a_param(assess, sample, params):
    changed = {**params, "consolidation": {**params["consolidation"], "eligibility_gates": {
        **params["consolidation"]["eligibility_gates"], "minimum_settleable_accounts": 1}}}
    assert assess(accounts=sample["accounts"][:1], params_document=changed)["eligibility_route"] == "proceed"
