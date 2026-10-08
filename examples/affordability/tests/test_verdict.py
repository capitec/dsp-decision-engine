import math

import pytest

from assessment import verdict

def test_a_proposal_is_compared_with_the_maximum_and_a_five_percent_margin(assess):
    maximum = assess()["max_affordable_instalment"]
    assert maximum == 5980.99
    for proposed, expected in [(5980.99, verdict.PASS), (5981.0, verdict.MARGINAL), (6280.03, verdict.MARGINAL),
                               (6280.04, verdict.FAIL)]:
        assert assess(proposed_instalment=proposed)["affordability_verdict_code"] == expected, proposed


def test_asking_for_capacity_without_a_proposal_passes_and_reports_no_margin(assess):
    out = assess(proposed_instalment=None)
    assert out["affordability_verdict_code"] == verdict.PASS
    assert out["max_affordable_instalment"] == 5980.99
    assert math.isnan(out["discretionary_income_after"])


@pytest.mark.parametrize("changes,code", [
    (dict(bureau_as_of_date="2026-08-01"), verdict.BUREAU_STALE),
    (dict(is_joint_application=True, applicant2_employment_type_code=1), verdict.APPLICANT2_INCOME_UNESTABLISHED),
    (dict(applicant1_payslip_income=None, applicant1_declared_income=18500.0), verdict.INCOME_BELOW_MINIMUM_TIER),
    (dict(applicant1_payslip_income=0.0), verdict.NO_NET_INCOME),
])
def test_a_gap_in_the_evidence_is_indeterminate_however_large_the_proposal(assess, changes, code):
    out = assess(proposed_instalment=1_000_000.0, **changes)
    assert out["evidence_sufficiency_code"] == code
    assert out["affordability_verdict_code"] == verdict.INDETERMINATE


def test_the_evidence_cause_reported_is_the_first_in_the_order_an_adjudicator_asks(assess):
    out = assess(bureau_as_of_date="2026-08-01", is_joint_application=True, applicant2_employment_type_code=1)
    assert out["evidence_sufficiency_code"] == verdict.APPLICANT2_INCOME_UNESTABLISHED


@pytest.mark.parametrize("as_of,product,stale", [
    ("2026-09-17", 10, False), ("2026-09-16", 10, True), ("2026-09-10", 40, False), ("2026-09-09", 40, True)])
def test_a_bureau_view_goes_stale_after_seven_days_or_fourteen_for_a_mortgage(assess, as_of, product, stale):
    assert assess(bureau_as_of_date=as_of, product_code=product)["bureau_is_stale"] is stale


def test_no_bureau_view_at_all_is_stale(assess):
    assert assess(bureau_as_of_date=None)["evidence_sufficiency_code"] == verdict.BUREAU_STALE


def test_an_indeterminate_verdict_reports_its_cause_and_never_the_failed_proposal(assess):
    out = assess(bureau_as_of_date="2026-08-01", proposed_instalment=1_000_000.0)
    assert out["decline_reason_codes"] == [204] and out["primary_reason_code"] == 204


def test_a_failed_proposal_reports_that_it_exceeds_capacity(assess):
    out = assess(proposed_instalment=9000.0)
    assert out["affordability_verdict_code"] == verdict.FAIL
    assert out["decline_reason_codes"] == [verdict.R_INSTALMENT_EXCEEDS_CAPACITY]
    assert out["reason_registry_version"] == verdict.REASON_REGISTRY.version


def test_every_evidence_code_has_a_registered_reason():
    assert all(200 + code in verdict.REASON_REGISTRY for code in range(1, 8))


SCENARIOS = [
    {},
    {"risk_grade": 12, "product_code": 40},
    {"applicant1_dependants_count": 5, "risk_grade": 1},
    {"applicant1_payslip_income": 9368.42},                    # gross lands on the R8 900 tax bracket floor
    {"applicant1_payslip_income": 22105.26},                   # and on the R21 000 one
    {"applicant1_bureau_accounts": [{"account_type_code": 2, "balance": 1.0, "instalment": 7813.5,
                                     "months_in_arrears": 0, "opened_date": "2022-01-01", "closed": False,
                                     "is_internal": False}], "applicant1_internal_accounts": []},
]


@pytest.mark.parametrize("changes", SCENARIOS)
def test_a_larger_proposal_never_gets_a_better_verdict(assess, changes):
    outs = [assess(proposed_instalment=float(p), **changes) for p in range(0, 12001, 250)]
    verdicts = [out["affordability_verdict_code"] for out in outs]
    assert verdicts == sorted(verdicts)
    assert len({out["max_affordable_instalment"] for out in outs}) == 1  # the proposal never moves the capacity
    assert verdict.PASS in verdicts


def test_more_existing_debt_never_raises_the_capacity(assess, sample):
    card = sample["applicant1_bureau_accounts"][0]
    capacities = [assess(applicant1_bureau_accounts=[card | {"limit": limit}])["max_affordable_instalment"]
                  for limit in range(0, 400_001, 20_000)]
    assert capacities == sorted(capacities, reverse=True)
    assert capacities[0] > capacities[-1]
