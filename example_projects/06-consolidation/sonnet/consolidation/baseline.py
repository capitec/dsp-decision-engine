"""Stage 5.4 -- baseline assessment and the short-circuit to project 03 (spec 06 §5.4).

Calls project 02's pipeline **once** per assessment (§5.6.1's invariant income and
expense determination), over the full, unreduced inventory -- this is also where
`consolidation.scenario_eval.EvidenceUnit` is built, so every later scenario reads
the identical income/expense figures rather than re-deriving them.

**The short-circuit.** If the plain request already passes affordability cleanly
(§5.4's six-condition test), consolidation is unnecessary and the flow "should
complete as plain granting under project 03" -- so this module calls project 03's
own pipeline (`consolidation.reuse.load_project03_pipeline`) for the hand-off,
rather than re-implementing granting. 03's request shape carries ~50 fields this
project's own request does not (scorecard characteristics, fraud/DQ inputs,
exclusion lists); an existing consolidation client is, by construction, already
known to the Bank (§5.1: "existing client only... this flow does not serve
new-to-bank"), so the fields 06 does not itself carry are defaulted to a
benign-pass value rather than re-derived -- a declared scope simplification, not
an oversight (NOTES.md "What I left out").
"""
from __future__ import annotations

from datetime import date

from decider import Engine

from credit_core.instalment import instalment_before_fees

from consolidation import reuse
from consolidation.scenario_eval import BaselineFacts, EvidenceUnit, compute_baseline_facts

REFERENCE_TERM_MONTHS = 36  # the short-circuit's own indicative term (§5.4 does not pin one -- see NOTES.md "Spec problems")
INDICATIVE_RATE = 0.20      # a flat stand-in for "the rate the client would achieve on a plain advance"
RATE_GAP_THRESHOLD_BPS = 0.03  # 300 basis points
DSR_CEILING = 0.40
ACTIVE_ACCOUNTS_THRESHOLD = 6
HIGH_RATE_THRESHOLD = 0.28
AFFORDABILITY_BUFFER = 0.12

_EMPTY_ACCOUNT_LIST = [{
    "account_type_code": 31, "balance": 0.0, "limit": 0.0, "instalment": 0.0, "months_in_arrears": 0,
    "opened_date": date(2020, 1, 1), "closed": True, "is_internal": False,
}]


def _to_obligations_accounts(accounts: list[dict]) -> tuple[list[dict], list[dict]]:
    from consolidation.scenario_eval import _to_obligations_shape
    bureau = [_to_obligations_shape(a) for a in accounts if not a.get("is_internal")]
    internal = [_to_obligations_shape(a) for a in accounts if a.get("is_internal")]
    return bureau or list(_EMPTY_ACCOUNT_LIST), internal or list(_EMPTY_ACCOUNT_LIST)


def _to_02_request(row: dict) -> dict:
    bureau_accounts, internal_accounts = _to_obligations_accounts(row.get("accounts") or [])
    for acc_list in (bureau_accounts, internal_accounts):
        for a in acc_list:
            a.setdefault("opened_date", row["decision_date"])
    return {
        "decision_id": row["decision_id"], "decision_date": row["decision_date"],
        "product_code": row.get("product_code_hint", 11), "channel_code": row["channel_code"],
        "segment_code": row.get("segment_code", 1), "assessment_mode_code": 1, "is_joint_application": False,
        "risk_grade": row["risk_grade"],
        "applicant1_dependants_count": row.get("dependants_count", 0),
        "applicant1_employment_type_code": row.get("employment_type_code", 1),
        "applicant1_payslip_income": row.get("payslip_income") or 0.0,
        "applicant1_variable_pay_history": row.get("variable_pay_history") or [0.0],
        "applicant1_declared_expenses": row.get("declared_expenses") or {},
        "applicant1_statement_expenses": row.get("statement_expenses") or {},
        "applicant1_bureau_accounts": bureau_accounts, "applicant1_internal_accounts": internal_accounts,
        "applicant2_dependants_count": 0, "applicant2_employment_type_code": 0,
        "applicant2_variable_pay_history": [0.0], "applicant2_declared_expenses": {},
        "applicant2_statement_expenses": {}, "applicant2_bureau_accounts": list(_EMPTY_ACCOUNT_LIST),
        "applicant2_internal_accounts": list(_EMPTY_ACCOUNT_LIST),
        "bureau_as_of_date": row.get("bureau_as_of_date") or row["decision_date"],
        "court_ordered_deductions": row.get("court_ordered_deductions") or 0.0,
    }


_PIPELINE_02 = None


def _engine02():
    global _PIPELINE_02
    if _PIPELINE_02 is None:
        _PIPELINE_02 = reuse.load_project02_pipeline()
    return Engine().bind(_PIPELINE_02.build(), mode="interpreted")


def build_evidence_and_baseline(row: dict) -> tuple[EvidenceUnit, BaselineFacts, dict]:
    """One call to project 02 (no `proposed_instalment`: shape (b), capacity only) for
    the invariant income/expense/obligations figures every scenario shares."""
    request = _to_02_request(row)
    out = _engine02().score(request)
    evidence = EvidenceUnit(
        net_monthly_income=out["net_monthly_income"], living_expenses=out["living_expenses"],
        dependants_count=row.get("dependants_count", 0),
        court_ordered_deductions=row.get("court_ordered_deductions") or 0.0,
        risk_grade=row["risk_grade"], applicant_age_years=row.get("applicant_age_years", 40.0),
        evidence_sufficiency_code=out["evidence_sufficiency_code"],
    )
    baseline_facts = compute_baseline_facts(row.get("accounts") or [])
    return evidence, baseline_facts, out


def evaluate_short_circuit(
    row: dict, evidence: EvidenceUnit, baseline: BaselineFacts, out02: dict, achievable_consolidation_rate: float,
) -> dict:
    """§5.4's six-condition test. Returns the verdict plus every condition's own
    evaluation (09 §5.15 item 14: recorded, not only the outcome)."""
    requested_amount = row.get("requested_amount")
    conditions = {}

    if requested_amount:
        indicative_instalment = round(instalment_before_fees(requested_amount, REFERENCE_TERM_MONTHS, INDICATIVE_RATE), 2)
        discretionary_after = out02["discretionary_income"] - indicative_instalment
        conditions["request_affordable"] = (
            out02["max_affordable_instalment"] >= indicative_instalment and discretionary_after >= 0
        )
        post_advance_dsr = (out02["existing_obligations"] + indicative_instalment) / evidence.net_monthly_income \
            if evidence.net_monthly_income else 1.0
    else:
        indicative_instalment = 0.0
        conditions["request_affordable"] = True
        post_advance_dsr = out02["existing_obligations"] / evidence.net_monthly_income if evidence.net_monthly_income else 1.0

    conditions["dsr_within_ceiling"] = post_advance_dsr <= DSR_CEILING
    accounts = row.get("accounts") or []
    conditions["no_account_in_arrears"] = all((a.get("months_in_arrears") or 0) == 0 for a in accounts)
    conditions["rate_gap_within_threshold"] = (
        baseline.weighted_average_rate - achievable_consolidation_rate
    ) <= RATE_GAP_THRESHOLD_BPS
    conditions["fewer_than_threshold_accounts"] = baseline.account_count < ACTIVE_ACCOUNTS_THRESHOLD
    conditions["no_high_rate_settleable_account"] = not any(
        (a.get("nominal_annual_rate") or 0) >= HIGH_RATE_THRESHOLD for a in accounts
        if (row.get("settleable_account_refs") or []).count(a.get("account_ref")) > 0
    )

    consolidation_unnecessary = all(conditions.values())
    return {
        "short_circuit_conditions": conditions,
        "short_circuit_applies": consolidation_unnecessary,
        "indicative_plain_instalment": indicative_instalment,
        "post_advance_debt_service_ratio": round(post_advance_dsr, 4),
    }


_PIPELINE_03 = None


def _engine03(rate_card_flex_loan):
    global _PIPELINE_03
    if _PIPELINE_03 is None:
        _PIPELINE_03 = reuse.load_project03_pipeline()
    return Engine().bind(_PIPELINE_03.build(rate_card_flex_loan), mode="interpreted")


def _to_03_request(row: dict, evidence: EvidenceUnit) -> dict:
    """Fields 06 carries, mapped directly; fields it does not carry (an existing
    client's own origination-time checks: fraud, DQ, exclusion lists, prior
    applications) defaulted to a benign pass -- see module docstring."""
    bureau_accounts, internal_accounts = _to_obligations_accounts(row.get("accounts") or [])
    return {
        "decision_id": row["decision_id"], "decision_date": row["decision_date"], "product_code": 10,
        "channel_code": row["channel_code"], "purpose_code": 1,
        "requested_amount": row.get("requested_amount") or 0.0, "requested_term_months": REFERENCE_TERM_MONTHS,
        "is_joint_application": False, "campaign_id": 0, "campaign_is_preapproved": False,
        "applicant_age_years": row.get("applicant_age_years", 40.0),
        "employment_type_code": row.get("employment_type_code", 1), "months_employed": 60,
        "employer_id": 1, "residency_code": 1, "credit_life_substitution_declared": False,
        "has_contractual_capacity": True, "dependants_count": row.get("dependants_count", 0),
        "internal_tenure_months": row.get("client_tenure_months", 132), "internal_exposure_total": 0.0,
        "group_exposure_limit": 1_000_000.0, "in_flight_application_count": 0,
        "debt_review_status_code": 0, "administration_order_flag": False, "insolvency_status_code": 0,
        "insolvency_rehabilitated_months": 0, "deceased_flag": False, "estate_flag": False,
        "exclusion_list_hits": [0], "is_duplicate_application": False, "has_adverse_internal_history": False,
        "marketing_opt_out": False, "is_regulated_notice": True, "data_sharing_consent": True,
        "fraud_verdict_code": 1, "fraud_response_ms": 50, "bureau_response_status": 1,
        "bureau_account_count": len(row.get("accounts") or []), "bureau_history_months": 60,
        "bureau_identity_subject_count": 1, "bureau_has_unparseable_account": False,
        "bureau_account_list_truncated": False, "bureau_has_out_of_domain_status": False,
        "bureau_identity_mismatch": False, "bureau_as_of_date": row.get("bureau_as_of_date") or row["decision_date"],
        "bureau_enquiry_count_60d": 0, "bureau_score": 650, "worst_arrears_months": 0,
        "accounts_in_arrears_count": 0, "revolving_utilisation": 0.3,
        "payslip_income": row.get("payslip_income") or 0.0, "variable_pay_history": row.get("variable_pay_history") or [0.0],
        "declared_expenses": row.get("declared_expenses") or {}, "statement_expenses": row.get("statement_expenses") or {},
        "bureau_accounts": bureau_accounts, "internal_accounts": internal_accounts,
        "applicant2_dependants_count": 0, "applicant2_employment_type_code": 0,
        "applicant2_variable_pay_history": [0.0], "applicant2_declared_expenses": {},
        "applicant2_statement_expenses": {}, "applicant2_bureau_accounts": list(_EMPTY_ACCOUNT_LIST),
        "applicant2_internal_accounts": list(_EMPTY_ACCOUNT_LIST),
        "court_ordered_deductions": row.get("court_ordered_deductions") or 0.0,
    }


def plain_grant_via_project03(row: dict, evidence: EvidenceUnit, rate_card_flex_loan) -> dict:
    request = _to_03_request(row, evidence)
    return _engine03(rate_card_flex_loan).score(request)
