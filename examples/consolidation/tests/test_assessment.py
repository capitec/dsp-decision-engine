"""The affordability assessment is reused, not forked: one evidence pass, the capacity half per scenario, and the same answers as the whole."""
import copy
from datetime import date

import pytest
from assessment.pipeline import build as build_assessment

from consolidation.capacity import as_assessment_accounts
from conftest import parse_dates

WHOLE = build_assessment().compile("interpreted")


def whole_assessment(request: dict, retained_refs, product_code: int, proposed_instalment: float | None = None) -> dict:
    """The full assessment of the client holding only `retained_refs`, as a new application at `product_code` would see it."""
    record = parse_dates(request)
    bureau, internal = as_assessment_accounts([a for a in record["accounts"] if a["account_ref"] in retained_refs], {})
    keep = ("decision_date", "channel_code", "segment_code", "risk_grade", "bureau_as_of_date", "court_ordered_deductions",
            "is_joint_application")
    assessed = {k: record[k] for k in keep} | {k: v for k, v in record.items() if k.startswith("applicant1_")}
    assessed |= {"assessment_mode_code": 4, "product_code": product_code,
                 "applicant1_bureau_accounts": bureau, "applicant1_internal_accounts": internal}
    if proposed_instalment is not None:
        assessed["proposed_instalment"] = proposed_instalment
    return WHOLE.score(assessed, WHOLE_PARAMS)


WHOLE_PARAMS = build_assessment().parameters().defaults()
FIGURES = ("gross_monthly_income", "net_monthly_income", "living_expenses", "statutory_deductions", "income_haircut_applied",
           "expense_basis_code")


def test_the_income_and_expense_determination_equals_the_assessments_and_is_made_once(assess, sample):
    out = assess()
    whole = whole_assessment(sample, {a["account_ref"] for a in sample["accounts"]}, 11)
    assert {name: out["invariants"][name] for name in FIGURES} == {name: whole[name] for name in FIGURES}
    assert out["invariants"]["net_monthly_income"] == 35_620.0 and out["invariants"]["living_expenses"] == 5_450.0


def test_the_baseline_capacity_equals_the_whole_assessment_of_the_unreduced_inventory(assess, sample):
    out = assess()
    whole = whole_assessment(sample, {a["account_ref"] for a in sample["accounts"]}, 11)
    base = out["baseline"]
    assert (base["existing_obligations"], base["discretionary_income"], base["max_affordable_instalment"]) == (
        whole["existing_obligations"], whole["discretionary_income"], whole["max_affordable_instalment"])


def test_every_priced_scenario_has_the_capacity_of_the_whole_assessment_run_on_what_remains(assess, sample):
    out = assess()
    checked = {11: 0, 20: 0}
    for s in out["scenarios"]:
        if "offered_amount" not in s or checked[s["product_code"]] >= 4:
            continue
        retained = {a["account_ref"] for a in sample["accounts"]} - set(s["settlement_set"])
        whole = whole_assessment(sample, retained, s["product_code"], s["tested_payment"])
        assert (s["existing_obligations_after"], s["discretionary_income"], s["max_affordable_instalment"],
                s["affordability_verdict_code"]) == (whole["existing_obligations"], whole["discretionary_income"],
                                                     whole["max_affordable_instalment"], whole["affordability_verdict_code"])
        checked[s["product_code"]] += 1
    assert checked == {11: 4, 20: 4}


def test_a_scenario_is_affordable_only_if_the_assessment_says_pass(assess):
    out = assess()
    for s in out["scenarios"]:
        if s.get("affordability_verdict_code") != 1:
            assert 1102 in s["rejection_reason_codes"]


def test_the_buffer_is_the_only_figure_that_depends_on_the_product(assess):
    out = assess()
    flex, card = (next(s for s in out["scenarios"] if s["product_code"] == p and s["settlement_set"] == [2, 4, 7]) for p in (11, 20))
    assert flex["existing_obligations_after"] == card["existing_obligations_after"]
    assert flex["max_affordable_instalment"] != card["max_affordable_instalment"]
