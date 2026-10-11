import copy
from datetime import date

import pytest
from credit_core.adjustments import Effect, Overlay

from assessment import capacity, documents, verdict

# A vehicle loan whose R7 813.50 instalment leaves exactly R2 000 of the sample's discretionary income.
HEAVY_DEBT = {"applicant1_bureau_accounts": [
    {"account_type_code": 2, "balance": 200000.0, "instalment": 7813.5, "months_in_arrears": 0,
     "opened_date": "2022-01-01", "closed": False, "is_internal": False}], "applicant1_internal_accounts": []}
OVERLAY = "affordability_assessment", "capacity", "overlay", "apply_max_affordable_instalment_adjustments"


def test_the_buffer_binds_when_it_leaves_less_than_the_floor_does(assess):
    out = assess()
    assert out["affordability_binding_constraint_code"] == capacity.BUFFER_BOUND


def test_the_residual_floor_binds_when_discretionary_income_is_small(assess):
    out = assess(**HEAVY_DEBT)
    assert out["discretionary_income"] == 2000.0
    # buffer alone would allow 2 000 x (1 - 19.09%) = 1 618.20; the floor leaves 2 000 - 1 400 = 600
    assert out["affordability_binding_constraint_code"] == capacity.RESIDUAL_FLOOR_BOUND
    assert out["max_affordable_instalment_unadjusted"] == 600.0
    assert out["max_affordable_instalment"] == 540.0


def test_an_overcommitted_applicant_has_negative_discretionary_income_and_no_capacity(assess):
    out = assess(applicant1_bureau_accounts=[HEAVY_DEBT["applicant1_bureau_accounts"][0] | {"instalment": 20000.0}],
                 proposed_instalment=1.0)
    assert out["discretionary_income"] < 0
    assert out["max_affordable_instalment"] == 0.0
    assert out["affordability_verdict_code"] == verdict.FAIL


@pytest.mark.parametrize("grade,product,buffer", [(1, 10, 0.10), (12, 10, 0.35), (12, 40, 0.40), (5, 11, 0.2009)])
def test_the_buffer_rises_with_risk_grade_and_term(assess, grade, product, buffer):
    assert assess(risk_grade=grade, product_code=product)["affordability_buffer_applied"] == buffer


def test_an_unlisted_product_gets_the_most_conservative_buffer_and_no_cell(assess):
    out = assess(product_code=99)
    assert out["affordability_buffer_applied"] == 0.35 and out["buffer_grid_cell_id"] is None


@pytest.mark.parametrize("dependants,floor", [(0, 800.0), (1, 1100.0), (3, 1700.0), (5, 2300.0), (8, 2300.0)])
def test_the_residual_floor_rises_with_dependants(assess, dependants, floor):
    assert assess(applicant1_dependants_count=dependants)["residual_floor_amount"] == floor


def test_an_overlay_outside_its_product_scope_changes_nothing(assess):
    out = assess(product_code=11)
    assert out["adjustments_applied"] == []
    assert out["max_affordable_instalment"] == out["max_affordable_instalment_unadjusted"]


def test_the_overlay_stack_can_be_switched_off_through_the_same_pipeline(assess, params):
    off = copy.deepcopy(params)
    node = off
    for key in OVERLAY:
        node = node[key]
    node["adjustment_stack_enabled"] = False
    out = assess(params_document=off)
    assert out["adjustments_applied"] == []
    assert out["max_affordable_instalment"] == out["max_affordable_instalment_unadjusted"] == 6645.54


def test_an_overlay_that_would_raise_capacity_is_rejected_when_it_is_defined():
    with pytest.raises(ValueError, match="loosens"):
        Overlay(adjustment_id="ADJ-X", kind="cap_adjustment", target="max_affordable_instalment",
                effect=Effect(op="multiply", value=1.1), scope={}, stack_position=1, owner="Credit Risk Policy",
                approval_reference="CRC-X", rationale="more lending", effective_from=date(2026, 1, 1),
                review_date=date(2026, 12, 31), tighten_only=True)


def test_the_shipped_overlays_only_tighten():
    stack = documents.shipped("capacity_adjustments")
    assert stack.adjustments and all(a.tighten_only and a.effect.value <= 1.0 for a in stack.adjustments)
