"""The sample application, worked by hand: every figure below can be re-derived from the sample's own numbers."""
import json
from pathlib import Path

from assessment import capacity, verdict
from assessment.pipeline import build

ROOT = Path(__file__).resolve().parents[1]


def test_the_sample_application_is_worked_to_the_cent(assess):
    out = assess()
    assert out["gross_monthly_income"] == 17575.0          # R18 500 payslip less its 5% haircut
    assert out["statutory_deductions"] == 1561.5           # 18% of the R8 675 above the R8 900 bracket floor
    assert out["net_monthly_income"] == 16013.5
    assert out["living_expenses"] == 6200.0                # declared, above both norms (R3 500 statutory, R3 920 internal)
    assert out["existing_obligations"] == 1600.0           # R1 150 stated instalment + 3% of the R15 000 card limit
    assert out["discretionary_income"] == 8213.5
    assert out["affordability_buffer_applied"] == 0.1909   # grade 5, product 10
    assert out["max_affordable_instalment_unadjusted"] == 6645.54
    assert out["max_affordable_instalment"] == 5980.99     # the product 10 overlay takes 10% off
    assert out["adjustments_applied"] == ["ADJ-02-2026-001"]
    assert (out["affordability_verdict_code"], out["evidence_sufficiency_code"]) == (verdict.PASS, 0)
    assert out["discretionary_income_after"] == 5713.5
    assert out["decline_reason_codes"] == [] and out["primary_reason_code"] is None


def test_every_table_read_is_recorded_with_its_version_and_cell(assess):
    out = assess()
    assert out["applicant1_tax_table_version"] == "tax@2026"
    assert out["applicant1_tax_table_cell_id"] == "tax_table@tax@2026#1"
    assert out["statutory_norm_cell_id"] == "expense_norms.statutory@norms.statutory@2025#10"
    assert out["internal_norm_cell_id"] == "expense_norms.internal@norms.internal@2026#10|10"
    assert out["buffer_grid_cell_id"] == f"buffer_grid@{capacity.BUFFER_GRID_VERSION}#10|5"
    assert out["residual_floor_cell_id"] == f"residual_floor@{capacity.RESIDUAL_FLOOR_VERSION}#2"
    assert out["adjustment_set_id"] == capacity.ADJUSTMENT_SET_ID


def test_the_ladder_to_discretionary_income_adds_up(assess):
    out = assess(court_ordered_deductions=500.0)
    assert out["discretionary_income"] == round(
        out["net_monthly_income"] - out["living_expenses"] - out["court_ordered_deductions"]
        - out["existing_obligations"], 2)
    assert out["discretionary_income"] == 7713.5


def test_a_decision_is_replayed_with_the_tables_in_force_on_its_own_date(assess):
    replay = assess(decision_date="2025-06-01", bureau_as_of_date="2025-05-28")
    assert replay["applicant1_tax_table_version"] == "tax@2025"
    assert replay["statutory_deductions"] == 1633.5        # 18% of R9 075 above that year's R8 500 floor
    assert replay["adjustments_applied"] == []             # the overlay was not yet in force
    assert replay["max_affordable_instalment"] == replay["max_affordable_instalment_unadjusted"]


def test_the_expense_norms_change_on_the_day_they_are_gazetted(assess):
    before, after = assess(decision_date="2026-09-30", bureau_as_of_date="2026-09-28"), \
        assess(decision_date="2026-10-01", bureau_as_of_date="2026-09-28")
    assert "statutory@2025" in before["statutory_norm_cell_id"]
    assert "statutory@2026" in after["statutory_norm_cell_id"]


def test_the_same_inputs_give_the_same_answer_every_time(assess):
    assert assess() == assess()


def test_the_params_document_on_disk_is_the_pipelines_defaults(params):
    assert params == build().parameters().defaults()
