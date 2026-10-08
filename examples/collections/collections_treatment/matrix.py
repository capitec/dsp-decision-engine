"""The treatment matrix: 8 buckets x 6 bands x 7 balance bands x 4 contact bands x 4 product families = 5 376 cells.

The matrix is a config document (`configs/<version>/treatment_matrix.json`) so Collections Strategy can issue a new
one without a release. Two overlays work in the month: an intensity dial and a treatment suppression.
"""
from __future__ import annotations

from datetime import date

from decider import DecisionTableConfig, allow_fallback, param, step

from credit_core.adjustments import Adjustment, AdjustmentEffect
from credit_core.evidence import cell_id

from collections_treatment import overlays, vocab as v

MATRIX_VERSION = "matrix-2026.09"
ADJUSTMENT_SET_ID = "AS-COLL-MATRIX-2026.09"

_TREATMENT_BY_SEVERITY = [
    v.NO_ACTION, v.AUTOMATED_SMS, v.AUTOMATED_EMAIL, v.IN_APP_MESSAGE, v.AUTOMATED_VOICE_MESSAGE, v.AGENT_CALL_LOW,
    v.AGENT_CALL_STANDARD, v.AGENT_CALL_HIGH, v.FIELD_VISIT, v.AGENCY_HANDOVER, v.PRE_LEGAL_NOTICE, v.LEGAL_HANDOVER,
]

ADJUSTMENTS = overlays.register([
    Adjustment(
        adjustment_id="ADJ-COLL-MATRIX-001", kind="cap_adjustment", target="matrix_treatment_intensity",
        effect=AdjustmentEffect("add", 1.0), scope={"arrears_bucket_code": 3}, stack_position=1,
        owner="Collections Strategy", approval_reference="CCF-2026-033",
        rationale="Escalate bucket 3 by one intensity level for six weeks",
        effective_from=date(2026, 8, 1), effective_to=date(2026, 9, 12), review_date=date(2026, 9, 5)),
    Adjustment(
        adjustment_id="ADJ-COLL-MATRIX-002", kind="treatment_suppression", target="matrix_treatment_code",
        effect=AdjustmentEffect("set", float(v.NO_ACTION)), scope={"matrix_treatment_code": v.FIELD_VISIT},
        stack_position=2, owner="Collections Strategy", approval_reference="CCF-2026-041",
        rationale="Suspend field visits for the quarter",
        effective_from=date(2026, 7, 1), effective_to=date(2026, 10, 1), review_date=date(2026, 9, 15)),
])


def generate_matrix() -> dict:
    """A synthetic matrix document: intensity rises with bucket, band and contact band; small early balances stay
    automated longer; bucket 8 goes to a notice or legal handover.

    Example: `lookup(DecisionTableConfig.load(generate_matrix()))` is the step `build` takes.
    """
    rows = []
    for bucket in range(1, 9):
        for band in range(1, 7):
            for balance_band in range(1, 8):
                for contact_band in range(1, 5):
                    severity = (bucket - 1) + (band - 1) // 2 + (contact_band - 1)
                    if balance_band == 1 and bucket <= 3:
                        severity = max(0, severity - 1)
                    severity = min(11, max(severity, 10) if bucket == 8 else severity)
                    for family in range(1, 5):
                        rows.append({
                            "bucket": bucket, "band": band, "balance_band": balance_band,
                            "contact_band": contact_band, "product_family": family,
                            "treatment_code": _TREATMENT_BY_SEVERITY[severity],
                            "treatment_intensity": min(5, 1 + severity // 2),
                            "permitted_retries": max(0, 4 - severity // 3),
                            "cooling_off_days": min(21, 1 + severity * 2)})
    return {
        "type": "decision_table", "name": "treatment_matrix",
        "columns": {"bucket": "Int64", "band": "Int64", "balance_band": "Int64", "contact_band": "Int64",
                    "product_family": "Int64", "treatment_code": "Int64", "treatment_intensity": "Int64",
                    "permitted_retries": "Int64", "cooling_off_days": "Int64"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "arrears_bucket_code", "value_column": "bucket"},
            {"type": "eq", "variable": "collections_band_code", "value_column": "band"},
            {"type": "eq", "variable": "balance_band_code", "value_column": "balance_band"},
            {"type": "eq", "variable": "contact_band_code", "value_column": "contact_band"},
            {"type": "eq", "variable": "product_family_code", "value_column": "product_family"}]},
        "outputs": ["treatment_code", "treatment_intensity", "permitted_retries", "cooling_off_days"],
        "default": [v.NO_ACTION, 1, 0, 21],
    }


def lookup(matrix: DecisionTableConfig):
    """The matrix document as a step writing `matrix_treatment_code`, `matrix_treatment_intensity`,
    `matrix_permitted_retries` and `matrix_cooling_off_days`."""
    return matrix.relabel(writes={name: f"matrix_{name}" for name in matrix.outputs})


@allow_fallback
def matrix_cell_id(arrears_bucket_code: int, collections_band_code: int, balance_band_code: int,
                   contact_band_code: int, product_family_code: float) -> str:
    return cell_id("treatment_matrix", MATRIX_VERSION, arrears_bucket_code, collections_band_code, balance_band_code,
                   contact_band_code, int(product_family_code))


@allow_fallback
def intensity_dial(
    matrix_treatment_intensity: int, decision_date: date, arrears_bucket_code: int,
    adjustment_stack_enabled: bool = param(True),
) -> tuple[int, list[str]]:
    result = ADJUSTMENTS.apply_stack(
        "matrix_treatment_intensity", float(matrix_treatment_intensity), {"arrears_bucket_code": arrears_bucket_code},
        decision_date, ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled)
    return int(min(5, max(1, round(result.adjusted_value)))), list(result.adjustments_applied)


@allow_fallback
def treatment_suppression(
    matrix_treatment_code: int, decision_date: date, adjustment_stack_enabled: bool = param(True),
) -> tuple[int, list[str]]:
    """A suppressed treatment falls back to no action."""
    result = ADJUSTMENTS.apply_stack(
        "matrix_treatment_code", float(matrix_treatment_code), {"matrix_treatment_code": matrix_treatment_code},
        decision_date, ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled)
    return int(result.adjusted_value), list(result.adjustments_applied)


intensity_dial_step = step(intensity_dial, outputs=("treatment_intensity", "intensity_dial_adjustments_applied"))
treatment_suppression_step = step(
    treatment_suppression, outputs=("overlaid_treatment_code", "suppression_adjustments_applied"))
