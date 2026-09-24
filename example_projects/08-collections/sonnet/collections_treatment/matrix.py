"""The treatment matrix (spec 08 §5.4) -- this slice's dominant-difficulty table.

8 arrears buckets x 6 collections bands x 7 balance bands x 4 contact bands
x 4 product families = 5 376 cells, generated at full declared size (SCOPE.md
rule 1: "generating a 63 360-cell grid is cheap ... do not shrink them" --
this project's one dominant-difficulty artefact is this matrix, kept at
full size the way project 00 kept the Flex Loan rate card at 96x55x12).
Every cell carries `treatment_code` plus the three modifiers (§5.4 table).

Matrix overlays (§5.4's three kinds -- intensity dial, suppression/
substitution, allocation weighting dial) are a *second* `AdjustmentRegister`,
independent of `scoring.SCORE_ADJUSTMENTS` (08 §6: "independently versioned
and independently expiring" -- `matrix_version` and the overlay set are
never the same artefact). Allocation weighting is consumed downstream by
`capacity_alloc.py`, not applied here -- it does not change any of the four
cell values, only ranking weight.
"""
from __future__ import annotations

from datetime import date

from decider import param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.evidence import cell_id as _cell_id

from collections_treatment import vocab
from collections_treatment.scoring import assert_no_statutory_target

MATRIX_VERSION = "matrix-2026.09"

_BUCKETS = range(1, 9)
_BANDS = range(1, 7)
_BALANCE_BANDS = range(1, 8)
_CONTACT_BANDS = range(1, 5)
_PRODUCT_FAMILIES = range(1, 5)

# A small, deterministic rule standing in for the analyst-authored spreadsheet: intensity
# rises with bucket and band, tempered by balance (small balances get automated channels
# for longer) and contact responsiveness (silent/unreachable escalate faster to agent/field).
_TREATMENT_BY_SEVERITY = {
    0: vocab.NO_ACTION, 1: vocab.AUTOMATED_SMS, 2: vocab.AUTOMATED_EMAIL, 3: vocab.IN_APP_MESSAGE,
    4: vocab.AUTOMATED_VOICE_MESSAGE, 5: vocab.AGENT_CALL_LOW, 6: vocab.AGENT_CALL_STANDARD,
    7: vocab.AGENT_CALL_HIGH, 8: vocab.FIELD_VISIT, 9: vocab.AGENCY_HANDOVER,
    10: vocab.PRE_LEGAL_NOTICE, 11: vocab.LEGAL_HANDOVER,
}


def _cell(bucket: int, band: int, balance_band: int, contact_band: int, product_family: int) -> dict:
    severity = (bucket - 1) + (band - 1) // 2 + (contact_band - 1)
    if balance_band <= 1 and bucket <= 3:
        severity = max(0, severity - 1)  # small, early balances: stay automated longer
    if bucket >= 8:
        severity = max(severity, 10)  # 365+ days: pre-legal or legal, never automated-only
    severity = min(severity, 11)
    treatment_code = _TREATMENT_BY_SEVERITY[severity]
    intensity = min(5, max(1, 1 + severity // 2))
    permitted_retries = max(0, 4 - severity // 3)
    cooling_off_days = min(21, 1 + severity * 2)
    return {
        "bucket": bucket, "band": band, "balance_band": balance_band,
        "contact_band": contact_band, "product_family": product_family,
        "treatment_code": treatment_code, "treatment_intensity": intensity,
        "permitted_retries": permitted_retries, "cooling_off_days": cooling_off_days,
        "cell_id": _cell_id("treatment_matrix", MATRIX_VERSION, bucket, band, balance_band,
                             contact_band, product_family),
    }


def build_treatment_matrix() -> DecisionTableConfig:
    """5 376 cells (8x6x7x4x4), generated. §4.4's sparsity note (41% of cells see fewer
    than 50 accounts a month, 12% see none) is a traffic property, not a construction
    property -- every cell still exists and is independently addressable and diffable."""
    rows = [
        _cell(bucket, band, balance_band, contact_band, product_family)
        for bucket in _BUCKETS for band in _BANDS for balance_band in _BALANCE_BANDS
        for contact_band in _CONTACT_BANDS for product_family in _PRODUCT_FAMILIES
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "treatment_matrix",
        "columns": {
            "bucket": "Int64", "band": "Int64", "balance_band": "Int64", "contact_band": "Int64",
            "product_family": "Int64", "treatment_code": "Int64", "treatment_intensity": "Int64",
            "permitted_retries": "Int64", "cooling_off_days": "Int64", "cell_id": "String",
        },
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "arrears_bucket_code", "value_column": "bucket"},
            {"type": "eq", "variable": "collections_band_code", "value_column": "band"},
            {"type": "eq", "variable": "balance_band_code", "value_column": "balance_band"},
            {"type": "eq", "variable": "contact_band_code", "value_column": "contact_band"},
            {"type": "eq", "variable": "product_family_code", "value_column": "product_family"},
        ]},
        "outputs": ["treatment_code", "treatment_intensity", "permitted_retries", "cooling_off_days", "cell_id"],
        "default": [vocab.NO_ACTION, 1, 0, 21, None],
    }).relabel(writes={
        "treatment_code": "matrix_treatment_code", "treatment_intensity": "matrix_treatment_intensity",
        "permitted_retries": "matrix_permitted_retries", "cooling_off_days": "matrix_cooling_off_days",
        "cell_id": "matrix_cell_id",
    })


# --- Matrix overlays (§5.4: intensity dial, suppression/substitution, allocation weight) ---

MATRIX_ADJUSTMENT_SET_ID = "AS-08-MATRIX-2026.09"
MATRIX_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-08-MATRIX-001", kind="cap_adjustment", target="matrix_treatment_intensity",
        effect=AdjustmentEffect("add", 1.0), scope={"arrears_bucket_code": 3}, stack_position=1,
        owner="Collections Strategy", approval_reference="CCF-2026-033",
        rationale="Escalate bucket 3 by one intensity level for six weeks",
        effective_from=date(2026, 8, 1), effective_to=date(2026, 9, 12), review_date=date(2026, 9, 5),
    ),
    Adjustment(
        adjustment_id="ADJ-08-MATRIX-002", kind="treatment_suppression", target="matrix_treatment_code",
        effect=AdjustmentEffect("set", float(vocab.NO_ACTION)), scope={"matrix_treatment_code": vocab.FIELD_VISIT},
        stack_position=2, owner="Collections Strategy", approval_reference="CCF-2026-041",
        rationale="Suspend field visits for the quarter", effective_from=date(2026, 7, 1),
        effective_to=date(2026, 10, 1), review_date=date(2026, 9, 15),
    ),
])
assert_no_statutory_target(MATRIX_ADJUSTMENTS)


def _intensity_dial(
    matrix_treatment_intensity: int, decision_date: date,
    arrears_bucket_code: int,
    adjustment_stack_enabled: bool = param(True),
) -> tuple[int, list[str]]:
    provenance = {"arrears_bucket_code": arrears_bucket_code}
    result = MATRIX_ADJUSTMENTS.apply_stack(
        "matrix_treatment_intensity", float(matrix_treatment_intensity), provenance, decision_date,
        MATRIX_ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    return int(min(5, max(1, round(result.adjusted_value)))), list(result.adjustments_applied)


intensity_dial_step = step(_intensity_dial, outputs=("treatment_intensity", "intensity_dial_adjustments_applied"))


def _treatment_suppression(
    matrix_treatment_code: int, decision_date: date,
    adjustment_stack_enabled: bool = param(True),
) -> tuple[int, list[str]]:
    """A treatment suppressed by an overlay falls back to no action (§5.4: "optionally
    falling back to a named alternative" -- this slice's fallback is always no-action; a
    named per-scope substitute is noted in NOTES.md "What I would do next")."""
    provenance = {"matrix_treatment_code": matrix_treatment_code}
    result = MATRIX_ADJUSTMENTS.apply_stack(
        "matrix_treatment_code", float(matrix_treatment_code), provenance, decision_date,
        MATRIX_ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    suppressed = bool(result.adjustments_applied)
    return int(result.adjusted_value), list(result.adjustments_applied)


treatment_suppression_step = step(
    _treatment_suppression, outputs=("treatment_code", "suppression_adjustments_applied"),
)
