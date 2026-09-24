"""Stage 8 -- the combined business grade (spec 05 §5.10), a reduced two-component
blend (financial, people -- the behavioural and qualitative-override components are
out of this slice) and a **reduced overlay stack**: one business-level PD multiplier
(stack position, spec 05's own numbering, "6" -- the sector PD multiplier), on top of
the entity-level overlay already applied inside `scoring.py` (position "2") and the
event-threshold overlay in `events.py` (position "1"). Three of spec 05 §5.10's
eleven declared positions, not all eleven (SCOPE.md: "a reduced overlay stack").

This is a `DecisionTableConfig` used directly in the *outer* application-level dag
(not through a nested `Engine`, unlike every table in `scoring.py`/`events.py`) --
because the combined grade runs once per application, not once per entity, an
ordinary top-level table composition is all it needs.
"""
from __future__ import annotations

from datetime import date

from decider import dag, missing_as, param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.evidence import cell_id as _cell_id

from business_nested import vocab

BUSINESS_RISK_GRADE_VERSION = "biz-rg-2026.09"

# product_code -> 11 ascending PD cut points. Revolving (51) is one notch stricter
# than term (50) at every boundary -- illustrative, per this repo's front matter.
_BUSINESS_BOUNDARIES = {
    vocab.PRODUCT_BUSINESS_TERM_FACILITY:
        [0.008, 0.016, 0.028, 0.045, 0.07, 0.10, 0.14, 0.19, 0.26, 0.35, 0.48],
    vocab.PRODUCT_BUSINESS_REVOLVING_FACILITY:
        [0.006, 0.013, 0.023, 0.038, 0.06, 0.088, 0.125, 0.17, 0.235, 0.32, 0.44],
}


def build_business_risk_grade_table() -> DecisionTableConfig:
    rows = []
    for product, cuts in _BUSINESS_BOUNDARIES.items():
        edges = [float("-inf"), *cuts, float("inf")]
        for grade in range(1, 13):
            rows.append({
                "product": product, "lo": edges[grade - 1], "hi": edges[grade], "grade": grade,
                "cell_id": _cell_id("business_risk_grade", BUSINESS_RISK_GRADE_VERSION, product, grade),
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "business_risk_grade_boundaries",
        "columns": {"product": "Int64", "lo": "Float64", "hi": "Float64", "grade": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "between", "variable": "probability_of_default", "lower_bound_column": "lo",
             "upper_bound_column": "hi", "allow_gaps": True},
        ]},
        "outputs": ["grade", "cell_id"],
        "default": [12, None],
    }).relabel(writes={"cell_id": "business_risk_grade_cell_id"})


def combined_pd_before_overlay(
    financial_pd: float, people_pd: float,
    financial_weight: float = param(0.40, ge=0.0, le=1.0), people_weight: float = param(0.60, ge=0.0, le=1.0),
) -> float:
    """Log-odds blend (spec 05 §5.10: "blending is on log-odds, as at PP-05" --
    the same rule, applied one level up)."""
    import math
    w_total = financial_weight + people_weight
    fw, pw = financial_weight / w_total, people_weight / w_total

    def logit(p: float) -> float:
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return math.log(p / (1.0 - p))

    x = fw * logit(financial_pd) + pw * logit(people_pd)
    return 1.0 / (1.0 + math.exp(-x))


def build_business_overlays() -> tuple[AdjustmentRegister, str]:
    register = AdjustmentRegister([
        Adjustment(
            adjustment_id="ADJ-05-BIZ-001", kind="odds_multiplier", target="probability_of_default",
            effect=AdjustmentEffect("multiply", 1.25), scope={"product_code": vocab.PRODUCT_BUSINESS_TERM_FACILITY},
            stack_position=6, owner="Business Credit Risk Policy", approval_reference="CRC-2026-055",
            rationale="Tighten construction-sector term facilities by 25% for two quarters",
            effective_from=date(2026, 1, 1), effective_to=date(2026, 12, 31), review_date=date(2026, 11, 1),
            tighten_only=True,
        ),
    ])
    return register, "AS-05-BIZ-2026.09"


_BUSINESS_OVERLAYS, _BUSINESS_ADJUSTMENT_SET_ID = build_business_overlays()


def business_grade_output(grade: int) -> int:
    return grade


def build_grade_unit():
    overlay_step = _BUSINESS_OVERLAYS.apply_stack_step(
        "probability_of_default", _BUSINESS_ADJUSTMENT_SET_ID, base_field="probability_of_default_before_overlay",
    )
    return dag(
        step(combined_pd_before_overlay, output="probability_of_default_before_overlay"),
        overlay_step,
        build_business_risk_grade_table(),
        step(business_grade_output, output="risk_grade"),
        name="business_grade",
    ).emit(
        "probability_of_default_before_overlay", "probability_of_default", "probability_of_default_unadjusted",
        "adjustment_set_id", "adjustments_applied", "risk_grade", "business_risk_grade_cell_id",
    )
