"""The business grade: the financial and people components blended on log-odds, an overlay, then the product's ladder."""
from __future__ import annotations

from types import SimpleNamespace

from decider import allow_fallback, flow, param, step

from credit_core.adjustments import AdjustmentStack
from credit_core.vocab import PRODUCT_BUSINESS_REVOLVING_FACILITY, PRODUCT_BUSINESS_TERM_FACILITY

from business_nested import scoring
from business_nested.blend import log_odds_blend
from business_nested.documents import shipped

GRADE_VERSION = "biz-rg-2026.09"

# product -> the 11 ascending probability-of-default cut points. Revolving is stricter than term at every boundary.
_GRADE_CUTS = {
    PRODUCT_BUSINESS_TERM_FACILITY: [0.008, 0.016, 0.028, 0.045, 0.07, 0.10, 0.14, 0.19, 0.26, 0.35, 0.48],
    PRODUCT_BUSINESS_REVOLVING_FACILITY: [0.006, 0.013, 0.023, 0.038, 0.06, 0.088, 0.125, 0.17, 0.235, 0.32, 0.44],
}

@allow_fallback
def combined_pd_before_overlay(
    financial_pd: float, people_pd: float,
    financial_weight: float = param(0.40, ge=0.0, le=1.0), people_weight: float = param(0.60, ge=0.0, le=1.0),
) -> float:
    """The two components blended on the log-odds scale, by weights that are scaled to sum to 1."""
    total = financial_weight + people_weight
    return log_odds_blend([financial_pd, people_pd], [financial_weight / total, people_weight / total])


def business_grade_unit(pd_adjustments: AdjustmentStack | None = None):
    """Writes `probability_of_default_before_overlay`, `probability_of_default`, its unadjusted twin, the overlays
    applied, and `risk_grade` with and without the overlay.

    `pd_adjustments` is the `business_pd_adjustments` document; by default the one in `configs/0.1.0`.
    """
    return flow(
        step(combined_pd_before_overlay, output="probability_of_default_before_overlay"),
        pd_adjustments or shipped("business_pd_adjustments"),
        scoring.grade_lookup("probability_of_default", "risk_grade", "product_code", _GRADE_CUTS, GRADE_VERSION),
        scoring.grade_lookup("probability_of_default_unadjusted", "risk_grade_unadjusted", "product_code", _GRADE_CUTS,
                             GRADE_VERSION),
        name="business_grade",
    ).emit("probability_of_default_before_overlay", "probability_of_default", "probability_of_default_unadjusted",
           "adjustment_set_id", "adjustments_applied", "risk_grade", "risk_grade_cell_id", "risk_grade_unadjusted")


# governance-replay lists the overlays through this name; no step reads it.
BUSINESS_OVERLAYS = SimpleNamespace(_all=shipped("business_pd_adjustments").adjustments)
