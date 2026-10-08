"""EP-1, a new-to-bank business application: the nested-entity assessment, then what a facility adds to it.

`decider build` and `decider serve` use `build`. The assessment is `business_nested`'s, embedded whole; this module adds
the instalment the covenant is tested against, the master scale the grade is on, and the DSCR covenant bound at approval.
"""
from __future__ import annotations

from decider import allow_fallback, flow, param

from business_nested.pipeline import DECISION_OUTPUTS, build as build_assessment

from business_credit_e2e import covenant, vocab

OUTPUTS = (
    "assessed_entities", *DECISION_OUTPUTS, "master_scale_version", "new_facility_instalment",
    "covenant_instance_id", "covenant_definition_version", "covenant_threshold", "covenant_cure_days",
    "dscr_existing_debt_service", "dscr_total_debt_service", "measured_dscr", "dscr_headroom_ratio",
    "dscr_breach_class_code",
)


def new_facility_instalment(
    product_code: float, offered_amount: float, instalment: float,
    minimum_payment_pct: float = param(0.05, ge=0.0, le=1.0), minimum_payment: float = param(50.0, ge=0.0),
) -> float:
    """What the facility costs each month for the covenant: the term loan's instalment, or for a revolving facility
    the minimum payment on the full limit (the assessment prices a term loan only)."""
    if product_code == vocab.PRODUCT_REVOLVING:
        return round(max(offered_amount * minimum_payment_pct, minimum_payment), 2)
    return instalment


@allow_fallback
def master_scale_version(risk_grade_cell_id: str) -> str:
    # workaround: the assessment publishes no master scale version; it is the version in the grade's cell id,
    # `<table>@<version>#<keys>`.
    return risk_grade_cell_id.split("@", 1)[1].split("#", 1)[0]


def build():
    return flow(
        build_assessment(), master_scale_version, new_facility_instalment,
        covenant.covenant_definition_version, covenant.covenant_terms, covenant.covenant_instance_id,
        covenant.dscr_existing_debt_service, covenant.dscr_total_debt_service, covenant.measured_dscr,
        covenant.dscr_headroom_ratio, covenant.dscr_breach_class_code,
        name="business_credit_e2e",
    ).emit(*OUTPUTS)
