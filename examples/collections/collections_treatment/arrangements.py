"""Payment arrangements: the affordability assessment in arrangement mode, then a sustainability test of our own."""
from __future__ import annotations

import math

from decider import allow_fallback, dag, missing_as, param, step

from assessment import modes
from assessment.pipeline import capacity_unit, evidence_unit


def arrangement_assessment_mode(mode: int = param(modes.ARRANGEMENT)) -> int:
    """The `assessment_mode_code` this pipeline runs the assessment in: arrangement mode, which accepts any income evidence."""
    return mode


def assessment_unit():
    """The affordability assessment's two halves, wired as `assessment.pipeline.build()` wires them."""
    return dag(step(arrangement_assessment_mode, output="assessment_mode_code"), evidence_unit(), capacity_unit(),
               name="arrangement_assessment")


@allow_fallback
def arrangement_sustainability(
    net_monthly_income: float, discretionary_income: float, proposed_instalment: float = missing_as(math.nan),
    minimum_residual: float = param(350.0, ge=0.0), minimum_residual_income_pct: float = param(0.05, ge=0.0, le=1.0),
    max_instalment_to_discretionary_pct: float = param(0.85, ge=0.0, le=1.0),
) -> tuple[bool, float, str]:
    """Whether the client can keep the arrangement, not whether they could take new credit.

    Returns `arrangement_sustainable`, `arrangement_residual` (discretionary income left after the instalment)
    and `arrangement_failure_detail` (every test failed, or "no_proposal" when no instalment was proposed).
    """
    if math.isnan(proposed_instalment):
        return False, math.nan, "no_proposal"
    residual = discretionary_income - proposed_instalment
    failures = [name for name, failed in (
        ("below_minimum_residual", residual < minimum_residual),
        ("below_minimum_residual_income_pct", net_monthly_income <= 0 or residual / net_monthly_income < minimum_residual_income_pct),
        ("instalment_exceeds_discretionary_income_pct",
         discretionary_income <= 0 or proposed_instalment / discretionary_income > max_instalment_to_discretionary_pct),
    ) if failed]
    return not failures, round(residual, 2), ",".join(failures)


arrangement_sustainability_step = step(
    arrangement_sustainability,
    outputs=("arrangement_sustainable", "arrangement_residual", "arrangement_failure_detail"))
