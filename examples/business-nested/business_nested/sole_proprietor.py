"""A sole proprietor is a natural person lending in their own name, so the statutory affordability assessment applies.

The assessment is the `affordability` example, embedded as it is: its policy numbers appear in this project's
params document under `business_credit/sole_proprietor/...`. It runs only where `legal_form_code` says sole proprietor.
"""
from __future__ import annotations

from decider import branch, flow, param, step

from assessment.pipeline import build as build_assessment

from business_nested import vocab

NEW_APPLICATION = 1
SELF_EMPLOYED = 3.0


def sole_proprietor_assessment_ran(legal_form_code: int) -> bool:
    return legal_form_code == vocab.SOLE_PROPRIETOR


@step(outputs=("sole_proprietor_affordability_verdict_code", "sole_proprietor_max_affordable_instalment"))
def not_assessed() -> tuple[int, float]:
    return 0, 0.0


@step(outputs=("principal_employment_type_code", "principal_statement_income", "principal_statement_confidence",
               "principal_assessment_mode_code", "principal_is_joint_application"))
def principal_income_evidence(
    declared_annual_turnover: float, assumed_profit_margin: float = param(0.15, ge=0.0, le=1.0),
    statement_confidence: float = param(0.8, ge=0.0, le=1.0),
) -> tuple[float, float, float, int, bool]:
    """The principal's monthly income as the business's profit on its declared turnover, evidenced by statements.

    A stand-in for the net profit after drawings, which the request does not carry.
    """
    return SELF_EMPLOYED, declared_annual_turnover * assumed_profit_margin / 12.0, statement_confidence, NEW_APPLICATION, False


def sole_proprietor_unit():
    """Writes `sole_proprietor_assessment_ran`, and the assessment's verdict code and maximum instalment (0 and 0
    when it did not run).

    The assessment tests the quoted `instalment` against the principal's own capacity at the business's `risk_grade`.
    Their personal debts and costs are optional `principal_*` fields of the request.
    """
    assessment = build_assessment().relabel(
        reads={"bureau_as_of_date": "decision_date", "proposed_instalment": "instalment",
               "applicant1_employment_type_code": "principal_employment_type_code",
               "applicant1_statement_income": "principal_statement_income",
               "applicant1_statement_confidence": "principal_statement_confidence",
               "assessment_mode_code": "principal_assessment_mode_code",
               "is_joint_application": "principal_is_joint_application",
               "applicant1_bureau_accounts": "principal_bureau_accounts",
               "applicant1_internal_accounts": "principal_internal_accounts",
               "applicant1_declared_expenses": "principal_declared_expenses",
               "applicant1_dependants_count": "principal_dependants_count"},
        writes={"affordability_verdict_code": "sole_proprietor_affordability_verdict_code",
                "max_affordable_instalment": "sole_proprietor_max_affordable_instalment"})
    return flow(
        step(sole_proprietor_assessment_ran), not_assessed,
        branch("sole_proprietor_assessment_ran", flow(principal_income_evidence, assessment, name="assess"),
               modifies=["sole_proprietor_affordability_verdict_code", "sole_proprietor_max_affordable_instalment"],
               name="statutory"),
        name="sole_proprietor",
    ).emit("sole_proprietor_assessment_ran")
