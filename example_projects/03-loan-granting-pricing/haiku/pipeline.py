"""Decider pipeline for unsecured Flex Loan granting and pricing."""

from decider import step, param, flow, missing_as, RequestHandler
from datetime import date
from typing import Optional, List, Dict, Any
import json
import uuid

# Import granting modules
from granting.types import GrantingResult, OfferOutput
from granting.eligibility import evaluate_eligibility_gates
from granting.caps import build_cap_register, evaluate_cap_waterfall
from granting.solve import bounded_solve
from granting.offers import construct_offer_set, validate_offer

# Stub imports for dependencies (would normally import from 00 and 02)
# For now, provide minimal implementations


def stub_evaluate_scorecard(bureau_data, application_data):
    """Stub: Return a score of 600 (mid-range)."""
    return {
        "scorecard_id": "1011",
        "score": 600,
        "score_unadjusted": 600,
        "probability_of_default": 0.03,
        "probability_of_default_unadjusted": 0.03,
        "risk_grade": 7,
        "risk_grade_unadjusted": 7,
        "score_contributions": {},
    }


def stub_assess_affordability(income, deductions, expenses, obligations, mode=1):
    """Stub: Return a reasonable affordability assessment."""
    return {
        "max_affordable_instalment": 4350.0,
        "discretionary_income": 9230.0,
        "verdict": "pass",
    }


def stub_lookup_rate_card(amount, term, grade):
    """Stub: Return a rate based on simple linear interpolation."""
    # Simple: grade 1 -> 12.9%, grade 12 -> 27.5%, linear
    base_rate = 0.129 + (grade - 1) * (0.275 - 0.129) / 11
    # Slight term adjustment
    term_adjustment = -0.0001 * term
    rate = min(base_rate + term_adjustment, 0.2850)
    return {"rate": rate, "cell_id": f"rate_{grade}_{term}_{int(amount/1000)}"}


def stub_lookup_fees(amount):
    """Stub: Simple fee calculation."""
    # Base 180 + 10% of amount above 1000, capped at 1350
    if amount <= 1000:
        fee = 180
    else:
        fee = 180 + (amount - 1000) * 0.10
    fee = min(fee, 1350)
    return {"initiation_fee": fee, "fee_exclusive_tax": fee}


def stub_lookup_credit_life(amount_financed, age, term, employment):
    """Stub: Credit life premium."""
    # Simple: R2.50 per R1000
    premium_per_k = 2.50
    monthly_premium = (amount_financed / 1000) * premium_per_k
    return {"monthly_premium": monthly_premium}


def stub_calculate_instalment(amount_financed, rate, term_months, service_fee, credit_life_premium):
    """Stub: Calculate monthly instalment."""
    if term_months == 0 or rate == 0:
        return 0
    monthly_rate = rate / 12
    # Standard amortization: PMT = P * (r * (1+r)^n) / ((1+r)^n - 1)
    numerator = monthly_rate * (1 + monthly_rate) ** term_months
    denominator = (1 + monthly_rate) ** term_months - 1
    if denominator == 0:
        base_instalment = amount_financed / term_months
    else:
        base_instalment = amount_financed * (numerator / denominator)

    total = base_instalment + service_fee + credit_life_premium
    return total


@step(param("decision_date_str", "2026-09-24"))
def intake_and_parse(decision_date_str: str) -> Dict[str, Any]:
    """Parse and prepare inputs."""
    decision_date = date.fromisoformat(decision_date_str)
    return {"decision_date": decision_date}


@step()
def eligibility_gates(
    applicant_age_years: float,
    decision_date: date,
    product_code: int = 10,
    channel_code: int = 2,
    residency_code: int = 1,
    employment_type_code: int = 1,
    debt_review_status_code: int = 0,
    administration_order_flag: bool = False,
    insolvency_status_code: int = 0,
    deceased_flag: bool = False,
    estate_flag: bool = False,
    exclusion_list_hits: Optional[List[str]] = missing_as([]),
    in_flight_applications: Optional[List[Dict]] = missing_as([]),
    duplicate_detected: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate eligibility gates (14 gates, all evaluated)."""
    verdicts, is_eligible = evaluate_eligibility_gates(
        applicant_age_years=applicant_age_years,
        decision_date=decision_date,
        product_code=product_code,
        channel_code=channel_code,
        residency_code=residency_code,
        employment_type_code=employment_type_code,
        debt_review_status_code=debt_review_status_code,
        administration_order_flag=administration_order_flag,
        insolvency_status_code=insolvency_status_code,
        deceased_flag=deceased_flag,
        estate_flag=estate_flag,
        exclusion_list_hits=exclusion_list_hits,
        in_flight_applications=in_flight_applications,
        duplicate_detected=duplicate_detected,
    )
    return {
        "is_eligible": is_eligible,
        "eligibility_verdicts": verdicts,
        "decision_id": str(uuid.uuid4()),
    }


@step()
def scorecard_and_grading(
    is_eligible: bool,
    decision_date: date,
    applicant_age_years: float = 38,
    # Other fields stubbed
) -> Dict[str, Any]:
    """Evaluate scorecard and assign grade."""
    if not is_eligible:
        return {
            "score": 0,
            "risk_grade": 12,
            "probability_of_default": 0.99,
        }

    result = stub_evaluate_scorecard({}, {})
    return result


@step()
def cap_waterfall(
    is_eligible: bool,
    risk_grade: int,
    employment_tenure_months: Optional[float] = missing_as(None),
    is_new_to_bank: bool = False,
    arrears_3m_count: int = 0,
    arrears_2m_count: int = 0,
    arrears_2m_recency_days: int = 365,
    credit_enquiry_60d: int = 0,
    credit_enquiry_90d: int = 0,
    employer_on_watchlist: bool = False,
    group_exposure_limit: float = 500000,
    internal_exposure: float = 0,
    channel_code: int = 2,
    campaign_id: Optional[int] = missing_as(None),
    campaign_uplift_authorized: bool = False,
) -> Dict[str, Any]:
    """Run the 52-rule cap waterfall."""
    if not is_eligible:
        return {"amount_cap": 0, "term_cap": 0, "worst_acceptable_grade": 12}

    rules = build_cap_register()
    ceilings, chains, verdicts = evaluate_cap_waterfall(
        risk_grade=risk_grade,
        employment_tenure_months=employment_tenure_months,
        is_new_to_bank=is_new_to_bank,
        arrears_3m_count=arrears_3m_count,
        arrears_2m_count=arrears_2m_count,
        arrears_2m_recency_days=arrears_2m_recency_days,
        credit_enquiry_60d=credit_enquiry_60d,
        credit_enquiry_90d=credit_enquiry_90d,
        employer_on_watchlist=employer_on_watchlist,
        group_exposure_limit=group_exposure_limit,
        internal_exposure=internal_exposure,
        channel_code=channel_code,
        campaign_id=campaign_id,
        campaign_uplift_authorized=campaign_uplift_authorized,
        rules=rules,
    )

    return {
        "amount_cap": ceilings["amount_cap"],
        "term_cap": ceilings["term_cap"],
        "worst_acceptable_grade": ceilings["worst_acceptable_grade"],
        "cap_chains": chains,
        "rule_verdicts": verdicts,
    }


@step()
def affordability_assessment(
    is_eligible: bool,
    risk_grade: int,
    net_income: float = 24800,
    living_expenses: float = 9450,
    existing_obligations: float = 6120,
) -> Dict[str, Any]:
    """Consume affordability assessment from project 02."""
    if not is_eligible:
        return {
            "max_affordable_instalment": 0,
            "affordability_verdict": "fail",
        }

    result = stub_assess_affordability(net_income, 0, living_expenses, existing_obligations)
    return result


@step()
def solve_and_pricing(
    is_eligible: bool,
    amount_cap: float,
    term_cap: int,
    max_affordable_instalment: float,
    risk_grade: int,
    requested_amount: Optional[float] = missing_as(120000),
    requested_term: Optional[int] = missing_as(60),
    applicant_age_years: float = 38,
    employment_type_code: int = 1,
    channel_code: int = 2,
) -> Dict[str, Any]:
    """Run bounded solve for max affordable amount at each term."""
    if not is_eligible or amount_cap <= 0:
        return {"term_results": [], "offers": []}

    def pricing_fn(amount, term_months):
        rate_data = stub_lookup_rate_card(amount, term_months, risk_grade)
        fee_data = stub_lookup_fees(amount)
        amount_financed = amount + fee_data["initiation_fee"] * 1.15  # With tax
        credit_life = stub_lookup_credit_life(amount_financed, applicant_age_years, term_months, employment_type_code)
        service_fee = 82.80 / term_months  # Annualized
        instalment = stub_calculate_instalment(
            amount_financed, rate_data["rate"], term_months, service_fee, credit_life["monthly_premium"]
        )
        total_cost = instalment * term_months

        return {
            "rate": rate_data["rate"],
            "cell_id": rate_data["cell_id"],
            "initiation_fee": fee_data["initiation_fee"],
            "monthly_service_fee": service_fee,
            "credit_life_premium": credit_life["monthly_premium"],
            "instalment": instalment,
            "total_cost": total_cost,
            "financed": amount_financed,
            "effective_annual_rate": rate_data["rate"],  # Simplified
        }

    permitted_terms = [t for t in [6, 12, 18, 24, 36, 48, 60, 72, 84] if t <= term_cap]

    term_results = bounded_solve(
        amount_cap=amount_cap,
        requested_amount=requested_amount,
        max_affordable_instalment=max_affordable_instalment,
        permitted_terms=permitted_terms,
        pricing_fn=pricing_fn,
        term_cap=term_cap,
    )

    return {"term_results": term_results}


@step()
def construct_offers(
    term_results: List = None,
    max_affordable_instalment: float = 4350,
    amount_cap: float = 500000,
    term_cap: int = 84,
    risk_grade: int = 7,
    worst_acceptable_grade: int = 12,
) -> Dict[str, Any]:
    """Construct offer set from term results."""
    if not term_results:
        return {"offers": [], "suppressions": []}

    def pricing_fn(amount, term):
        # Stub
        return {
            "rate": 0.18,
            "cell_id": "",
            "initiation_fee": 200,
            "monthly_service_fee": 83,
            "credit_life_premium": 2.5,
            "instalment": 4000,
            "total_cost": 4000 * term,
            "effective_annual_rate": 0.30,
        }

    offers, suppressions = construct_offer_set(
        term_results=term_results,
        pricing_fn=pricing_fn,
        recommendation_objective="largest_amount",
    )

    return {"offers": offers, "suppressions": suppressions}


@step()
def final_validation(
    offers: List = None,
    max_affordable_instalment: float = 4350,
    amount_cap: float = 500000,
    term_cap: int = 84,
    risk_grade: int = 7,
    worst_acceptable_grade: int = 12,
) -> Dict[str, Any]:
    """Final validation of offers."""
    if not offers:
        return {"outcome": "decline", "validated_offers": []}

    validated = []
    for offer in offers:
        is_valid, errors = validate_offer(
            offer,
            max_affordable_instalment=max_affordable_instalment,
            amount_cap=amount_cap,
            term_cap=term_cap,
            risk_grade=risk_grade,
            worst_acceptable_grade=worst_acceptable_grade,
        )
        if is_valid:
            validated.append(offer)

    if validated:
        return {
            "outcome": "approve",
            "validated_offers": validated,
            "recommended_offer": validated[0],
        }
    else:
        return {"outcome": "decline", "validated_offers": []}


@flow(
    name="flex_loan_granting",
    steps=[
        intake_and_parse,
        eligibility_gates,
        scorecard_and_grading,
        cap_waterfall,
        affordability_assessment,
        solve_and_pricing,
        construct_offers,
        final_validation,
    ],
)
def flex_loan_granting(request: dict) -> dict:
    """Complete Flex Loan granting decision flow."""
    return request


def build():
    """Build the pipeline for decider."""
    return flex_loan_granting


class InferenceHandler(RequestHandler):
    """Handler for serving predictions."""

    def invoke(self, request):
        result = self.pipeline(request)
        return {
            "decision_id": result.get("decision_id"),
            "outcome": result.get("outcome"),
            "offers": result.get("validated_offers", []),
        }
