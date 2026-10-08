"""From the ceilings and the affordable instalment to the offers a client is shown, and the check on the one recommended.

Three steps, built around the rate card: `solve_terms` solves every permitted term, `offer_set` applies the minimum
viable offer rules, removes near-duplicates and recommends one, and `final_validation` re-derives that offer from its
own amount and term, carrying nothing forward, and refuses to ship it if anything disagrees.
"""
from __future__ import annotations

import math
from dataclasses import asdict

from decider import ParamTable, allow_fallback, missing_as, param, param_table, step

from credit_core.vocab import SEGMENT_NEW_TO_BANK

from loan_granting import reasons
from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, RateCardIndex
from loan_granting.solve import BIND_EXH, GRID, PRODUCT_MAX_AMOUNT, PRODUCT_MIN_AMOUNT, solve_term

LARGEST_AMOUNT, LOWEST_TOTAL_COST, BEST_EXPECTED_VALUE = "largest_amount", "lowest_total_cost", "best_expected_value"

# One table, shared by the solve and the final validation, so a weekly change to the term list cannot reach only one.
PERMITTED_TERMS = param_table({"term": int, "new_to_bank": bool}, shared_key="permitted_terms", default=[
    {"term": term, "new_to_bank": term != 84} for term in (6, 12, 18, 24, 36, 48, 60, 72, 84)])


def total_cost_ratio_threshold(total_cost_ratio_threshold: float = param(1.85, gt=0.0)) -> float:
    """Total cost of credit per rand advanced above which an offer is suppressed; the final validation checks it too."""
    return total_cost_ratio_threshold


@allow_fallback
def recommendation_objective(recommendation_objective: str = param(LARGEST_AMOUNT)) -> str:
    """`largest_amount`, `lowest_total_cost` or `best_expected_value`: product changes it without a release."""
    return recommendation_objective


def _evaluator(rate_card_index, credit_life_index, statutory_ceiling, substitution, initiation_fee_cap, service_cap):
    return PriceEvaluator(rate_card_index, credit_life_index, statutory_ceiling, substitution, initiation_fee_cap,
                          service_cap)


def _offer(solution: dict) -> dict:
    pricing = solution["pricing"]
    return {
        "term_months": solution["term_months"], "offered_amount": pricing["offered_amount"],
        "nominal_annual_rate": pricing["nominal_annual_rate"], "rate_cell_id": pricing["rate_cell_id"],
        "rate_card_version": pricing["rate_card_version"], "initiation_fee": pricing["initiation_fee"],
        "monthly_service_fee": pricing["monthly_service_fee"], "amount_financed": pricing["amount_financed"],
        "credit_life_premium": pricing["credit_life_premium"], "instalment": pricing["instalment"],
        "total_cost_of_credit": pricing["total_cost_of_credit"],
        "effective_annual_rate": pricing["effective_annual_rate"],
        "binding_constraint_code": solution["binding_constraint_code"], "evaluation_count": solution["evaluation_count"],
        "is_recommended": False,
    }


def _suppression_codes(offer: dict, rules: dict) -> list[int]:
    """Every minimum viable offer rule the offer breaks, as reason codes."""
    advanced, cost = offer["offered_amount"], offer["total_cost_of_credit"]
    broken = {
        reasons.R_BELOW_MIN_AMOUNT: advanced < rules["minimum_offer_amount"],
        reasons.R_BELOW_MIN_INSTALMENT: offer["instalment"] < rules["minimum_instalment"],
        reasons.R_TOTAL_COST_RATIO_EXCEEDED: cost / advanced > rules["total_cost_ratio_threshold"],
        reasons.R_IN_DUPLUM_EXCEEDED: cost - advanced > advanced,  # charges over the term above the amount advanced
        reasons.R_EFFECTIVE_RATE_EXCEEDED: offer["effective_annual_rate"] > rules["effective_rate_ceiling"],
    }
    return [code for code, is_broken in broken.items() if is_broken]


def expected_value(offer: dict, probability_of_default: float, lgd: float, average_ead: float, funding_cost: float) -> float:
    """Margin after funding cost, weighted by survival, less the expected loss; the 12-month PD is spread over the term."""
    years = offer["term_months"] / 12.0
    pd_term = 1.0 - (1.0 - min(max(probability_of_default, 0.0), 0.999)) ** years
    margin = offer["total_cost_of_credit"] - offer["offered_amount"] - offer["offered_amount"] * funding_cost * years
    return round(margin * (1.0 - pd_term) - lgd * average_ead * offer["offered_amount"] * pd_term, 2)


def build_offer_set(
    term_solutions: list[dict], objective: str, probability_of_default: float, rules: dict,
) -> tuple[list[dict], list[dict]]:
    """The surviving offers (one flagged `is_recommended`) and a suppression record for every term without one.

    Of the offers whose amounts are equal and whose instalments differ by less than `rules["dedup_tolerance"]`, the
    cheaper survives. The tie-break is fixed: the objective, then the larger amount, then the lower total cost, then
    the shorter term.
    """
    survivors, suppressed = [], []
    for solution in sorted(term_solutions, key=lambda s: s["term_months"]):
        term = solution["term_months"]
        if solution.get("excluded"):
            suppressed.append({"term_months": term, "reason": solution["excluded"], "reason_codes": []})
        elif solution["pricing"] is None:
            exhausted = solution["binding_constraint_code"] == BIND_EXH
            suppressed.append({"term_months": term, "reason_codes": [reasons.R_EVALUATION_CEILING_REACHED] if exhausted else [],
                               "reason": "evaluation ceiling reached" if exhausted else "no affordable amount"})
        else:
            offer = _offer(solution)
            broken = _suppression_codes(offer, rules)
            duplicate = next((o for o in survivors if o["offered_amount"] == offer["offered_amount"]
                              and abs(offer["instalment"] - o["instalment"]) / o["instalment"] < rules["dedup_tolerance"]), None)
            if broken:
                suppressed.append({"term_months": term, "reason": "minimum viable offer rules", "reason_codes": broken})
            elif duplicate is None:
                survivors.append(offer)
            elif offer["total_cost_of_credit"] < duplicate["total_cost_of_credit"]:
                survivors[survivors.index(duplicate)] = offer

    def rank(offer: dict) -> tuple:
        if objective == LOWEST_TOTAL_COST:
            primary = offer["total_cost_of_credit"] / offer["offered_amount"]
        elif objective == BEST_EXPECTED_VALUE:
            primary = -expected_value(offer, probability_of_default, rules["lgd"], rules["average_ead"],
                                      rules["funding_cost"])
        else:
            primary = 0.0
        return primary, -offer["offered_amount"], offer["total_cost_of_credit"], offer["term_months"]

    if survivors:
        min(survivors, key=rank)["is_recommended"] = True
    return survivors, suppressed


def steps(rate_card_index: RateCardIndex):
    """`(solve_terms, offer_set, final_validation)`, pricing from `rate_card_index` (build it once, from the card)."""
    credit_life_index = CreditLifeIndex()

    @allow_fallback
    @step(outputs=("term_solutions", "solve_total_evaluations"))
    def solve_terms(
        risk_grade: int, applicant_age_years: float, is_joint_application: bool, amount_cap: float, term_cap: float,
        max_affordable_instalment: float, segment_code: float, statutory_ceiling: float, initiation_fee_cap: float,
        monthly_service_fee_cap: float, requested_amount: float = missing_as(math.nan),
        credit_life_substitution_declared: bool = missing_as(False),
        evaluation_ceiling_per_term: int = param(24, ge=1, le=64), permitted_terms: ParamTable = PERMITTED_TERMS,
    ) -> tuple[list[dict], int]:
        """Solve every permitted term not ruled out by the term cap or the segment; the amounts are found, not guessed."""
        evaluator = _evaluator(rate_card_index, credit_life_index, statutory_ceiling,
                               credit_life_substitution_declared, initiation_fee_cap, monthly_service_fee_cap)
        requested = None if math.isnan(requested_amount) else requested_amount
        solutions = []
        for term, new_to_bank in zip(permitted_terms.term, permitted_terms.new_to_bank):
            term = int(term)
            if term > term_cap:
                solutions.append({"term_months": term, "excluded": "above the term cap"})
            elif segment_code == SEGMENT_NEW_TO_BANK and not new_to_bank:
                solutions.append({"term_months": term, "excluded": "not offered to new-to-bank clients"})
            else:
                result = solve_term(evaluator, term, risk_grade, applicant_age_years, is_joint_application, amount_cap,
                                    requested, max_affordable_instalment, evaluation_ceiling_per_term)
                solutions.append({
                    "term_months": term, "amount": result.amount, "binding_constraint_code": result.binding_constraint_code,
                    "evaluation_count": result.evaluation_count,
                    "pricing": asdict(result.pricing) if result.pricing else None})
        return solutions, sum(s.get("evaluation_count", 0) for s in solutions)

    @allow_fallback
    @step(outputs=("offers", "recommended_offer", "suppressed_terms", "suppression_reason_codes", "has_any_offer"))
    def offer_set(
        term_solutions: list[dict], probability_of_default: float, recommendation_objective: str,
        total_cost_ratio_threshold: float, minimum_offer_amount: float = param(PRODUCT_MIN_AMOUNT, ge=0.0),
        minimum_instalment: float = param(150.0, ge=0.0), effective_rate_ceiling: float = param(0.60, gt=0.0),
        dedup_tolerance: float = param(0.02, ge=0.0), lgd: float = param(0.72, ge=0.0, le=1.0),
        average_ead: float = param(0.55, ge=0.0, le=1.0), funding_cost: float = param(0.0875, ge=0.0),
    ) -> tuple[list[dict], dict, list[dict], list[int], bool]:
        """The offers that survive the minimum viable offer rules, the recommended one, and why each other term has none."""
        rules = {"minimum_offer_amount": minimum_offer_amount, "minimum_instalment": minimum_instalment,
                 "total_cost_ratio_threshold": total_cost_ratio_threshold, "effective_rate_ceiling": effective_rate_ceiling,
                 "dedup_tolerance": dedup_tolerance, "lgd": lgd, "average_ead": average_ead, "funding_cost": funding_cost}
        offers, suppressed = build_offer_set(term_solutions, recommendation_objective, probability_of_default, rules)
        recommended = next((o for o in offers if o["is_recommended"]), {})
        codes = sorted({code for s in suppressed for code in s["reason_codes"]})
        return offers, recommended, suppressed, codes, bool(offers)

    @allow_fallback
    @step(outputs=("final_validation_passed", "final_validation_failures"))
    def final_validation(
        recommended_offer: dict, risk_grade: int, applicant_age_years: float, is_joint_application: bool,
        amount_cap: float, term_cap: float, worst_acceptable_grade: float, max_affordable_instalment: float,
        statutory_ceiling: float, initiation_fee_cap: float, monthly_service_fee_cap: float,
        total_cost_ratio_threshold: float, credit_life_substitution_declared: bool = missing_as(False),
        permitted_terms: ParamTable = PERMITTED_TERMS,
    ) -> tuple[bool, list[str]]:
        """Re-derive the recommended offer from its amount and term alone and list every check that fails."""
        if not recommended_offer:
            return True, []
        offer = recommended_offer
        evaluator = _evaluator(rate_card_index, credit_life_index, statutory_ceiling,
                               credit_life_substitution_declared, initiation_fee_cap, monthly_service_fee_cap)
        amount, term = offer["offered_amount"], offer["term_months"]
        fresh = evaluator.evaluate(amount, term, risk_grade, applicant_age_years, is_joint_application)
        checks = {
            "rate_matches_the_card_cell": fresh.priced and fresh.rate_cell_id == offer["rate_cell_id"],
            "rate_within_the_statutory_ceiling": fresh.within_statutory_ceiling,
            "initiation_fee_recomputes": fresh.initiation_fee == offer["initiation_fee"] <= initiation_fee_cap,
            "service_fee_recomputes": fresh.monthly_service_fee == offer["monthly_service_fee"] <= monthly_service_fee_cap,
            "credit_life_premium_recomputes": fresh.credit_life_premium == offer["credit_life_premium"],
            "instalment_recomputes_to_the_cent": fresh.instalment == offer["instalment"],
            "instalment_within_what_is_affordable": fresh.instalment <= max_affordable_instalment,
            "charges_within_in_duplum": fresh.total_cost_of_credit - amount <= amount,
            "total_cost_ratio_within_threshold": fresh.total_cost_of_credit / amount <= total_cost_ratio_threshold,
            "amount_within_every_ceiling": PRODUCT_MIN_AMOUNT <= amount <= min(amount_cap, PRODUCT_MAX_AMOUNT),
            "term_permitted": term <= term_cap and term in permitted_terms.term,
            "grade_within_the_worst_acceptable": risk_grade <= worst_acceptable_grade,
            "amount_is_a_multiple_of_the_grid": amount % GRID == 0,
            "rate_card_version_matches": fresh.rate_card_version == offer["rate_card_version"],
        }
        failures = [name for name, ok in checks.items() if not ok]
        return not failures, failures

    return solve_terms, offer_set, final_validation
