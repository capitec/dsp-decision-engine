"""Stages 5.8-5.10: run the solve over every permitted term, build the offer
set (§5.9), and independently re-validate the recommended offer (§5.10).

One `frame_step`, not three, built by `build_granting_step(rate_card_index)`:
the three stages share so much per-application state (the `PriceEvaluator`,
the nine terms' solve results) that splitting them into separate decider
steps would mean re-serialising every per-term evaluation through frame
columns for no reader other than the next stage. Each stage is still a
separate, independently callable, docstringed function inside this module
-- the framework boundary and the code's own module boundary don't have to
be the same line.

`rate_card_index` is built once, from the `rate_card_flex_loan`
`DecisionTableConfig` `pipeline.py` already loaded (63 360 rows), and
closed over here -- never rebuilt per application, which would dominate
batch throughput at 14 M records (spec 03 §8).
"""
from __future__ import annotations

from decider import frame_step

from loan_granting import reasons
from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, PricingResult, RateCardIndex
from loan_granting.solve import BIND_AFF, BIND_EXH, SolveResult, solve_term

PERMITTED_TERMS = (6, 12, 18, 24, 36, 48, 60, 72, 84)
NEW_TO_BANK_EXCLUDED_TERM = 84

MIN_OFFER_AMOUNT = 2_000.0
MIN_INSTALMENT = 150.0
TOTAL_COST_RATIO_THRESHOLD = 1.85
EFFECTIVE_RATE_CEILING = 0.60
DEDUP_INSTALMENT_TOLERANCE_PCT = 0.02

LGD = 0.72
AVERAGE_EAD_PCT = 0.55
FUNDING_COST_RATE = 0.0875

OBJECTIVE_LARGEST_AMOUNT = "largest_amount"
OBJECTIVE_LOWEST_TOTAL_COST = "lowest_total_cost"
OBJECTIVE_BEST_EXPECTED_VALUE = "best_expected_value"


def _permitted_terms(term_cap: float, is_new_to_bank: bool) -> list[int]:
    terms = [t for t in PERMITTED_TERMS if t <= term_cap]
    if is_new_to_bank:
        terms = [t for t in terms if t != NEW_TO_BANK_EXCLUDED_TERM]
    return terms


def _run_solve_for_every_term(evaluator: PriceEvaluator, app: dict, terms: list[int],
                               evaluation_ceiling: int) -> dict[int, SolveResult]:
    return {
        t: solve_term(
            evaluator, t, app["risk_grade"], app["applicant_age_years"], app.get("is_joint_application", False),
            app["amount_cap"], app.get("requested_amount"), app["max_affordable_instalment"],
            evaluation_ceiling=evaluation_ceiling,
        )
        for t in terms
    }


def _expected_value(pricing: PricingResult, probability_of_default: float) -> float:
    pd_12m = min(max(probability_of_default, 0.0), 0.999)
    years = pricing.term_months / 12.0
    pd_over_term = 1.0 - (1.0 - pd_12m) ** years
    funding_cost = pricing.offered_amount * FUNDING_COST_RATE * years
    margin = pricing.total_cost_of_credit - pricing.offered_amount - funding_cost
    expected_loss = LGD * AVERAGE_EAD_PCT * pricing.offered_amount * pd_over_term
    return round(margin * (1.0 - pd_over_term) - expected_loss, 2)


def build_offer_set(app: dict, term_results: dict[int, SolveResult],
                     objective: str, probability_of_default: float) -> dict:
    """Spec 03 §5.9: filter each term's winning candidate through the five minimum
    viable offer rules, dedup near-identical amounts, rank and flag one recommended."""
    candidates = []
    suppressed_terms: list[int] = []
    suppression_reasons: list[list[int]] = []
    for term, result in term_results.items():
        if result.amount is None or result.pricing is None:
            suppressed_terms.append(term)
            suppression_reasons.append([reasons.R_EVALUATION_CEILING_REACHED] if result.binding_constraint_code
                                        in (BIND_EXH,) else [])
            continue
        pr = result.pricing
        fired = []
        if pr.offered_amount < MIN_OFFER_AMOUNT:
            fired.append(reasons.R_BELOW_MIN_AMOUNT)
        if pr.instalment < MIN_INSTALMENT:
            fired.append(reasons.R_BELOW_MIN_INSTALMENT)
        total_cost_ratio = pr.total_cost_of_credit / pr.offered_amount if pr.offered_amount else 0.0
        if total_cost_ratio > TOTAL_COST_RATIO_THRESHOLD:
            fired.append(reasons.R_TOTAL_COST_RATIO_EXCEEDED)
        scheduled_charges = pr.total_cost_of_credit - pr.offered_amount
        if scheduled_charges > pr.offered_amount:
            fired.append(reasons.R_IN_DUPLUM_EXCEEDED)
        if pr.effective_annual_rate > EFFECTIVE_RATE_CEILING:
            fired.append(reasons.R_EFFECTIVE_RATE_EXCEEDED)
        if fired:
            suppressed_terms.append(term)
            suppression_reasons.append(sorted(set(fired)))
            continue
        candidates.append((term, result, pr, total_cost_ratio))

    # Dedup: same amount, instalments within tolerance -> keep the lower total cost.
    survivors = []
    for term, result, pr, ratio in sorted(candidates, key=lambda c: c[0]):
        dup_of = None
        for i, (_, _, other_pr, _) in enumerate(survivors):
            if other_pr.offered_amount == pr.offered_amount and other_pr.instalment > 0 and \
                    abs(pr.instalment - other_pr.instalment) / other_pr.instalment < DEDUP_INSTALMENT_TOLERANCE_PCT:
                dup_of = i
                break
        if dup_of is None:
            survivors.append((term, result, pr, ratio))
        elif pr.total_cost_of_credit < survivors[dup_of][2].total_cost_of_credit:
            survivors[dup_of] = (term, result, pr, ratio)

    scored = []
    for term, result, pr, ratio in survivors:
        if objective == OBJECTIVE_LOWEST_TOTAL_COST:
            key = (pr.total_cost_of_credit / pr.offered_amount if pr.offered_amount else float("inf"),)
            rank_value = key[0]
        elif objective == OBJECTIVE_BEST_EXPECTED_VALUE:
            rank_value = -_expected_value(pr, probability_of_default)
        else:  # largest_amount
            rank_value = (-pr.offered_amount, pr.total_cost_of_credit)
        scored.append((rank_value, term, result, pr))
    scored.sort(key=lambda s: s[0])
    recommended_term = scored[0][1] if scored else None

    offers = []
    for term, result, pr, _ratio in sorted(survivors, key=lambda c: c[0]):
        offers.append({
            "term_months": term, "offered_amount": pr.offered_amount, "nominal_annual_rate": pr.nominal_annual_rate,
            "rate_cell_id": pr.rate_cell_id, "initiation_fee": pr.initiation_fee,
            "monthly_service_fee": pr.monthly_service_fee, "credit_life_premium": pr.credit_life_premium,
            "instalment": pr.instalment, "total_cost_of_credit": pr.total_cost_of_credit,
            "effective_annual_rate": pr.effective_annual_rate, "binding_constraint_code": result.binding_constraint_code,
            "evaluation_count": result.evaluation_count, "is_recommended": term == recommended_term,
        })

    return {
        "offers": offers, "recommended_term": recommended_term,
        "suppressed_terms": suppressed_terms, "suppression_reasons": suppression_reasons,
    }


def final_validation(evaluator: PriceEvaluator, app: dict, offer: dict | None) -> tuple[bool, list[str]]:
    """Spec 03 §5.10: re-derive the recommended offer from its own amount and term alone,
    asserting every one of the 14 checks. Any mismatch is a hard failure (never a warning)."""
    if offer is None:
        return True, []
    amount, term = offer["offered_amount"], offer["term_months"]
    fresh = evaluator.evaluate(amount, term, app["risk_grade"], app["applicant_age_years"],
                                app.get("is_joint_application", False))
    failures = []
    if not fresh.priced or fresh.rate_cell_id != offer["rate_cell_id"]:
        failures.append("rate_cell_mismatch")
    if not fresh.within_statutory_ceiling:
        failures.append("rate_above_statutory_ceiling")
    if round(fresh.initiation_fee, 2) != round(offer["initiation_fee"], 2):
        failures.append("initiation_fee_mismatch")
    if fresh.initiation_fee > 6_500.0:
        failures.append("initiation_fee_above_cap")
    if round(fresh.monthly_service_fee, 2) != round(offer["monthly_service_fee"], 2):
        failures.append("service_fee_mismatch")
    if round(fresh.credit_life_premium, 2) != round(offer["credit_life_premium"], 2):
        failures.append("credit_life_mismatch")
    if fresh.credit_life_premium > 350.0:
        failures.append("credit_life_above_cap")
    if round(fresh.instalment, 2) != round(offer["instalment"], 2):
        failures.append("instalment_does_not_recompute")
    if fresh.instalment > app["max_affordable_instalment"]:
        failures.append("instalment_exceeds_affordable_maximum")
    scheduled_charges = fresh.total_cost_of_credit - fresh.offered_amount
    if scheduled_charges > fresh.offered_amount:
        failures.append("in_duplum_exceeded")
    ratio = fresh.total_cost_of_credit / fresh.offered_amount if fresh.offered_amount else 0.0
    if ratio > TOTAL_COST_RATIO_THRESHOLD:
        failures.append("total_cost_ratio_exceeded")
    if not (MIN_OFFER_AMOUNT <= amount <= app["amount_cap"] and amount <= 500_000.0):
        failures.append("amount_outside_ceilings")
    if not (term <= app["term_cap"] and term in PERMITTED_TERMS):
        failures.append("term_outside_ceilings")
    if app["risk_grade"] > app["worst_acceptable_grade"]:
        failures.append("grade_worse_than_acceptable")
    if amount % 100.0 != 0.0:
        failures.append("amount_not_a_multiple_of_100")
    if fresh.rate_card_version != offer["rate_cell_id"].split("@")[1].split("#")[0] if offer["rate_cell_id"] else False:
        failures.append("table_version_mismatch")
    return len(failures) == 0, failures


def build_granting_step(rate_card_index: RateCardIndex):
    credit_life_index = CreditLifeIndex()

    reads = [
        "risk_grade", "applicant_age_years", "is_joint_application", "requested_amount", "requested_term_months",
        "amount_cap", "term_cap", "worst_acceptable_grade", "max_affordable_instalment",
        "credit_life_substitution_declared", "statutory_ceiling", "recommendation_objective",
        "probability_of_default", "evaluation_ceiling_per_term", "segment_code",
    ]
    writes = [
        "offer_term_months", "offer_amounts", "offer_nominal_rates", "offer_instalments",
        "offer_total_costs_of_credit", "offer_effective_rates", "offer_is_recommended",
        "offer_binding_constraint_codes",
        "recommended_term", "recommended_amount", "recommended_instalment",
        "recommended_total_cost_of_credit", "recommended_effective_annual_rate", "recommended_binding_constraint_code",
        "suppressed_terms", "suppression_reason_codes", "solve_total_evaluations",
        "final_validation_passed", "final_validation_failures", "has_any_offer",
    ]

    @frame_step(reads=reads, writes=writes)
    def granting_and_pricing(df):
        import polars as pl
        results = []
        for app in df.select(reads).to_dicts():
            evaluator = PriceEvaluator(rate_card_index, credit_life_index, app["statutory_ceiling"],
                                        app.get("credit_life_substitution_declared", False))
            is_new_to_bank = app.get("segment_code") == 3
            terms = _permitted_terms(app["term_cap"], is_new_to_bank)
            term_results = _run_solve_for_every_term(evaluator, app, terms, int(app["evaluation_ceiling_per_term"]))
            offer_set = build_offer_set(app, term_results, app["recommendation_objective"], app["probability_of_default"])
            recommended = next((o for o in offer_set["offers"] if o["is_recommended"]), None)
            passed, failures = final_validation(evaluator, app, recommended)
            total_evaluations = sum(r.evaluation_count for r in term_results.values())
            suppression_codes = sorted({c for codes in offer_set["suppression_reasons"] for c in codes})
            offers = offer_set["offers"]  # ragged (0..9); as parallel lists, not list[struct] -- see NOTES.md.
            results.append({
                "offer_term_months": [o["term_months"] for o in offers],
                "offer_amounts": [o["offered_amount"] for o in offers],
                "offer_nominal_rates": [o["nominal_annual_rate"] for o in offers],
                "offer_instalments": [o["instalment"] for o in offers],
                "offer_total_costs_of_credit": [o["total_cost_of_credit"] for o in offers],
                "offer_effective_rates": [o["effective_annual_rate"] for o in offers],
                "offer_is_recommended": [o["is_recommended"] for o in offers],
                "offer_binding_constraint_codes": [o["binding_constraint_code"] for o in offers],
                "recommended_term": offer_set["recommended_term"] or 0,
                "recommended_amount": recommended["offered_amount"] if recommended else 0.0,
                "recommended_instalment": recommended["instalment"] if recommended else 0.0,
                "recommended_total_cost_of_credit": recommended["total_cost_of_credit"] if recommended else 0.0,
                "recommended_effective_annual_rate": recommended["effective_annual_rate"] if recommended else 0.0,
                "recommended_binding_constraint_code": recommended["binding_constraint_code"] if recommended
                else (BIND_AFF if offer_set["suppressed_terms"] else ""),
                "suppressed_terms": offer_set["suppressed_terms"],
                "suppression_reason_codes": suppression_codes,
                "solve_total_evaluations": total_evaluations,
                "final_validation_passed": passed and (recommended is not None or not offer_set["offers"]),
                "final_validation_failures": failures,
                "has_any_offer": recommended is not None,
            })
        return df.with_columns(pl.DataFrame(results))

    return granting_and_pricing
