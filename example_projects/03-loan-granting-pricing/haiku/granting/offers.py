"""Offer set construction and final validation."""

from typing import List, Optional, Dict, Any
from .types import TermResult, OfferOutput, OfferSuppressionReasonCode


def construct_offer_set(
    term_results: List[TermResult],
    pricing_fn,  # (amount, term) -> dict with rate, fees, premium, instalment, etc.
    min_amount: float = 2000,
    min_instalment: float = 150.0,
    total_cost_ratio_threshold: float = 1.85,
    effective_rate_ceiling: float = 0.60,  # 60%
    scheduled_in_duplum_threshold: float = 1.0,
    recommendation_objective: str = "largest_amount",  # or "lowest_total_cost", "best_expected_value"
    probability_of_default: float = 0.05,
) -> tuple[List[OfferOutput], List[Dict[str, Any]]]:
    """
    Construct offer set from solve results.

    Applies minimum viable offer rules and builds the final ranked set.

    Returns:
    - offers: List of OfferOutput
    - suppressions: List of suppressed terms with reasons
    """

    offers: List[OfferOutput] = []
    suppressions: List[Dict[str, Any]] = []

    for term_result in term_results:
        if term_result.affordable_amount is None:
            # Term produced no feasible amount
            suppression_reasons = [OfferSuppressionReasonCode.EVAL_CEILING.value]
            suppressions.append({
                "term_months": term_result.term_months,
                "reason_codes": suppression_reasons,
                "primary_reason": suppression_reasons[0],
            })
            continue

        # Get pricing for the winning amount
        pricing = term_result.pricing
        if pricing is None:
            pricing = pricing_fn(term_result.affordable_amount, term_result.term_months)

        amount = term_result.affordable_amount
        instalment = pricing.get("instalment", 0)
        total_cost = pricing.get("total_cost", 0)

        # Apply minimum viable offer rules
        suppression_reasons = []

        if amount < min_amount:
            suppression_reasons.append(OfferSuppressionReasonCode.MIN_AMOUNT.value)

        if instalment < min_instalment:
            suppression_reasons.append(OfferSuppressionReasonCode.MIN_INSTALMENT.value)

        # Total cost ratio
        if amount > 0:
            tcr = total_cost / amount
            if tcr > total_cost_ratio_threshold:
                suppression_reasons.append(OfferSuppressionReasonCode.TOTAL_COST_RATIO.value)

        # Scheduled in duplum (charges should not exceed advance)
        charges = total_cost - amount
        if amount > 0 and (charges / amount) > scheduled_in_duplum_threshold:
            suppression_reasons.append(OfferSuppressionReasonCode.SCHEDULED_IN_DUPLUM.value)

        # Effective annual rate ceiling
        ear = pricing.get("effective_annual_rate", 0)
        if ear > effective_rate_ceiling:
            suppression_reasons.append(OfferSuppressionReasonCode.EAR_CEILING.value)

        if suppression_reasons:
            suppressions.append({
                "term_months": term_result.term_months,
                "reason_codes": suppression_reasons,
                "primary_reason": suppression_reasons[0],
            })
            continue

        # Offer survived suppression rules
        offer = OfferOutput(
            term_months=term_result.term_months,
            offered_amount=amount,
            rate_cell_id=pricing.get("cell_id", ""),
            nominal_annual_rate=pricing.get("rate", 0),
            initiation_fee=pricing.get("initiation_fee", 0),
            monthly_service_fee=pricing.get("monthly_service_fee", 0),
            credit_life_premium=pricing.get("credit_life_premium", 0),
            instalment=instalment,
            total_cost_of_credit=total_cost,
            effective_annual_rate=ear,
            total_cost_ratio=total_cost / amount if amount > 0 else 0,
            binding_constraint=term_result.binding_constraint or "",
        )
        offers.append(offer)

    # Deduplication (2% instalment tolerance)
    # For now, skip this for simplicity

    # Ranking and recommendation
    if offers:
        if recommendation_objective == "largest_amount":
            # Sort by amount desc, then total cost asc, then term asc
            offers.sort(key=lambda o: (-o.offered_amount, o.total_cost_of_credit, o.term_months))
        elif recommendation_objective == "lowest_total_cost":
            # Sort by total cost ratio (total cost per rand advanced)
            offers.sort(key=lambda o: (o.total_cost_ratio, -o.offered_amount, o.term_months))
        elif recommendation_objective == "best_expected_value":
            # Sort by expected margin (simple heuristic)
            # margin = total_cost - advance - funding_cost * advance, adjusted by PD
            lgd = 0.72
            avg_ead = 0.55
            funding_cost = 0.0875
            for o in offers:
                margin = (o.total_cost_of_credit - o.offered_amount - funding_cost * o.offered_amount * o.term_months / 12)
                margin = margin * (1 - probability_of_default * o.term_months / 12)
                margin = margin - lgd * avg_ead * o.offered_amount * probability_of_default * o.term_months / 12
                o.expected_margin = margin  # Attach for sorting
            offers.sort(key=lambda o: (-getattr(o, "expected_margin", 0), -o.offered_amount, o.term_months))

        # Mark first as recommended
        offers[0].is_recommended = True

    return offers, suppressions


def validate_offer(
    offer: OfferOutput,
    max_affordable_instalment: float,
    amount_cap: float,
    term_cap: int,
    risk_grade: int,
    worst_acceptable_grade: int,
) -> tuple[bool, List[str]]:
    """
    Validate that an offer is correct.

    Returns (is_valid, error_messages)
    """
    errors = []

    if offer.instalment > max_affordable_instalment:
        errors.append(f"Instalment {offer.instalment} exceeds max affordable {max_affordable_instalment}")

    if offer.offered_amount > amount_cap:
        errors.append(f"Amount {offer.offered_amount} exceeds cap {amount_cap}")

    if offer.term_months > term_cap:
        errors.append(f"Term {offer.term_months} exceeds cap {term_cap}")

    if risk_grade > worst_acceptable_grade:
        errors.append(f"Grade {risk_grade} worse than acceptable {worst_acceptable_grade}")

    if offer.offered_amount % 100 != 0:
        errors.append(f"Amount {offer.offered_amount} not rounded to R100")

    return len(errors) == 0, errors
