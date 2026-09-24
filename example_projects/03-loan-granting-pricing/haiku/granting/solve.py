"""Bounded solve for maximum affordable amount across terms."""

from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from .types import TermResult, ConstraintCode, PricingEvaluation


@dataclass
class SolveCandidate:
    """Candidate amount and its pricing."""
    amount: float
    instalment: float
    total_cost: float
    rate_cell_id: str
    affordable: bool


def bounded_solve(
    min_amount: float = 2000,
    max_amount: float = 500000,
    amount_cap: float = 500000,
    requested_amount: Optional[float] = None,
    max_affordable_instalment: float = 5000,
    permitted_terms: List[int] = None,
    pricing_fn: Callable[[float, int], Optional[dict]] = None,  # (amount, term_months) -> {"rate", "fees", "premium", "instalment", "total_cost", "cell_id"}
    term_cap: int = 84,
    rounding_unit: float = 100,
    eval_ceiling: int = 24,
    tiebreak_by_total_cost: bool = True,  # If tied on amount, prefer lower total cost
) -> List[TermResult]:
    """
    Bounded solve for maximum affordable amount at each permitted term.

    Key properties:
    - Non-monotone affordability (band-edge inversions in rate card)
    - Hard ceiling of eval_ceiling evaluations per term
    - Deterministic: same inputs -> same outputs
    - Records every evaluation for evidence

    Args:
        pricing_fn: (amount, term) -> dict with rate, fees, premium, instalment, total_cost, cell_id
                   Returns None if amount is out of bounds for the card

    Returns list of TermResult, one per permitted term.
    """

    if permitted_terms is None:
        permitted_terms = [6, 12, 18, 24, 36, 48, 60, 72, 84]

    permitted_terms = [t for t in permitted_terms if t <= term_cap]
    results = []

    # Set the search ceiling
    search_max = min(requested_amount or max_amount, amount_cap, max_amount)

    for term_months in permitted_terms:
        evaluations: List[PricingEvaluation] = []
        best_affordable: Optional[Tuple[float, float, str]] = None  # (amount, total_cost, cell_id)

        eval_count = 0
        binding_constraint = None

        # Start from search_max, round down to nearest rounding_unit, work downward
        current = int(search_max / rounding_unit) * rounding_unit

        # Scan candidates downward until ceiling reached or bottom reached
        while current >= min_amount and eval_count < eval_ceiling:
            eval_count += 1

            # Get pricing for this candidate
            pricing = pricing_fn(current, term_months)
            if pricing is None:
                # Out of bounds; try lower
                current -= rounding_unit
                continue

            instalment = pricing.get("instalment", 0)
            affordable = instalment <= max_affordable_instalment

            evaluation = PricingEvaluation(
                candidate_amount=current,
                rate_cell_id=pricing.get("cell_id", ""),
                nominal_rate=pricing.get("rate", 0),
                advance=current,
                financed=pricing.get("financed", 0),
                initiation_fee=pricing.get("initiation_fee", 0),
                monthly_service_fee=pricing.get("monthly_service_fee", 0),
                credit_life_premium=pricing.get("credit_life_premium", 0),
                instalment=instalment,
                affordable=affordable,
            )
            evaluations.append(evaluation)

            # Track best feasible
            if affordable:
                if best_affordable is None:
                    best_affordable = (current, pricing.get("total_cost", 0), pricing.get("cell_id", ""))
                    # Keep going; there might be a better one further down (due to band inversions)
                elif current > best_affordable[0]:
                    # Larger amount is better; update
                    best_affordable = (current, pricing.get("total_cost", 0), pricing.get("cell_id", ""))
                # else: current is smaller, keep looking

            current -= rounding_unit

        # Determine binding constraint
        if best_affordable is None:
            # No feasible amount found
            if eval_count >= eval_ceiling:
                binding_constraint = ConstraintCode.EVAL_CEILING.value
            elif current < min_amount:
                binding_constraint = ConstraintCode.MINIMUM.value
            else:
                binding_constraint = ConstraintCode.AFFORDABILITY.value

            results.append(TermResult(
                term_months=term_months,
                affordable_amount=None,
                binding_constraint=binding_constraint,
                evaluation_count=eval_count,
                evaluations=evaluations,
            ))
        else:
            # Found a feasible amount
            amount, total_cost, cell_id = best_affordable

            # Determine what bound it
            if amount >= amount_cap:
                binding_constraint = ConstraintCode.CAP.value
            elif amount >= search_max:
                binding_constraint = ConstraintCode.REQUESTED.value if requested_amount and amount >= requested_amount else ConstraintCode.MAXIMUM.value
            else:
                binding_constraint = ConstraintCode.AFFORDABILITY.value

            # Build full pricing for the winning amount
            winning_pricing = pricing_fn(amount, term_months)

            results.append(TermResult(
                term_months=term_months,
                affordable_amount=amount,
                binding_constraint=binding_constraint,
                evaluation_count=eval_count,
                evaluations=evaluations,
                pricing=winning_pricing,
            ))

    return results


def validate_solve_correctness(
    term_result: TermResult,
    domain_min: float = 2000,
    domain_max: float = 500000,
    rounding_unit: float = 100,
) -> Tuple[bool, str]:
    """
    Validate that the solve result is correct by checking:
    1. Returned amount is within [domain_min, domain_max]
    2. Returned amount is a multiple of rounding_unit
    3. Returned amount is actually affordable
    4. No larger amount in the same evaluation set is also affordable

    Returns (is_valid, error_message)
    """

    if term_result.affordable_amount is None:
        # No feasible amount case - check that evaluations show nothing was affordable
        has_affordable = any(e.affordable for e in term_result.evaluations)
        if has_affordable:
            return False, "No feasible amount returned but evaluations show affordable candidates"
        return True, ""

    amount = term_result.affordable_amount

    # Check bounds
    if amount < domain_min or amount > domain_max:
        return False, f"Amount {amount} outside domain [{domain_min}, {domain_max}]"

    # Check rounding
    if amount % rounding_unit != 0:
        return False, f"Amount {amount} not a multiple of {rounding_unit}"

    # Check affordability
    is_affordable = any(e.candidate_amount == amount and e.affordable for e in term_result.evaluations)
    if not is_affordable:
        return False, f"Returned amount {amount} not in evaluations or not marked affordable"

    # Check for larger affordable amounts
    for eval_ in term_result.evaluations:
        if eval_.candidate_amount > amount and eval_.affordable:
            return False, f"Larger affordable amount {eval_.candidate_amount} exists than returned {amount}"

    return True, ""
