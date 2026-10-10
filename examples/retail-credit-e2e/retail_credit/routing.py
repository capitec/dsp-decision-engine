"""Product routing: whether this product may carry the request. One product here, so one gate."""
from __future__ import annotations

from decikit import allow_fallback, param, step

R_AMOUNT_OUT_OF_RANGE = 1201
R_TERM_OUT_OF_RANGE = 1202
R_GRADE_WORSE_THAN_ACCEPTABLE = 1204


def product_range(
    min_amount: float = param(2_000.0, ge=0.0), max_amount: float = param(500_000.0, ge=0.0),
    min_term: int = param(6, ge=1), max_term: int = param(84, ge=1), grid: float = param(250.0, gt=0.0),
) -> tuple[float, float, int, int, float]:
    """The Flex Loan's range and the grid its advances round to, stated once for the caps, routing, solve and validation."""
    return min_amount, max_amount, min_term, max_term, grid


@allow_fallback
def product_eligibility(
    requested_amount: float, term_months: int, risk_grade: int, worst_acceptable_grade: int, product_min_amount: float,
    product_max_amount: float, product_min_term: int, product_max_term: int,
) -> tuple[bool, list[int]]:
    """(product_10_eligible, product_routing_reasons): amount and term in the product's range, and a grade no worse
    than the policy caps allow."""
    failed = [
        (not product_min_amount <= requested_amount <= product_max_amount, R_AMOUNT_OUT_OF_RANGE),
        (not product_min_term <= term_months <= product_max_term, R_TERM_OUT_OF_RANGE),
        (risk_grade > worst_acceptable_grade, R_GRADE_WORSE_THAN_ACCEPTABLE),
    ]
    reasons = [code for hit, code in failed if hit]
    return not reasons, reasons


product_range_step = step(product_range, outputs=("product_min_amount", "product_max_amount", "product_min_term", "product_max_term", "amount_grid"))
product_eligibility_step = step(product_eligibility, outputs=("product_10_eligible", "product_routing_reasons"))
