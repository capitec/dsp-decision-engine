"""P11 -- Product routing (spec 10 §5.12): product 10 only.

Full-width P11 fans out across six products with substitution and
preference logic (10 §5.12); this project's slice runs product 10 alone
(SCOPE.md), so fan-out, substitution and preference collapse to one
eligibility gate. The genuine cycle this phase sits in (O-09: amount
depends on the solve, the solve depends on affordability, affordability's
buffer depends on the product) is still real even with one product -- it
is *why* `pipeline.py` runs P10 twice (product-neutral, then routed) --
but with one candidate product the routed buffer and the product-neutral
buffer are the same table lookup, so the second pass is a no-op
arithmetically while still being a real second pass structurally (proven
by `tests/test_pipeline.py::test_p10_runs_twice_around_p11`).
"""
from __future__ import annotations

from decider import missing_as, param, step

R_AMOUNT_OUT_OF_RANGE = 1201
R_TERM_OUT_OF_RANGE = 1202
R_EVIDENCE_TIER_TOO_LOW = 1203
R_GRADE_BELOW_FLOOR = 1204

PRODUCT_10_MIN_AMOUNT = 2_000.0
PRODUCT_10_MAX_AMOUNT = 500_000.0
PRODUCT_10_MIN_TERM = 6
PRODUCT_10_MAX_TERM = 84


def product_10_eligible(
    requested_amount: float, term_months: int, risk_grade: int, income_evidence_tier: int = missing_as(1),
    min_amount: float = param(PRODUCT_10_MIN_AMOUNT), max_amount: float = param(PRODUCT_10_MAX_AMOUNT),
    min_term: int = param(PRODUCT_10_MIN_TERM), max_term: int = param(PRODUCT_10_MAX_TERM),
    grade_floor: int = param(12, ge=1, le=12),
) -> tuple[bool, list[int]]:
    reasons = []
    if not (min_amount <= requested_amount <= max_amount):
        reasons.append(R_AMOUNT_OUT_OF_RANGE)
    if not (min_term <= term_months <= max_term):
        reasons.append(R_TERM_OUT_OF_RANGE)
    if income_evidence_tier == 7:
        reasons.append(R_EVIDENCE_TIER_TOO_LOW)
    if risk_grade > grade_floor:
        reasons.append(R_GRADE_BELOW_FLOOR)
    return len(reasons) == 0, reasons


product_10_eligible_step = step(product_10_eligible, outputs=("product_10_eligible", "product_routing_reasons"))
