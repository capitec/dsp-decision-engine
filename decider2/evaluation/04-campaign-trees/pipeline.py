"""Campaign targeting trees implementation.

A simplified version of the campaign targeting decision trees from doc 04.
Exercises core requirements:
- Multi-level decision trees with sequential decision points
- Tree depth (4 decision levels)
- Targeting decisions based on multiple checks

Simplified from the worked example in doc 04 section 5.3.3 (campaign 23).

Note on path capture (doc 04 section 5.4): Path recording in this implementation
is documented through decision logic but not explicitly emitted as a string,
since numba (used in fused mode) cannot handle Python strings directly.
The path is implicitly recorded through the sequence of boolean checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the decider2 shim package to path
# File location: /home/sholto/.../decider2/evaluation/04-campaign-trees/pipeline.py
# Want to add: /home/sholto/.../decider2 (the shim dir with decider2/__init__.py and decider2/)
_decider2_shim = Path(__file__).parent.parent.parent.resolve()
if str(_decider2_shim) not in sys.path:
    sys.path.insert(0, str(_decider2_shim))

from decider2 import flow


# Level 1: Basic facility check
def check_facility(
    has_active_flex_loan: bool,
    months_on_book_flex: int,
) -> bool:
    """Check if client has active flex loan with sufficient tenure."""
    return has_active_flex_loan and months_on_book_flex >= 6


# Level 2: Payment history
def check_payment_history(
    settlement_ratio: float,
    payments_missed_12m: int,
) -> bool:
    """Check settlement ratio and missed payments."""
    return settlement_ratio <= 0.65 and payments_missed_12m == 0


# Level 3: Credit score check
def check_credit_score(
    behaviour_score: float,
) -> bool:
    """Check if behaviour score meets minimum threshold."""
    return behaviour_score >= 600.0


# Level 4: Affordability check
def check_affordability(
    discretionary_income: float,
    estimated_instalment_to_income: float,
) -> bool:
    """Check discretionary income and affordability ratio."""
    return discretionary_income >= 2500.0 and estimated_instalment_to_income <= 0.33


# Final targeting outcome
def targeting_decision(
    check_facility: bool,
    check_payment_history: bool,
    check_credit_score: bool,
    check_affordability: bool,
) -> int:
    """Final targeting decision.

    Returns: 1 if target, 0 if do not target.
    This is the leaf outcome from the tree.
    """
    if (check_facility and check_payment_history and check_credit_score and
        check_affordability):
        return 1
    return 0


# Build the complete pipeline
pipeline = flow(
    check_facility,
    check_payment_history,
    check_credit_score,
    check_affordability,
    targeting_decision,
).emit("targeting_decision")
