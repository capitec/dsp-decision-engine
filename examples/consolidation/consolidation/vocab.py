"""Codes this example declares; the shared library does not publish them."""
from credit_core.vocab import (  # noqa: F401  (re-exported so callers have one place to look)
    OUTCOME_APPROVE, OUTCOME_DECLINE, OUTCOME_REFER, PRODUCT_EVERYDAY_CARD, PRODUCT_FLEX_LOAN,
    PRODUCT_FLEX_LOAN_CONSOLIDATION)

# settleability_code, in the order the rules are tried (first match wins)
UNKNOWN, BLOCKED_BY_STATUS, BLOCKED_BY_POLICY, BLOCKED_BY_PROVIDER = 7, 6, 5, 4
SECURITY_RELEASE, PARTIAL_REVOLVING, QUOTABLE, QUOTED, INTERNAL = 3, 8, 2, 1, 0
SETTLEABLE = frozenset({INTERNAL, QUOTED, QUOTABLE, SECURITY_RELEASE, PARTIAL_REVOLVING})

CARD_TYPES = frozenset({10, 11})  # accounts a balance transfer can take
REVOLVING_TYPES = frozenset({10, 11, 12})
CLOSED_STATUS = 2  # account_status_code

# amount_basis
QUOTED_BASIS, DERIVED_BASIS, ESTIMATED_BASIS = "quoted", "derived-internal", "estimated"

# the five objectives, in the order the weight columns list them
OBJECTIVES = ("new_money", "min_commitment", "min_total_cost", "bank_value", "client_outcome")
CLIENT_OUTCOME = "client_outcome"

# search termination
EXHAUSTED, OUT_OF_BUDGET = "candidates_exhausted", "budget_exhausted"
