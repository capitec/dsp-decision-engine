"""Locally declared names (spec 06 §4.7). The library does not publish these -- they
are declared here once and used unchanged wherever this flow is consumed (11 in
particular: DEPS.md "11 consumes ... [06] Obligation inventory and settleability").
"""
from __future__ import annotations

# assessment_mode_code (06 §4.1) -- only mode 1 is built this slice (SCOPE.md: skip
# most of the restructure variant and all of batch identification).
MODE_CONSOLIDATION_ON_REQUEST = 1
MODE_CONSOLIDATION_OFFERED = 2
MODE_RESTRUCTURE = 3
MODE_BATCH_IDENTIFICATION = 4

# settleability_code (06 §5.2), first-match-wins order, most-blocked first so a
# fully-blocked account never accidentally matches a laxer rule below it.
SETTLE_UNKNOWN = 7
SETTLE_BLOCKED_STATUS = 6
SETTLE_BLOCKED_POLICY = 5
SETTLE_BLOCKED_PROVIDER = 4
SETTLE_SECURITY_RELEASE = 3
SETTLE_PARTIAL_REVOLVING = 8
SETTLE_QUOTABLE = 2
SETTLE_QUOTED = 1
SETTLE_INTERNAL = 0

# The codes that make an account a *candidate* for a settlement set at all.
SETTLEABLE_CODES = frozenset({SETTLE_INTERNAL, SETTLE_QUOTED, SETTLE_QUOTABLE,
                               SETTLE_PARTIAL_REVOLVING, SETTLE_SECURITY_RELEASE})

# amount_basis_code (06 §5.3 "Emits")
BASIS_QUOTED = "quoted"
BASIS_DERIVED_INTERNAL = "derived-internal"
BASIS_ESTIMATED = "estimated"

# product_code (06 §5.6.3) -- this slice builds 11 and 20 only.
PRODUCT_FLEX_CONSOLIDATION = 11
PRODUCT_BALANCE_TRANSFER = 20
PRODUCT_DRIVE_REFINANCE = 30    # routing recognised, pricing out of scope (SCOPE.md)
PRODUCT_HOME_FURTHER_ADVANCE = 40  # routing recognised, pricing out of scope (SCOPE.md)

# objective_id (06 §5.8)
OBJ_NEW_MONEY = "OBJ-01"
OBJ_MIN_COMMITMENT = "OBJ-02"
OBJ_MIN_TOTAL_COST = "OBJ-03"
OBJ_BANK_VALUE = "OBJ-04"
OBJ_CLIENT_OUTCOME = "OBJ-05"

# outcome_code
OUTCOME_APPROVE = "approve"
OUTCOME_APPROVE_WITH_CONDITIONS = "approve_with_conditions"
OUTCOME_REFER = "refer"
OUTCOME_DECLINE = "decline"

# termination_cause (06 §5.5 requirement 2)
TERMINATION_EXHAUSTED = "candidates_exhausted"
TERMINATION_BUDGET = "budget_exhausted"

# rejection_reason_code (06 §4.7 -- distinct from decline_reason_codes) registry, see reasons.py
