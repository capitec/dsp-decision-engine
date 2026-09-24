"""Reason codes for this project's own decline/non-selection outcomes, through
`core.reason_codes` (unmodified) -- the same registry mechanism 00 and 02 use,
never a second reason-code vocabulary (09 §5.15 item 11)."""
from __future__ import annotations

from credit_core import reason_codes

R_EXCLUDED = 7001
R_MATRIX_ZERO_OR_BELOW_MINIMUM_INCREMENT = 7002
R_AFFORDABILITY_FAIL = 7003
R_BELOW_MINIMUM_MEANINGFUL_INCREASE = 7004
R_BELOW_FUNDING_LINE = 7005
R_REDUCED_BELOW_MINIMUM_BY_OVERLAY = 7006
R_SUPPRESSED_BY_CLIENT_DECREASE = 7007
R_FAIRNESS_CAPPED = 7008
R_TAIL_SKIPPED = 7009

REASON_REGISTRY = reason_codes.ReasonCodeRegistry("limit-mgmt-reasons-2026.09", [
    reason_codes.ReasonCode(R_EXCLUDED, 5, "Account excluded from the automatic programme", True),
    reason_codes.ReasonCode(R_MATRIX_ZERO_OR_BELOW_MINIMUM_INCREMENT, 30,
                             "Matrix proposes no increase, or below the cell's minimum increment", False),
    reason_codes.ReasonCode(R_AFFORDABILITY_FAIL, 10, "Affordability assessment did not pass", True),
    reason_codes.ReasonCode(R_BELOW_MINIMUM_MEANINGFUL_INCREASE, 35, "Increase below the minimum meaningful amount", False),
    reason_codes.ReasonCode(R_BELOW_FUNDING_LINE, 40, "Ranked below the funding line this cycle", False),
    reason_codes.ReasonCode(R_REDUCED_BELOW_MINIMUM_BY_OVERLAY, 25,
                             "Reduced below the minimum meaningful increase by an active overlay", False),
    reason_codes.ReasonCode(R_SUPPRESSED_BY_CLIENT_DECREASE, 15,
                             "Suppressed: a decrease fired on another account of this client this cycle", True),
    reason_codes.ReasonCode(R_FAIRNESS_CAPPED, 45, "Segment fairness floor reached", False),
    reason_codes.ReasonCode(R_TAIL_SKIPPED, 42, "Budget tail-skipped; did not fit remaining budget", False),
])
