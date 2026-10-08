"""Why an account got no increase, ranked by severity through `credit_core`'s reason registry."""
from __future__ import annotations

from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

R_EXCLUDED = 7001
R_NO_INCREASE_AFTER_CAPS = 7002
R_AFFORDABILITY_FAIL = 7003
R_BELOW_MINIMUM_MEANINGFUL_INCREASE = 7004
R_BELOW_FUNDING_LINE = 7005
R_REDUCED_BELOW_MINIMUM_BY_OVERLAY = 7006
R_SUPPRESSED_BY_CLIENT_DECREASE = 7007
R_FAIRNESS_CAPPED = 7008
R_TAIL_SKIPPED = 7009
R_AFFORDABILITY_UNESTABLISHED = 7010

REASON_REGISTRY = ReasonCodeRegistry("limit-mgmt-reasons-2026.09", [
    ReasonCode(R_EXCLUDED, 5, "Account excluded from the automatic programme", True),
    ReasonCode(R_AFFORDABILITY_FAIL, 10, "The client cannot carry any of the increase", True),
    ReasonCode(R_AFFORDABILITY_UNESTABLISHED, 12, "Affordability could not be established on the evidence available", True),
    ReasonCode(R_SUPPRESSED_BY_CLIENT_DECREASE, 15, "Suppressed: an immediate decrease fired on another account of this client", True),
    ReasonCode(R_REDUCED_BELOW_MINIMUM_BY_OVERLAY, 25, "Reduced below the minimum meaningful increase by an active overlay", False),
    ReasonCode(R_NO_INCREASE_AFTER_CAPS, 30, "The matrix and the caps leave no increase, or one under the cell's minimum increment", False),
    ReasonCode(R_BELOW_MINIMUM_MEANINGFUL_INCREASE, 35, "Increase below the minimum meaningful amount", False),
    ReasonCode(R_BELOW_FUNDING_LINE, 40, "Ranked below the funding line this cycle", False),
    ReasonCode(R_TAIL_SKIPPED, 42, "Did not fit the remaining budget; skipped", False),
    ReasonCode(R_FAIRNESS_CAPPED, 45, "Segment fairness floor reached", False),
])
