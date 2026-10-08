"""What to do with the application-fraud verdict, which arrives with the request.

The fraud engine is a separate service, called before this flow. Its answer is three request fields:
`fraud_verdict_code`, `fraud_reason_codes` and `fraud_response_ms`. This module decides what the answer means here.
"""
from __future__ import annotations

from decider import missing_as, param, step

from credit_core.vocab import FraudVerdictCode

CONTINUE, FORCED_REFER, DECLINE, BYPASSED = 1, 2, 3, 4  # fraud_handling_code


def fraud_handling_code(
    fraud_verdict_code: int, fraud_response_ms: float = missing_as(0.0),
    requested_amount: float = missing_as(0.0), internal_tenure_months: float = missing_as(0.0),
    has_adverse_internal_history: bool = missing_as(False),
    fraud_timeout_ms: float = param(800.0, ge=0.0), bypass_amount_ceiling: float = param(15_000.0, ge=0.0),
    bypass_tenure_floor_months: float = param(24.0, ge=0.0),
) -> int:
    """Continue on an approval, decline on a decline, refer on a refer.

    A verdict that took longer than `fraud_timeout_ms` is `UNAVAILABLE`, which is not an approval: it refers, unless
    the application is a small loan to a long-standing client with a clean internal history, which bypasses the check.
    """
    unavailable = fraud_response_ms > fraud_timeout_ms or fraud_verdict_code == FraudVerdictCode.UNAVAILABLE
    if unavailable:
        low_risk = (requested_amount < bypass_amount_ceiling and internal_tenure_months >= bypass_tenure_floor_months
                    and not has_adverse_internal_history)
        return BYPASSED if low_risk else FORCED_REFER
    if fraud_verdict_code == FraudVerdictCode.DECLINE:
        return DECLINE
    if fraud_verdict_code == FraudVerdictCode.REFER:
        return FORCED_REFER
    return CONTINUE


def fraud_bypass_applied(fraud_handling_code: int) -> bool:
    return fraud_handling_code == BYPASSED


fraud_handling_code_step = step(fraud_handling_code)
