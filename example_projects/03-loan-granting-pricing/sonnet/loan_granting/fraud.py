"""Stage 5.2 -- consent and the fraud handoff (spec 03 §5.2, §4.4).

Project 01 (application fraud) does not exist yet (DEPS.md: "03 does not
publish what 03 expects from it... stub it with the three-field verdict
contract"). The verdict itself -- `fraud_verdict_code`, `fraud_reason_codes`,
`fraud_response_ms` -- is therefore an **input** to this pipeline, exactly as
`bureau_response` is an input in project 00/02's pipelines: some upstream
gateway (project 01, in production) supplies it synchronously before this
flow runs. `credit_core.vocab.FraudVerdictCode` is project 00's own
addendum-A5 vocabulary for the same three fields; imported here so 03 does
not invent a second spelling of it.

What 03 *does* own is the handling logic in §5.2's table: what to do with
each of the four verdict codes, including the 800 ms timeout's low-risk
bypass. That is real decision logic, not a stub.
"""
from __future__ import annotations

from decider import missing_as, param, step

from credit_core.vocab import FraudVerdictCode

CONTINUE = "continue"
FORCED_REFER = "forced_refer"
DECLINE = "decline"
BYPASSED = "bypassed"

QUEUE_CONSENT = 1
QUEUE_DQ = 2
QUEUE_FRAUD_REFER = 3
QUEUE_NO_HIT_MANUAL = 4
QUEUE_AFFORDABILITY_INDETERMINATE = 5
QUEUE_EVALUATION_CEILING = 6
QUEUE_FINAL_VALIDATION = 7

_BYPASS_AMOUNT_CEILING = 15_000.0
_BYPASS_TENURE_FLOOR_MONTHS = 24.0


def fraud_handling_path(
    fraud_verdict_code: int,
    fraud_response_ms: float = missing_as(0.0),
    fraud_timeout_ms: float = param(800.0, ge=0.0),
    requested_amount: float = missing_as(0.0),
    internal_tenure_months: float = missing_as(0.0),
    has_adverse_internal_history: bool = missing_as(False),
    bypass_amount_ceiling: float = param(_BYPASS_AMOUNT_CEILING, ge=0.0),
    bypass_tenure_floor_months: float = param(_BYPASS_TENURE_FLOOR_MONTHS, ge=0.0),
) -> str:
    """Which of §5.2's four handling paths this application takes."""
    timed_out = fraud_response_ms > fraud_timeout_ms
    code = FraudVerdictCode.UNAVAILABLE if timed_out else FraudVerdictCode(fraud_verdict_code)

    if code == FraudVerdictCode.DECLINE:
        return DECLINE
    if code == FraudVerdictCode.REFER:
        return FORCED_REFER
    if code == FraudVerdictCode.UNAVAILABLE:
        low_risk = (
            requested_amount < bypass_amount_ceiling
            and internal_tenure_months >= bypass_tenure_floor_months
            and not has_adverse_internal_history
        )
        return BYPASSED if low_risk else FORCED_REFER
    return CONTINUE  # APPROVE


def fraud_declined(fraud_handling_path: str) -> bool:
    return fraud_handling_path == DECLINE


def fraud_forced_refer(fraud_handling_path: str) -> bool:
    return fraud_handling_path == FORCED_REFER


def fraud_bypass_applied(fraud_handling_path: str) -> bool:
    return fraud_handling_path == BYPASSED


fraud_handling_path_step = step(fraud_handling_path)
fraud_declined_step = step(fraud_declined)
fraud_forced_refer_step = step(fraud_forced_refer)
fraud_bypass_applied_step = step(fraud_bypass_applied)
