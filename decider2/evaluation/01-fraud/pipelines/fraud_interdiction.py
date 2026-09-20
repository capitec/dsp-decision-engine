"""
Real-time fraud interdiction pipeline.

Evaluates a flat set of fraud rules across multiple families and determines
the most severe action to take. This is a simplified implementation of the spec
in ../../../example_projects/01-transaction-fraud-interdiction.md

Rules are organized by family:
- CF: Card fraud
- MS: Mule and scam
- AT: Account takeover
"""

from decider2 import step, param, flow, missing_as
from pydantic import BaseModel, Field


class SharedParams(BaseModel):
    """Global parameters shared across all rules."""
    model_score_threshold: float = Field(
        750.0, ge=0, le=1000, description="Model score threshold for decline decision"
    )
    max_transaction_multiplier: float = Field(
        5.0, ge=1, le=10, description="Velocity multiplier for decline threshold"
    )


# ============================================================================
# CARD FRAUD FAMILY (CF)
# ============================================================================


@step(output="cf_0101")
def cf_0101_velocity_spike(
    transaction_count_1h: int,
    historic_max_1h: int,
    threshold_multiplier: float = param(
        5.0, ge=2, le=10, description="How many times historic max before flagging"
    ),
) -> bool:
    """High transaction velocity in 1 hour window.

    Implements: Credit Policy Section 2.1.1
    """
    if historic_max_1h == 0:
        return transaction_count_1h > 10
    return transaction_count_1h > (historic_max_1h * threshold_multiplier)


@step(output="cf_0102")
def cf_0102_amount_spike(
    transaction_amount: float,
    historic_max_amount: float = missing_as(10000.0),
    spike_factor: float = param(3.0, ge=1.5, le=8.0, description="How many times average"),
) -> bool:
    """Single transaction amount significantly above historic average."""
    return transaction_amount > (historic_max_amount * spike_factor)


@step(output="cf_0103")
def cf_0103_unusual_merchant(
    merchant_risk_band: int,
    client_prior_merchants: int = missing_as(0),
    risk_band_threshold: int = param(4, ge=1, le=5, description="Risk band to flag (5 highest)"),
) -> bool:
    """High-risk merchant category not in client prior usage."""
    return merchant_risk_band >= risk_band_threshold and client_prior_merchants == 0


# ============================================================================
# MULE AND SCAM FAMILY (MS)
# ============================================================================


@step(output="ms_0208")
def ms_0208_first_payment_new_beneficiary(
    beneficiary_age_hours: int,
    transaction_amount: float,
    device_change_hours: int = missing_as(999),
    amount_threshold: float = param(8000.0, ge=1000, le=50000, description="Amount threshold ZAR"),
    age_threshold_hours: int = param(2, ge=1, le=24, description="Beneficiary age threshold hours"),
    device_change_threshold: int = param(72, ge=1, le=720, description="Device change threshold hours"),
) -> bool:
    """First payment to recently added beneficiary with amount and device change.

    Implements: Credit Policy Section 3.2.1 - Mule/scam interdiction
    """
    return (
        beneficiary_age_hours < age_threshold_hours
        and transaction_amount > amount_threshold
        and device_change_hours < device_change_threshold
    )


@step(output="ms_0209")
def ms_0209_rapid_beneficiary_cycling(
    beneficiary_count_24h: int,
    unique_beneficiaries_7d: int,
    cycle_count_threshold: int = param(5, ge=2, le=20, description="Number of beneficiaries in 24h"),
    total_threshold: int = param(15, ge=5, le=50, description="Total unique in 7d"),
) -> bool:
    """Rapidly adding and paying many different beneficiaries."""
    return (
        beneficiary_count_24h >= cycle_count_threshold
        and unique_beneficiaries_7d >= total_threshold
    )


# ============================================================================
# ACCOUNT TAKEOVER FAMILY (AT)
# ============================================================================


@step(output="at_0301")
def at_0301_device_change_high_velocity(
    transaction_count_since_device_change: int,
    hours_since_device_change: int,
    velocity_threshold: int = param(
        50, ge=10, le=200, description="Transaction count threshold"
    ),
) -> bool:
    """High transaction velocity immediately after device change."""
    return (
        hours_since_device_change < 24
        and transaction_count_since_device_change >= velocity_threshold
    )


# ============================================================================
# ACTION RESOLUTION - determine action based on fired rules
# ============================================================================

def compute_action_code(
    cf_0101: bool,
    cf_0102: bool,
    cf_0103: bool,
    ms_0208: bool,
    ms_0209: bool,
    at_0301: bool,
    model_score: float = missing_as(0.0),
    shared=None,
) -> int:
    """Determine the action code based on fired rules.

    Action precedence (most severe first):
    70 = freeze_account
    60 = block_channel
    50 = decline
    40 = hold_for_review
    30 = step_up
    20 = monitor
    10 = allow
    """
    max_action = 10  # default allow

    # Check each rule and track the most severe action
    # Rules are checked in order of severity

    if at_0301:  # block_channel
        max_action = max(max_action, 60)

    if ms_0208:  # decline
        max_action = max(max_action, 50)

    if ms_0209:  # hold_for_review
        max_action = max(max_action, 40)

    if cf_0101:  # hold_for_review
        max_action = max(max_action, 40)

    if cf_0102:  # step_up
        max_action = max(max_action, 30)

    if cf_0103:  # monitor
        max_action = max(max_action, 20)

    # Check model score if no rules fired
    if max_action == 10 and shared is not None and model_score < shared.model_score_threshold:
        max_action = 50  # decline based on model score

    return max_action


@step(output="fired_rule_count")
def count_fired_rules(
    cf_0101: bool,
    cf_0102: bool,
    cf_0103: bool,
    ms_0208: bool,
    ms_0209: bool,
    at_0301: bool,
) -> int:
    """Count how many rules fired."""
    return sum([cf_0101, cf_0102, cf_0103, ms_0208, ms_0209, at_0301])


@step(output="max_severity")
def determine_max_severity(
    cf_0101: bool,
    cf_0102: bool,
    cf_0103: bool,
    ms_0208: bool,
    ms_0209: bool,
    at_0301: bool,
) -> int:
    """Determine the maximum severity among fired rules.

    Severity: CF=3, MS=4, AT=3
    """
    max_sev = 0

    if cf_0101 or cf_0102 or cf_0103:
        max_sev = max(max_sev, 3)

    if ms_0208 or ms_0209:
        max_sev = max(max_sev, 4)

    if at_0301:
        max_sev = max(max_sev, 3)

    return max_sev


# ============================================================================
# PIPELINE ASSEMBLY
# ============================================================================

# Build the pipeline using flow() for simple rule evaluation
pipeline = flow(
    cf_0101_velocity_spike,
    cf_0102_amount_spike,
    cf_0103_unusual_merchant,
    ms_0208_first_payment_new_beneficiary,
    ms_0209_rapid_beneficiary_cycling,
    at_0301_device_change_high_velocity,
    count_fired_rules,
    determine_max_severity,
).emit("fired_rule_count", "max_severity", "cf_0101", "ms_0208", "at_0301")
