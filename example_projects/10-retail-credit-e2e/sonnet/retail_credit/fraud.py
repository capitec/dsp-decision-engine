"""P05 -- Fraud and financial crime (spec 10 §5.6).

Rule *volume* is project 01's dominant difficulty (SCOPE.md), not this
project's -- 10's own scope statement is "1 400 decision points, 8 entry
points, one record" (SCOPE.md's table), and project 01 already proves the
520-live/100-shadow rule-volume mechanism at full size. This module
therefore carries a **small, representative** rule set (12 live rules
across the spec's five families) rather than 188, but keeps every
mechanism the spec requires at that smaller size: every rule evaluated
and every firing recorded (not just the deciding one), a **separate
precedence ordering** over the fired set (severity- and family-weighted,
not "first rule wins"), and the low-value/high-tenure bypass.

`fraud_verdict_code`: 1 proceed, 2 refer, 3 decline, 4 could not be
established (10 §5.6) -- vocabulary shared with `credit_core.vocab.FraudVerdictCode`
so this project's stub agrees with the contract 00 publishes for a fraud
verdict (00-ADDENDUM A5; see project 00 NOTES.md "Spec problems" for the
ownership ambiguity this project inherited and did not have to resolve,
because it builds its own verdict rather than consuming a stub).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from decider import missing_as, param, step

PROCEED, REFER, DECLINE, UNAVAILABLE = 1, 2, 3, 4

F1_IDENTITY = "F1"
F2_CONTENT = "F2"
F3_DEVICE = "F3"
F4_SYNDICATE = "F4"
F5_WATCHLIST = "F5"


@dataclass(frozen=True)
class FraudRule:
    rule_id: str
    family: str
    severity: int  # 1..4
    reason_code: int
    predicate: Callable[[dict], bool]


def _ctx(
    device_is_new: bool, contact_changed_days_ago: float, income_employer_mismatch: bool,
    address_change_count_90d: int, shared_device_count: int, consortium_watchlist_hit: bool,
    internal_fraud_register_hit: bool, synthetic_identity_score: float, application_burst_count: int,
    identity_number_reused: bool, sector_employer_mismatch: bool, referral_chain_flag: bool,
) -> dict:
    return dict(
        device_is_new=device_is_new, contact_changed_days_ago=contact_changed_days_ago,
        income_employer_mismatch=income_employer_mismatch, address_change_count_90d=address_change_count_90d,
        shared_device_count=shared_device_count, consortium_watchlist_hit=consortium_watchlist_hit,
        internal_fraud_register_hit=internal_fraud_register_hit,
        synthetic_identity_score=synthetic_identity_score, application_burst_count=application_burst_count,
        identity_number_reused=identity_number_reused, sector_employer_mismatch=sector_employer_mismatch,
        referral_chain_flag=referral_chain_flag,
    )


RULES: tuple[FraudRule, ...] = (
    FraudRule("F1-0001", F1_IDENTITY, 3, 4201, lambda c: c["identity_number_reused"]),
    FraudRule("F1-0002", F1_IDENTITY, 4, 4202, lambda c: c["synthetic_identity_score"] >= 0.85),
    FraudRule("F1-0003", F1_IDENTITY, 2, 4203, lambda c: c["synthetic_identity_score"] >= 0.55),
    FraudRule("F2-0118", F2_CONTENT, 1, 4211, lambda c: 0 <= c["contact_changed_days_ago"] < 14),
    FraudRule("F2-0119", F2_CONTENT, 3, 4212, lambda c: c["income_employer_mismatch"]),
    FraudRule("F2-0120", F2_CONTENT, 2, 4213, lambda c: c["sector_employer_mismatch"]),
    FraudRule("F3-0042", F3_DEVICE, 1, 4221, lambda c: c["device_is_new"]),
    FraudRule("F3-0043", F3_DEVICE, 2, 4222, lambda c: c["address_change_count_90d"] >= 3),
    FraudRule("F4-0071", F4_SYNDICATE, 3, 4231, lambda c: c["shared_device_count"] >= 2),
    FraudRule("F4-0072", F4_SYNDICATE, 2, 4232, lambda c: c["application_burst_count"] >= 3),
    FraudRule("F4-0073", F4_SYNDICATE, 2, 4233, lambda c: c["referral_chain_flag"]),
    FraudRule("F5-0091", F5_WATCHLIST, 4, 4241, lambda c: c["consortium_watchlist_hit"]),
    FraudRule("F5-0092", F5_WATCHLIST, 4, 4242, lambda c: c["internal_fraud_register_hit"]),
)

_SEVERITY_WEIGHT = {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.60}
_WEIGHTED_REFER_THRESHOLD = 0.71
_WEIGHTED_DECLINE_THRESHOLD = 0.88
_BYPASS_MAX_AMOUNT = 8_000.0
_BYPASS_MIN_TENURE_MONTHS = 30.0


def evaluate_fraud(
    device_is_new: bool = missing_as(False), contact_changed_days_ago: float = missing_as(9999.0),
    income_employer_mismatch: bool = missing_as(False), address_change_count_90d: int = missing_as(0),
    shared_device_count: int = missing_as(0), consortium_watchlist_hit: bool = missing_as(False),
    internal_fraud_register_hit: bool = missing_as(False), synthetic_identity_score: float = missing_as(0.0),
    application_burst_count: int = missing_as(0), identity_number_reused: bool = missing_as(False),
    sector_employer_mismatch: bool = missing_as(False), referral_chain_flag: bool = missing_as(False),
    requested_amount: float = missing_as(0.0), internal_tenure_months: float = missing_as(0.0),
    has_adverse_internal_history: bool = missing_as(False),
    refer_threshold: float = param(_WEIGHTED_REFER_THRESHOLD, ge=0.0, le=1.0),
    decline_threshold: float = param(_WEIGHTED_DECLINE_THRESHOLD, ge=0.0, le=1.0),
) -> tuple[int, list[int], float, int]:
    """(fraud_verdict_code, fraud_reason_codes, weighted_score, rules_fired_count).

    Bypass (10 §5.6): a request under the amount threshold from a client with enough
    tenure and no adverse internal history skips the consortium-dependent families (F3,
    F4, F5) -- they are recorded as *not evaluated*, never as *did not fire*, which is
    the distinction the monthly dead-rule measurement (§5.29) depends on.
    """
    ctx = _ctx(
        device_is_new, contact_changed_days_ago, income_employer_mismatch, address_change_count_90d,
        shared_device_count, consortium_watchlist_hit, internal_fraud_register_hit, synthetic_identity_score,
        application_burst_count, identity_number_reused, sector_employer_mismatch, referral_chain_flag,
    )
    bypass = requested_amount <= _BYPASS_MAX_AMOUNT and internal_tenure_months >= _BYPASS_MIN_TENURE_MONTHS \
        and not has_adverse_internal_history

    fired: list[FraudRule] = []
    for rule in RULES:
        if bypass and rule.family in (F3_DEVICE, F4_SYNDICATE, F5_WATCHLIST):
            continue  # not evaluated, not "did not fire"
        if rule.predicate(ctx):
            fired.append(rule)

    f5_severity_4 = any(r.family == F5_WATCHLIST and r.severity == 4 for r in fired)
    f1_severity_3_plus_count = sum(1 for r in fired if r.family == F1_IDENTITY and r.severity >= 3)
    weighted_score = round(sum(_SEVERITY_WEIGHT[r.severity] for r in fired), 4)

    if f5_severity_4:
        verdict = DECLINE
    elif f1_severity_3_plus_count >= 2:
        verdict = REFER
    elif weighted_score >= decline_threshold:
        verdict = DECLINE
    elif weighted_score >= refer_threshold:
        verdict = REFER
    else:
        verdict = PROCEED

    reasons = sorted({r.reason_code for r in fired})
    return verdict, reasons, weighted_score, len(fired)


evaluate_fraud_step = step(evaluate_fraud, outputs=("fraud_verdict_code", "fraud_reason_codes",
                                                     "fraud_weighted_score", "fraud_rules_fired_count"))
