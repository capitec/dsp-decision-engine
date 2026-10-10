"""Application fraud: every rule is checked, every firing is recorded, and the verdict has its own precedence."""
from __future__ import annotations

from decikit import allow_fallback, missing_as, param, step

from credit_core.vocab import FraudVerdictCode

# (rule id, family, severity 1..4, reason code, the feature it reads, its condition)
_RULES = (
    ("F1-0001", "F1", 3, 4201, "identity_number_reused", bool),
    ("F1-0002", "F1", 4, 4202, "synthetic_identity_score", lambda v: v >= 0.85),
    ("F1-0003", "F1", 2, 4203, "synthetic_identity_score", lambda v: v >= 0.55),
    ("F2-0118", "F2", 1, 4211, "contact_changed_days_ago", lambda v: 0 <= v < 14),
    ("F2-0119", "F2", 3, 4212, "income_employer_mismatch", bool),
    ("F2-0120", "F2", 2, 4213, "sector_employer_mismatch", bool),
    ("F3-0042", "F3", 1, 4221, "device_is_new", bool),
    ("F3-0043", "F3", 2, 4222, "address_change_count_90d", lambda v: v >= 3),
    ("F4-0071", "F4", 3, 4231, "shared_device_count", lambda v: v >= 2),
    ("F4-0072", "F4", 2, 4232, "application_burst_count", lambda v: v >= 3),
    ("F4-0073", "F4", 2, 4233, "referral_chain_flag", bool),
    ("F5-0091", "F5", 4, 4241, "consortium_watchlist_hit", bool),
    ("F5-0092", "F5", 4, 4242, "internal_fraud_register_hit", bool),
)
_SEVERITY_WEIGHT = (0.05, 0.15, 0.30, 0.60)  # by severity 1 to 4
_CONSORTIUM_FAMILIES = ("F3", "F4", "F5")  # skipped on the low-value bypass: no consortium call is made


# ponytail: a rule is a feature and a Python condition; `TreeConfig` in "all" mode would move the conditions into a
# config document.
@allow_fallback
def evaluate_fraud(
    device_is_new: bool = missing_as(False), contact_changed_days_ago: float = missing_as(9999.0),
    income_employer_mismatch: bool = missing_as(False), address_change_count_90d: int = missing_as(0),
    shared_device_count: int = missing_as(0), consortium_watchlist_hit: bool = missing_as(False),
    internal_fraud_register_hit: bool = missing_as(False), synthetic_identity_score: float = missing_as(0.0),
    application_burst_count: int = missing_as(0), identity_number_reused: bool = missing_as(False),
    sector_employer_mismatch: bool = missing_as(False), referral_chain_flag: bool = missing_as(False),
    requested_amount: float = missing_as(0.0), internal_tenure_months: float = missing_as(0.0),
    has_adverse_internal_history: bool = missing_as(False),
    refer_threshold: float = param(0.71, ge=0.0, le=1.0), decline_threshold: float = param(0.88, ge=0.0, le=1.0),
    bypass_max_amount: float = param(8_000.0, ge=0.0), bypass_min_tenure_months: float = param(30.0, ge=0.0),
) -> tuple[int, list[int], float, int]:
    """(fraud_verdict_code, fraud_reason_codes, fraud_weighted_score, fraud_rules_fired_count).

    A severity-4 watchlist hit declines; two severity-3 identity hits refer; otherwise the weighted score of
    everything that fired refers or declines at its thresholds. A small request from a long-standing client with
    clean history bypasses the consortium families, which are then not evaluated rather than not fired.
    """
    features = dict(locals())  # the rules name their feature after the argument that carries it
    bypass = (requested_amount <= bypass_max_amount and internal_tenure_months >= bypass_min_tenure_months
              and not has_adverse_internal_history)
    fired = [(family, severity, reason) for _, family, severity, reason, feature, condition in _RULES
             if not (bypass and family in _CONSORTIUM_FAMILIES) and condition(features[feature])]
    score = round(sum(_SEVERITY_WEIGHT[severity - 1] for _, severity, _ in fired), 4)
    if any(family == "F5" and severity == 4 for family, severity, _ in fired):
        verdict = FraudVerdictCode.DECLINE
    elif sum(family == "F1" and severity >= 3 for family, severity, _ in fired) >= 2:
        verdict = FraudVerdictCode.REFER
    elif score >= decline_threshold:
        verdict = FraudVerdictCode.DECLINE
    elif score >= refer_threshold:
        verdict = FraudVerdictCode.REFER
    else:
        verdict = FraudVerdictCode.APPROVE
    return int(verdict), sorted({reason for _, _, reason in fired}), score, len(fired)


evaluate_fraud_step = step(evaluate_fraud, outputs=("fraud_verdict_code", "fraud_reason_codes", "fraud_weighted_score",
                                                     "fraud_rules_fired_count"))
