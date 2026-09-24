"""`core.adverse_events` -- adverse event classification (spec 00 §6.14; addendum A12).

Classifies **one** adverse event into a severity; roll-up across a
collection is each consumer's own job (00 §6.14 "Hard part": results roll
up differently by count, by worst, by weighted amount in each consumer, so
this capability stays single-event and callable that way, addendum item
12). Amount thresholds are supplied by the caller per criticality class,
not baked into the library (addendum item 12).

14 event types (addendum: "not 7"): the 7 named in 00 §6.14 plus 7 more
representative of the same regulatory surface -- illustrative, per this
project's front matter.
"""
from __future__ import annotations

from decider import missing_as, param, step

# event_type_code
JUDGMENT = 1
DEFAULT_LISTING = 2
ADMINISTRATION_ORDER = 3
DEBT_REVIEW = 4
TAX_NON_COMPLIANCE = 5
INSOLVENCY = 6
LITIGATION = 7
SEQUESTRATION = 8
CURATORSHIP = 9
GARNISHEE_ORDER = 10
DEBT_COUNSELLING_WITHDRAWAL = 11
FRAUD_CONVICTION = 12
FOREIGN_ADVERSE_LISTING = 13
REGULATORY_FINDING = 14

IMMATERIAL = 1
MINOR = 2
MATERIAL = 3
DISQUALIFYING = 4

# event types that disqualify outright regardless of amount or age.
_ALWAYS_DISQUALIFYING = {SEQUESTRATION, FRAUD_CONVICTION, ADMINISTRATION_ORDER}
# event types with no amount dimension (status-only).
_STATUS_ONLY = {DEBT_REVIEW, CURATORSHIP, DEBT_COUNSELLING_WITHDRAWAL, REGULATORY_FINDING}


def event_severity_code(
    event_type_code: int, amount: float = missing_as(0.0), status_is_active: bool = missing_as(True),
    disputed: bool = missing_as(False), satisfied: bool = missing_as(False),
    material_threshold: float = param(5_000.0, ge=0.0), disqualifying_threshold: float = param(50_000.0, ge=0.0),
) -> int:
    """Caller supplies `material_threshold`/`disqualifying_threshold` per criticality class (addendum item 12)."""
    if satisfied and event_type_code not in _ALWAYS_DISQUALIFYING:
        return IMMATERIAL
    if disputed and not status_is_active:
        return MINOR
    if event_type_code in _ALWAYS_DISQUALIFYING and status_is_active:
        return DISQUALIFYING
    if event_type_code in _STATUS_ONLY:
        return MATERIAL if status_is_active else MINOR
    if amount >= disqualifying_threshold:
        return DISQUALIFYING
    if amount >= material_threshold:
        return MATERIAL
    if amount > 0:
        return MINOR
    return IMMATERIAL


def event_age_months(event_date, decision_date) -> int:
    return max(0, (decision_date.year - event_date.year) * 12 + (decision_date.month - event_date.month))


event_severity_code_step = step(event_severity_code)
event_age_months_step = step(event_age_months)
