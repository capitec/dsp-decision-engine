"""Severity of one adverse event. Rolling a collection up is each consumer's job, so this takes a single event."""
from __future__ import annotations

from datetime import date

from decikit import missing_as, param, step

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
EVENT_TYPES = tuple(range(JUDGMENT, REGULATORY_FINDING + 1))

# event_severity_code
IMMATERIAL = 1
MINOR = 2
MATERIAL = 3
DISQUALIFYING = 4

_ALWAYS_DISQUALIFYING = (SEQUESTRATION, FRAUD_CONVICTION, ADMINISTRATION_ORDER)  # while active, whatever the amount
_STATUS_ONLY = (DEBT_REVIEW, CURATORSHIP, DEBT_COUNSELLING_WITHDRAWAL, REGULATORY_FINDING)  # no amount dimension


def event_severity_code(
    event_type_code: int, amount: float = missing_as(0.0), status_is_active: bool = missing_as(True),
    disputed: bool = missing_as(False), satisfied: bool = missing_as(False),
    material_threshold: float = param(5_000.0, ge=0.0), disqualifying_threshold: float = param(50_000.0, ge=0.0),
) -> int:
    """The severity of one event. The caller sets the amount thresholds, for example per criticality class."""
    always = event_type_code in _ALWAYS_DISQUALIFYING
    if satisfied and not always:
        return IMMATERIAL
    if disputed and not status_is_active:
        return MINOR
    if always and status_is_active:
        return DISQUALIFYING
    if event_type_code in _STATUS_ONLY:
        return MATERIAL if status_is_active else MINOR
    if amount >= disqualifying_threshold:
        return DISQUALIFYING
    if amount >= material_threshold:
        return MATERIAL
    return MINOR if amount > 0 else IMMATERIAL


def event_age_months(event_date: date, decision_date: date) -> int:
    return max(0, (decision_date.year - event_date.year) * 12 + (decision_date.month - event_date.month))


event_severity_code_step = step(event_severity_code)
event_age_months_step = step(event_age_months)
