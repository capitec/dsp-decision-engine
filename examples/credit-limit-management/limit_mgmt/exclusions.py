"""Hard exclusions: every one that applies is recorded, not just the first.

Decreases ignore exclusions (an account in arrears is excluded from increases and is a candidate for a decrease).
"""
from __future__ import annotations

from datetime import date

from decikit import allow_fallback, flow, missing_as, param

from limit_mgmt.vocab import Exclusion


@allow_fallback
def exclusion_codes(
    decision_date: date,
    worst_arrears_months_now: int = missing_as(0),
    worst_arrears_months_6: int = missing_as(0),
    debt_review_active: bool = missing_as(False),
    deceased_marker: bool = missing_as(False),
    fraud_marker: bool = missing_as(False),
    last_change_date: date | None = None,
    consent_automatic_increase: bool = missing_as(False),
    consent_withdrawn: bool = missing_as(False),
    months_on_book: float = missing_as(0.0),
    treatment_suspension_active: bool = missing_as(False),
    cooling_off_days: int = param(180, ge=0),
    minimum_months_on_book: int = param(6, ge=0),
) -> list[int]:
    """The codes of every exclusion that applies. `treatment_suspension_active` is the collections team's feed."""
    codes = []
    if worst_arrears_months_now >= 1:
        codes.append(Exclusion.X01_ARREARS_NOW)
    if worst_arrears_months_6 >= 1:
        codes.append(Exclusion.X02_ARREARS_6M)
    if debt_review_active:
        codes.append(Exclusion.X03_DEBT_REVIEW)
    if deceased_marker:
        codes.append(Exclusion.X05_DECEASED)
    if fraud_marker:
        codes.append(Exclusion.X06_FRAUD)
    if last_change_date is not None and (decision_date - last_change_date).days < cooling_off_days:
        codes.append(Exclusion.X09_COOLING_OFF)
    if consent_withdrawn or not consent_automatic_increase:
        codes.append(Exclusion.X12_NO_CONSENT)
    if months_on_book < minimum_months_on_book:
        codes.append(Exclusion.X14_TOO_YOUNG)
    if treatment_suspension_active:
        codes.append(Exclusion.X16_TREATMENT_SUSPENSION)
    return [int(c) for c in codes]


@allow_fallback
def is_excluded(exclusion_codes: list[int]) -> bool:
    return len(exclusion_codes) > 0


unit = flow(exclusion_codes, is_excluded, name="exclusions")
