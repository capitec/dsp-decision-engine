"""§5.2 -- hard exclusions. Eight of the spec's sixteen codes, at working depth
(SCOPE.md: "a subset of the hard exclusions"), every one still evaluated and
recorded -- not merely the first that applies (§5.2: "Every exclusion that
applied, not merely the first one found... The requirement is both: complete
attribution, and no scoring, affordability or bureau work performed for an
excluded account").

This slice keeps the first half of that requirement (complete attribution:
every code is evaluated and `exclusion_codes` names all of them) and does
**not** implement the short-circuit that skips scoring/matrix/affordability
work for an excluded account -- doing that inside one served `decider`
pipeline needs a `branch()` arm around every downstream stage, which is a
real structural cost for an efficiency property this slice's acceptance
bar does not test. Recorded under NOTES.md "What I left out".

X16 is project 08's treatment-state feed (DEPS.md: "soft... a data feed,
used only by exclusion X16... a stub is enough"): this module reads
`treatment_suspension_active` (bool) exactly as 08 would publish it, so a
real feed slots in by populating that one field.
"""
from __future__ import annotations

from datetime import date

from decider import missing_as, param, step

from limit_mgmt.vocab import Exclusion


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
    months_on_book: int = missing_as(0),
    treatment_suspension_active: bool = missing_as(False),
    cooling_off_days_card: int = param(180, ge=0, description="X09: days since last change, card"),
    cooling_off_days_facility: int = param(180, ge=0),
    minimum_months_on_book: int = param(6, ge=0),
) -> list[int]:
    """Every exclusion that applied, evaluated independently (§5.2)."""
    codes: list[int] = []
    if worst_arrears_months_now and worst_arrears_months_now >= 1:
        codes.append(Exclusion.X01_ARREARS_NOW)
    if worst_arrears_months_6 and worst_arrears_months_6 >= 1:
        codes.append(Exclusion.X02_ARREARS_6M)
    if debt_review_active:
        codes.append(Exclusion.X03_DEBT_REVIEW)
    if deceased_marker:
        codes.append(Exclusion.X05_DECEASED)
    if fraud_marker:
        codes.append(Exclusion.X06_FRAUD)
    if last_change_date is not None:
        window = cooling_off_days_card
        elapsed = (decision_date - last_change_date).days
        if elapsed < window:
            codes.append(Exclusion.X09_COOLING_OFF)
    if (not consent_automatic_increase) or consent_withdrawn:
        codes.append(Exclusion.X12_NO_CONSENT)
    if months_on_book < minimum_months_on_book:
        codes.append(Exclusion.X14_TOO_YOUNG)
    if treatment_suspension_active:
        codes.append(Exclusion.X16_TREATMENT_SUSPENSION)
    return codes


def is_excluded(exclusion_codes: list[int]) -> bool:
    codes = [] if exclusion_codes is None else exclusion_codes
    return len(codes) > 0


exclusion_codes_step = step(exclusion_codes, output="exclusion_codes")
is_excluded_step = step(is_excluded)
