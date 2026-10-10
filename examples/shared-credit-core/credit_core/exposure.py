"""Group exposure as at a date, against a limit. The caller resolves which facilities are related."""
from __future__ import annotations

from datetime import date
from typing import TypedDict

from decikit import allow_fallback, param, step


class Facility(TypedDict, total=False):
    balance: float
    opened_date: date
    closed_date: date
    is_contingent: bool
    conversion_factor: float  # the share of a contingent facility counted as exposure


@allow_fallback
def group_total_exposure(related_facilities: list[Facility], exposure_as_of_date: date) -> float:
    """Balances of the facilities open on `exposure_as_of_date`; contingent ones count at their conversion factor."""
    total = 0.0
    for f in related_facilities:
        opened, closed = f.get("opened_date"), f.get("closed_date")
        if (opened and opened > exposure_as_of_date) or (closed and closed <= exposure_as_of_date):
            continue
        total += (f.get("balance") or 0.0) * (f.get("conversion_factor", 1.0) if f.get("is_contingent") else 1.0)
    return round(total, 2)


def group_exposure_limit(limit: float = param(5_000_000.0, ge=0.0)) -> float:
    return limit


def exposure_headroom(group_total_exposure: float, group_exposure_limit: float) -> float:
    return round(max(0.0, group_exposure_limit - group_total_exposure), 2)


group_total_exposure_step = step(group_total_exposure)
group_exposure_limit_step = step(group_exposure_limit)
exposure_headroom_step = step(exposure_headroom)
