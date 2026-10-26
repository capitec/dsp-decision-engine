"""Account state derived from the cycle history: the figures the matrix and the spend cap key off.

The cycle lists are ragged (a young account has fewer than six) and newest first.
"""
from __future__ import annotations

from decikit import allow_fallback, flow, missing_as


@allow_fallback
def revolving_utilisation_6m(cycle_balances: list[float] = missing_as([]), cycle_limits: list[float] = missing_as([])) -> float:
    """Mean of balance over limit across the latest six cycles that had a limit; 0 with none."""
    ratios = [b / limit for b, limit in zip(cycle_balances, cycle_limits) if limit][:6]
    return round(sum(ratios) / len(ratios), 4) if ratios else 0.0


@allow_fallback
def observed_spend_p90(cycle_purchase_values: list[float] = missing_as([])) -> float:
    """The 90th percentile of monthly purchases over the latest six cycles (nearest rank)."""
    ordered = sorted(cycle_purchase_values[:6])
    return round(ordered[round(0.90 * (len(ordered) - 1))], 2) if ordered else 0.0


@allow_fallback
def cycle_bucket_count(cycle_balances: list[float] = missing_as([])) -> int:
    return len(cycle_balances)


def utilisation_band(revolving_utilisation_6m: float) -> int:
    """1 is exactly 0%, then 0-10%, 10-25%, 25-40%, 40-55%, 55-70%, 70-90% and 8 is 90% and above; lower edge closed."""
    u = revolving_utilisation_6m
    if u <= 0.0:
        return 1
    return 2 + (u >= 0.10) + (u >= 0.25) + (u >= 0.40) + (u >= 0.55) + (u >= 0.70) + (u >= 0.90)


def mob_band(months_on_book: float) -> int:
    """1 is under 12 months, then 12-17, 18-23, 24-35, 36-59 and 6 is 60 and over."""
    m = months_on_book
    return 1 + (m >= 12) + (m >= 18) + (m >= 24) + (m >= 36) + (m >= 60)


unit = flow(revolving_utilisation_6m, observed_spend_p90, cycle_bucket_count, utilisation_band, mob_band, name="population")
