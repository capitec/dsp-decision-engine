"""`core.fees` -- statutory fee schedule (spec 00 §6.7).

A small piecewise function of `offered_amount` with a cap, effective-dated
because the caps are gazetted and inflation-adjusted annually. Ten lines of
arithmetic that need their own evidence trail because getting it wrong is a
regulatory finding, not a bug report -- so the capped-vs-calculated flag is
a required output, not a debugging nicety.
"""
from __future__ import annotations

from datetime import date

from decider import param, step

from credit_core.dates import EffectiveDatedSet, EffectiveVersion

# The initiation fee is R1000 + 10% of the amount over R10000, capped;
# the monthly service fee is a flat amount, capped. Rates below are
# illustrative (spec front matter): fictional, invented for this repo.
FEE_CAPS = EffectiveDatedSet(
    "fee_caps",
    [
        EffectiveVersion("fee_caps@2025", date(2025, 1, 1), date(2026, 1, 1)),
        EffectiveVersion("fee_caps@2026", date(2026, 1, 1), None),
    ],
)


def initiation_fee(
    offered_amount: float,
    base_fee: float = param(1050.0, ge=0.0),
    marginal_rate: float = param(0.10, ge=0.0, le=1.0),
    threshold: float = param(10000.0, ge=0.0),
    cap: float = param(6500.0, ge=0.0),
) -> float:
    raw = base_fee + marginal_rate * max(0.0, offered_amount - threshold)
    return min(raw, cap)


def initiation_fee_capped(initiation_fee: float, cap: float = param(6500.0, ge=0.0)) -> bool:
    return initiation_fee >= cap


def monthly_service_fee(
    offered_amount: float,
    flat_fee: float = param(69.0, ge=0.0),
    cap: float = param(69.0, ge=0.0),
) -> float:
    return min(flat_fee, cap)


initiation_fee_step = step(initiation_fee)
initiation_fee_capped_step = step(initiation_fee_capped)
monthly_service_fee_step = step(monthly_service_fee)
fee_caps_version_step = FEE_CAPS.resolver_step()
