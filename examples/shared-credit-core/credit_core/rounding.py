"""Money rounding in one place, so a pricing step and a contract step cannot disagree by a cent."""
from __future__ import annotations

from decikit import step
from decikit.ops import decimal


def round_advance(amount: float) -> float:
    """Advances round to the nearest R100, half up."""
    return decimal.round(amount / 100.0, 0) * 100.0


def round_instalment(amount: float) -> float:
    """Instalments round to the cent, half up."""
    return decimal.round(amount, 2)


def round_rate(rate: float) -> float:
    """Rates round to four decimal places."""
    return round(rate, 4)


round_advance_step = step(round_advance, output="offered_amount")
round_instalment_step = step(round_instalment, output="instalment")
round_rate_step = step(round_rate, output="nominal_annual_rate")
