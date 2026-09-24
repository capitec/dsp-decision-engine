from __future__ import annotations
"""Money and figure rounding (00 §6.20)."""
from decimal import Decimal, ROUND_HALF_UP


def round_instalment(amount: float, decimals: int = 2) -> float:
    """
    Round an instalment to the nearest cent.
    Used for all monetary figures to ensure determinism (09 §8.4).
    """
    d = Decimal(str(amount))
    rounded = d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP)
    return float(rounded)


def round_rate(rate: float, decimals: int = 4) -> float:
    """Round an interest rate to 4 decimal places (basis points)."""
    d = Decimal(str(rate))
    rounded = d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP)
    return float(rounded)


def round_percentage(pct: float, decimals: int = 2) -> float:
    """Round a percentage to 2 decimal places."""
    d = Decimal(str(pct))
    rounded = d.quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP)
    return float(rounded)
