"""`core.rounding` -- money conventions (spec 00 §6.20).

Stated as its own capability because a pricing step and a contract step
disagreeing on rounding direction is a real, cent-level defect class. Every
other capability in this library imports these instead of calling `round()`
itself.
"""
from __future__ import annotations

import math

from decider import step


def round_advance(amount: float) -> float:
    """Advances round to the nearest R100."""
    return round(amount / 100.0) * 100.0


def round_instalment(amount: float) -> float:
    """Instalments round to the cent, half up (never in the client's favour by chance)."""
    return math.floor(amount * 100.0 + 0.5) / 100.0


def round_rate(rate: float) -> float:
    """Rates round to four decimal places."""
    return round(rate, 4)


# Step wrappers, for use directly in a pipeline where the column is already
# the right name (e.g. a final `offered_amount` or `instalment` column).
round_advance_step = step(round_advance, output="offered_amount")
round_instalment_step = step(round_instalment, output="instalment")
round_rate_step = step(round_rate, output="nominal_annual_rate")
