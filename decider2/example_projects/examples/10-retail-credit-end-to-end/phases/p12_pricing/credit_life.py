"""P12(c) credit life — charged on the amount FINANCED (fees.py's output),
not the amount advanced. Spec 5.13(c). O-14: after fees, before the annuity.

The client's statutory substitution right zeroes this and lowers the
instalment — which is why the affordability re-test in P17 must read
`basis_of("instalment")` rather than assume a premium was charged
(values/bases.py, O-18 in ordering.py).
"""

from __future__ import annotations

from decider2 import module, Table, param

class CreditLifeRateTable(Table):
    key: tuple[int, int, int]   # (age_band, term_band, employment_type) — 480 cells
    rate_per_1000: float

def credit_life_rate(age_years: int, term_months: int, employment_type_code: int,
                     credit_life_rates: CreditLifeRateTable) -> float:
    """R1.70 to R4.25 per R1 000. R4.25 is the statutory ceiling; a cell above
    it is a card DEFECT, never a price."""
    pass  # credit_life_rates[(age_band(age_years), term_band(term_months), employment_type_code)]

def credit_life_substitution_elected(substitution_right_exercised: bool = False) -> bool:
    pass  # whether the client has elected to substitute their own policy

def credit_life_premium(amount_financed: float, credit_life_rate: float,
                        credit_life_substitution_elected: bool) -> float:
    """Zero where substituted. Otherwise (amount_financed / 1000) * rate."""
    pass  # 0.0 if substituted else (amount_financed / 1000.0) * credit_life_rate

Premium = module(credit_life_rate, credit_life_substitution_elected, credit_life_premium,
                 name="credit_life")
