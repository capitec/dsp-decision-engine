"""P12(b) fees — the initiation fee's piecewise shape. Spec 5.13(b). O-14
pins this AFTER the rate and BEFORE credit life: the premium is charged on
the amount FINANCED, which includes this fee, capitalised.

The R14 568 kink sits inside a rate band, which is exactly why P13's solve
cannot be a binary search (phases/p13_solve/search.py) — this file is where
that non-monotonicity is born, not where it is handled.
"""

from __future__ import annotations

from decider2 import module, param

def initiation_fee_excl_tax(offered_amount: float,
                            base_fee: float = param(210_00, ge=0),
                            marginal_rate: float = param(0.095, ge=0, le=1),
                            threshold: float = param(1_200_00, ge=0),
                            cap: float = param(1_480_00, ge=0)) -> float:
    """R210 + 9.50% of the advance above R1 200, capped at R1 480. The cap
    binds at exactly R14 568 — declared here as the arithmetic, not as a
    breakpoint; see phases/p13_solve/search.py for the declared breakpoint
    this creates."""
    pass  # min(base_fee + marginal_rate * max(0, offered_amount - threshold), cap)

def initiation_fee_incl_tax(initiation_fee_excl_tax: float,
                            indirect_tax_rate: float = param(0.15, ge=0)) -> float:
    pass  # initiation_fee_excl_tax * (1 + indirect_tax_rate)

def monthly_service_fee(decision_date: str,
                        base_fee_excl_tax: float = param(76_50, ge=0)) -> float:
    """R76.50 excl., R87.98 incl., statutorily capped, effective-dated,
    adjusted annually."""
    pass  # base_fee_excl_tax * (1 + indirect_tax_rate), capped at the statutory fee cap

def amount_financed(offered_amount: float, initiation_fee_incl_tax: float) -> float:
    """A DIFFERENT number from offered_amount, and the disclosure (P18) names
    both. This is the base credit_life.py's premium is computed on."""
    pass  # offered_amount + initiation_fee_incl_tax

Fees = module(initiation_fee_excl_tax, initiation_fee_incl_tax, monthly_service_fee,
              amount_financed, name="fees")
