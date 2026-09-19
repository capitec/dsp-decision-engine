"""P12(a) the rate — four cards, three representations, one phase. Spec 5.13(a).
83 decision points total for P12; this file is the largest slice.

Three representations (absolute, margin-over-reference, promotional-plus-
reversion) are a deliberate Treasury choice that must REMAIN VISIBLE as one —
`representations=` on the P12 envelope (phases/__init__.py) declares the
mapping from rule_id ranges to representation kind, so "the rate" is never
collapsed into a single shape the record can't tell apart.

Called as a BODY by P13 (up to 152 times) and P14 (up to 400 times) — neither
of which owns this file. `per_call_budget_us=18.0` on the envelope is a
correctness-adjacent performance constraint imposed on a phase by its callers.
"""

from __future__ import annotations

from decider2 import module, Table, param

class RateCardTable(Table):
    key: tuple[int, int, int]   # (amount_band, term_column, risk_grade)
    rate: float

def card_cell(product_code: int, offered_amount: float, term_months: int,
             risk_grade: int, rate_cards: RateCardTable, decision_date: str) -> float:
    """The card version in force at decision_date. Long-term loading
    (a separate 3x12 table) applies for terms 46-84, priced from the
    45-month column plus the loading."""
    pass  # look up rate_cards[(amount_band(offered_amount), term_column(term_months), risk_grade)]

def rate_add_on(card_cell: float, adjustment_set_id: int, product_code: int,
                term_months: int, offered_amount: float, channel_code: int) -> float:
    """Basis points over a declared scope range. AFTER the cell lookup,
    BEFORE the annuity (O-14). Never merged into the card (§6.4's prohibition)."""
    pass  # sum applicable rate-overlay basis points from the resolved adjustment set

def statutory_ceiling_check(card_cell: float, rate_add_on: float,
                            statutory_rate_ceiling: float) -> bool:
    """O-13: re-checked AFTER the add-on, not only at card validation. A card
    valid at validation plus a 60bp overlay can be non-compliant."""
    pass  # (card_cell + rate_add_on) <= statutory_rate_ceiling, else hard fail

def nominal_annual_rate(card_cell: float, rate_add_on: float,
                        statutory_ceiling_check: bool) -> float:
    pass  # card_cell + rate_add_on, having asserted statutory_ceiling_check

Rate = module(card_cell, rate_add_on, statutory_ceiling_check, nominal_annual_rate,
              name="rate")
