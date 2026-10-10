"""The Flex Loan rate card: 72 amount bands by 40 term bands by 12 grades (34 560 cells), finer at small amounts.

The card is a `param_table` in the params document, so Treasury can replace it without a code change or a rebuild.
Cells are ordered by amount band, then term band, then grade, and `card_rate` indexes them by position.
"""
from __future__ import annotations

from decikit import ParamTable, allow_fallback, param, param_table

from credit_core.evidence import cell_id

GRADES = 12
FIRST_TERM, LAST_TERM = 6, 45  # one column per month, then one for every longer term
TERM_BANDS = LAST_TERM - FIRST_TERM + 1

# (first amount, last amount, band width): fine bands at small amounts, wide ones at large amounts.
_AMOUNT_LADDER = [(2_000.0, 10_000.0, 500.0), (10_000.0, 60_000.0, 2_500.0), (60_000.0, 200_000.0, 10_000.0),
                  (200_000.0, 380_000.0, 15_000.0), (380_000.0, 500_000.0, 12_000.0)]


def _band_floors() -> list[float]:
    floors = [lo + i * width for lo, hi, width in _AMOUNT_LADDER for i in range(round((hi - lo) / width))]
    return [0.0, *floors[1:]]  # the first band also takes everything below the product minimum


def _rate(amount_band: int, term: int, grade: int) -> float:
    """A synthetic surface that rises with grade and term, falls with amount, and drops about 155 basis points at the
    R60 000 edge: a larger advance can cost less per month than a smaller one just below it."""
    edge = -1.55 if amount_band >= 36 else 0.0
    return round(min(max(12.45 + (grade - 1) * 1.30 - 0.02 * amount_band + 0.015 * (term - 6) + edge, 12.45), 26.90), 4)


def card_tables() -> dict[str, list[dict]]:
    """The two tables `card_rate` reads, as rows for the params document: band floors, and the rate of every cell in
    (amount band, term band, grade) order, as a percentage."""
    floors = _band_floors()
    return {
        "amount_bands": [{"floor": floor} for floor in floors],
        "cells": [{"rate": _rate(band, FIRST_TERM + term_band, grade)}
                  for band in range(len(floors)) for term_band in range(TERM_BANDS) for grade in range(1, GRADES + 1)],
    }


# workaround: this is not a `DecisionTableConfig` because a table of this size still scans its rows linearly (about 200
# microseconds per lookup) and the solve prices hundreds of candidate amounts per request, against half a microsecond here.
def card_rate(
    offered_amount: float, risk_grade: int, offer_term_months: float,
    amount_bands: ParamTable = param_table({"floor": float}, required=True),
    cells: ParamTable = param_table({"rate": float}, required=True),
) -> tuple[float, int, int]:
    """(card_rate, amt_band, term_band): the cell's rate as a fraction, and the two band numbers that name the cell."""
    band = 0
    for i in range(len(amount_bands.floor)):
        if amount_bands.floor[i] <= offered_amount:
            band = i
    term_band = int(min(max(offer_term_months, FIRST_TERM), LAST_TERM)) - FIRST_TERM
    grade = min(max(risk_grade, 1), GRADES)
    return cells.rate[(band * TERM_BANDS + term_band) * GRADES + grade - 1] / 100.0, band, term_band


@allow_fallback
def rate_cell_id(has_offer: bool, amt_band: int, term_band: int, risk_grade: int,
                 card_version: str = param("rc-2026.09")) -> str:
    """The id of the cell that priced the offer ("" when there is no offer)."""
    return cell_id("rate_card.flex_loan", card_version, amt_band, term_band, risk_grade) if has_offer else ""
