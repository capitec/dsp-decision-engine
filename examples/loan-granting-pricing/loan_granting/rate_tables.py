"""The rate card and the credit life table as `param_table`s, and the lookups over them.

`RateCardTable` and `CreditLifeTable` answer the same questions as `pricing.RateCardIndex` and `CreditLifeIndex`
(`PriceEvaluator` takes either), but read the arrays of a table param, so the data is in the params document and the
diff, not in an object a step closes over.
"""
from __future__ import annotations

import bisect

import numpy as np
from decider import DecisionTableConfig

from credit_core.credit_life import build_credit_life_table
from credit_core.evidence import cell_id

RATE_CARD_COLUMNS = {"grade": int, "amt_lo": float, "term_lo": float, "rate": float, "amt_band": int, "term_band": int}
CREDIT_LIFE_COLUMNS = {"cell": int, "cover_type": int, "age_lo": float, "age_hi": float, "term_lo": float,
                       "term_hi": float, "rate": float}

RATE_CARD_TABLE = "rate_card.flex_loan"
OPEN = 1e9  # an open band edge: a params document cannot hold infinity


def rate_card_rows(card: DecisionTableConfig) -> tuple[list[dict], str]:
    """The card's rows sorted by grade, amount band and term band, and the card's version.

    The cell ids are not stored: they are `rate_card.flex_loan@<version>#<amt_band>|<term_band>|<grade>`, and a card
    whose ids are anything else is refused.
    """
    frame = card.rows.df.sort("grade", "amt_lo", "term_lo")
    versions = frame["rate_card_version"].unique().to_list()
    if len(versions) != 1:
        raise ValueError(f"the rate card holds {len(versions)} versions; a card has exactly one")
    version = versions[0]
    expected = [cell_id(RATE_CARD_TABLE, version, a, t, g) for a, t, g in zip(*frame.select("amt_band", "term_band", "grade"))]
    if frame["cell_id"].to_list() != expected:
        raise ValueError(f"rate card cell ids must be {RATE_CARD_TABLE}@<version>#<amt_band>|<term_band>|<grade>")
    rows = frame.select(*RATE_CARD_COLUMNS).with_columns(
        [frame[c].clip(-OPEN, OPEN) for c in ("amt_lo", "term_lo")]).to_dicts()
    return rows, version


def credit_life_rows() -> list[dict]:
    """The shipped credit life rates, one row per cell; `cell` numbers the cells for their ids."""
    rows = []
    for i, row in enumerate(build_credit_life_table().rows.df.to_dicts()):
        rows.append({"cell": i, **{c: row[c] for c in CREDIT_LIFE_COLUMNS if c != "cell"}})
        for edge in ("age_lo", "age_hi", "term_lo", "term_hi"):
            rows[-1][edge] = max(-OPEN, min(OPEN, row[edge]))
    return rows


class RateCardTable:
    """The rate card as arrays sorted by grade, amount band and term band: `lookup` and `amount_bands` as `RateCardIndex`.

    It bisects the table's own arrays through memoryviews, which hand back Python numbers: a search asks hundreds of
    times, and numpy calls or copying a grade's rows cost more than the questions.

        index = RateCardTable(card, "rc-2026.09")     # `card` is a `param_table(RATE_CARD_COLUMNS)` argument
        index.lookup(50_000.0, 60, 9)                  # (rate in percent, cell id), or None
    """

    def __init__(self, card, version: str):
        self._grade_col, self._amount, self._term, self._rate, self._amount_band, self._term_band = (
            memoryview(column) for column in (card.grade, card.amt_lo, card.term_lo, card.rate, card.amt_band,
                                              card.term_band))
        self.version = version
        self._grades: dict[int, tuple[int, int, list[int]]] = {}

    def _rows(self, grade: int) -> tuple[int, int, list[int]]:
        """The grade's rows `[first, end)` and the first row of each of its amount bands."""
        if grade not in self._grades:
            first = bisect.bisect_left(self._grade_col, grade)
            end = bisect.bisect_right(self._grade_col, grade)
            starts = np.asarray(self._amount[first:end])
            opens = np.flatnonzero(np.r_[True, starts[1:] != starts[:-1]]) + first if end > first else []
            self._grades[grade] = (first, end, list(map(int, opens)))
        return self._grades[grade]

    def lookup(self, amount: float, term_months: float, grade: int) -> tuple[float, str] | None:
        first, end, opens = self._rows(grade)
        last = bisect.bisect_right(self._amount, amount, first, end) - 1
        if last < first:
            return None
        band = opens[bisect.bisect_right(opens, last) - 1]
        row = bisect.bisect_right(self._term, term_months, band, last + 1) - 1
        if row < band:
            return None
        return self._rate[row], cell_id(RATE_CARD_TABLE, self.version, self._amount_band[row], self._term_band[row], grade)

    def amount_bands(self, term_months: float, grade: int) -> list[tuple[float, float | None]]:
        """Each amount band's lower edge and its rate (percent, None where the card has no cell), lowest band first."""
        _, end, opens = self._rows(grade)
        bands = []
        for band, band_end in zip(opens, [*opens[1:], end]):
            row = bisect.bisect_right(self._term, term_months, band, band_end) - 1
            bands.append((self._amount[band], self._rate[row] if row >= band else None))
        return bands


class CreditLifeTable:
    """The credit life rates as arrays: `lookup` as `CreditLifeIndex`, with the cell id from the `cell` column.

        CreditLifeTable(table, "cl-2026.09").lookup(38.0, 60, False)    # (rate per R1 000, cell id)
    """

    def __init__(self, table, version: str):
        self._rows = list(zip(table.cover_type.tolist(), table.age_lo.tolist(), table.age_hi.tolist(),
                              table.term_lo.tolist(), table.term_hi.tolist(), table.rate.tolist(), table.cell.tolist()))
        self.version = version

    def lookup(self, applicant_age_years: float, term_months: float, is_joint: bool) -> tuple[float, str | None]:
        cover = 2 if is_joint else 1
        for cover_type, age_lo, age_hi, term_lo, term_hi, rate, cell in self._rows:
            if cover_type == cover and age_lo <= applicant_age_years < age_hi and term_lo <= term_months < term_hi:
                return rate, cell_id("credit_life", self.version, cell)
        return 0.0, None
