"""Price one candidate loan: rate from the card, fees, credit life premium, instalment.

Plain Python, because the solve calls it a few dozen times per application and the final validation calls the
very same code again: an offer can only be re-checked "from scratch" if there is one pricing implementation.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, replace

from credit_core import fees
from credit_core.credit_life import JOINT, SINGLE, build_credit_life_table, credit_life_cap_applied, credit_life_premium
from credit_core.instalment import (
    effective_annual_rate, instalment as add_charges, instalment_before_fees, total_cost_of_credit,
)
from credit_core.rate_card import rate_card_rate
from credit_core.rounding import round_instalment, round_rate

INF = float("inf")


@dataclass(frozen=True)
class PricingResult:
    """What one candidate costs. `priced` is False when the card has no cell, or the rate breaks the statutory ceiling."""

    offered_amount: float
    term_months: int
    risk_grade: int
    priced: bool = False
    nominal_annual_rate: float | None = None
    rate_cell_id: str | None = None
    rate_card_version: str | None = None
    rate_addon_bps: float = 0.0
    within_statutory_ceiling: bool = False
    initiation_fee: float = 0.0
    initiation_fee_capped: bool = False
    monthly_service_fee: float = 0.0
    amount_financed: float = 0.0
    credit_life_premium: float = 0.0
    credit_life_cell_id: str | None = None
    credit_life_capped: bool = False
    instalment: float = 0.0
    total_cost_of_credit: float = 0.0
    effective_annual_rate: float = 0.0


class RateCardIndex:
    """The rate card as sorted bands, for the many lookups a search makes.

    `rows` are the card's rows (`amt_lo`, `term_lo`, `grade`, `rate`, `cell_id`, `rate_card_version`); a band
    starts at its `amt_lo` and runs to the next band's. Build it from the card `build` receives, so a refreshed card
    reaches the search the moment it reaches the table:

        index = RateCardIndex.from_card(rate_card_flex_loan)
        index.lookup(50_000.0, 60, 9)   # (rate in percent, cell id), or None
    """

    def __init__(self, rows: list[dict]):
        cells: dict[int, dict[float, list[dict]]] = {}
        for row in rows:
            cells.setdefault(int(row["grade"]), {}).setdefault(row["amt_lo"], []).append(row)
        self._bands = {}
        for grade, by_amount in cells.items():
            starts = sorted(by_amount)
            terms = [sorted(by_amount[s], key=lambda r: r["term_lo"]) for s in starts]
            self._bands[grade] = (starts, [[r["term_lo"] for r in t] for t in terms], terms)
        self.version = rows[0]["rate_card_version"] if rows else None

    @classmethod
    def from_card(cls, card) -> RateCardIndex:
        return cls(card.rows.df.to_dicts())

    def _row(self, band: int, term_months: float, grade: int) -> dict | None:
        _, term_starts, terms = self._bands[grade]
        j = bisect.bisect_right(term_starts[band], term_months) - 1
        return terms[band][j] if j >= 0 else None

    def lookup(self, amount: float, term_months: float, grade: int) -> tuple[float, str] | None:
        if grade not in self._bands:
            return None
        i = bisect.bisect_right(self._bands[grade][0], amount) - 1
        row = self._row(i, term_months, grade) if i >= 0 else None
        return (row["rate"], row["cell_id"]) if row else None

    def amount_bands(self, term_months: float, grade: int) -> list[tuple[float, float | None]]:
        """Each amount band's lower edge and its rate (percent, None where the card has no cell), lowest band first."""
        if grade not in self._bands:
            return []
        starts = self._bands[grade][0]
        return [(start, (self._row(i, term_months, grade) or {}).get("rate")) for i, start in enumerate(starts)]


class CreditLifeIndex:
    """The credit life table as rows, looked up by age, term and cover type."""

    def __init__(self):
        self._rows = build_credit_life_table().rows.df.to_dicts()

    def lookup(self, applicant_age_years: float, term_months: float, is_joint: bool) -> tuple[float, str | None]:
        cover = JOINT if is_joint else SINGLE
        for r in self._rows:
            if (r["cover_type"] == cover and r["age_lo"] <= applicant_age_years < r["age_hi"]
                    and r["term_lo"] <= term_months < r["term_hi"]):
                return r["rate"], r["cell_id"]
        return 0.0, None


class PriceEvaluator:
    """Prices one amount and term. Build it once per application and call `evaluate` as often as the search needs.

    `statutory_ceiling` is the highest nominal rate the law allows; a cell above it is not priced. The fee caps
    come from `credit_core.fees.build_fee_caps_table` for the decision date. A declared substitution of the
    client's own credit life policy removes the premium.

        evaluator = PriceEvaluator(index, CreditLifeIndex(), statutory_ceiling=0.28, credit_life_substitution_declared=False)
        evaluator.evaluate(50_000.0, 60, 9, 38.0, False).instalment
    """

    def __init__(self, rate_card_index: RateCardIndex, credit_life_index: CreditLifeIndex, statutory_ceiling: float,
                 credit_life_substitution_declared: bool, initiation_fee_cap: float = 6_500.0,
                 monthly_service_fee_cap: float = 69.0):
        self.rate_card_index = rate_card_index
        self.credit_life_index = credit_life_index
        self.statutory_ceiling = statutory_ceiling
        self.credit_life_substitution_declared = credit_life_substitution_declared
        self.initiation_fee_cap = initiation_fee_cap
        self.monthly_service_fee_cap = monthly_service_fee_cap

    def _rate(self, card_rate_percent: float, rate_addon_bps: float = 0.0) -> float:
        return round_rate(rate_card_rate(card_rate_percent) + rate_addon_bps / 10_000.0)

    def band_rates(self, term_months: int, risk_grade: int) -> list[tuple[float, float]]:
        """Each amount band's lower edge and the nominal rate it prices at; infinite where it cannot be priced."""
        return [(start, INF if pct is None or self._rate(pct) > self.statutory_ceiling else self._rate(pct))
                for start, pct in self.rate_card_index.amount_bands(term_months, risk_grade)]

    def evaluate(self, amount: float, term_months: int, risk_grade: int, applicant_age_years: float, is_joint: bool,
                 rate_addon_bps: float = 0.0) -> PricingResult:
        """The rate add-on is applied to the card's rate after the lookup and before the annuity."""
        hit = self.rate_card_index.lookup(amount, term_months, risk_grade)
        unpriced = PricingResult(amount, term_months, risk_grade, rate_card_version=self.rate_card_index.version,
                                 rate_addon_bps=rate_addon_bps)
        if hit is None:
            return unpriced
        card_rate, cell_id = hit
        rate = self._rate(card_rate, rate_addon_bps)
        if rate > self.statutory_ceiling:
            return replace(unpriced, nominal_annual_rate=rate, rate_cell_id=cell_id)

        fee = fees.initiation_fee(amount, initiation_fee_cap=self.initiation_fee_cap)
        financed = round(amount + fee, 2)
        premium, premium_cell, premium_capped = 0.0, None, False
        if not self.credit_life_substitution_declared:
            cover_rate, premium_cell = self.credit_life_index.lookup(applicant_age_years, term_months, is_joint)
            premium = credit_life_premium(financed, cover_rate)
            premium_capped = credit_life_cap_applied(financed, cover_rate)
        service_fee = fees.monthly_service_fee(monthly_service_fee_cap=self.monthly_service_fee_cap)
        monthly = round_instalment(add_charges(instalment_before_fees(financed, term_months, rate), service_fee, premium))
        return PricingResult(
            amount, term_months, risk_grade, True, rate, cell_id, self.rate_card_index.version, rate_addon_bps, True,
            round(fee, 2), fees.initiation_fee_capped(fee, self.initiation_fee_cap), round(service_fee, 2), financed,
            premium, premium_cell, premium_capped, monthly, total_cost_of_credit(monthly, term_months, fee),
            round(effective_annual_rate(rate), 4),
        )
