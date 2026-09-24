"""Stage 5.7 -- pricing: the arithmetic the solve calls repeatedly (spec 03 §5.7).

Plain Python, not decider steps: the search (`solve.py`) and final
validation (`validation.py`) both call `PriceEvaluator.evaluate` directly,
many times per application, and both must call the *same* function --
"the offer must never fail affordability when re-checked from scratch"
(§5.8 requirement 7) only means something if there is exactly one pricing
implementation, not two that happen to agree today.

Reuses `credit_core.fees`, `credit_core.credit_life`, `credit_core.instalment`
and `credit_core.rounding` (project 00) as plain functions, called with
their own declared defaults -- see NOTES.md "Framework friction" for why a
`frame_step` (this project's solve and validation are both frame_steps)
cannot pick up a params-document override of those defaults the way a
scalar `step()` can, and why that is a real, if narrow, reproducibility
gap between this stage and the rest of the pipeline.

**Long-term loading (spec 03 §6.1: a separate 2x12 table for terms 61-84).**
Project 00's `generate_flex_loan_card` builds all 55 term columns spanning
6-84 directly in one table (`credit_core/rate_card.py`), not 55 columns for
6-60 plus a separate loading table -- a deviation from spec that 00 made
and this project cannot edit (BRIEF: never edit another project's
directory). Every term 6-84 is therefore priced from one direct cell read,
with no second table. Recorded in NOTES.md "Gaps in what I consumed".
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from credit_core import fees as fees_mod
from credit_core.credit_life import JOINT, SINGLE, build_credit_life_table, credit_life_cap_applied, credit_life_premium
from credit_core.instalment import (
    effective_annual_rate as effective_annual_rate_fn,
    instalment as instalment_fn,
    instalment_before_fees,
    total_cost_of_credit as total_cost_of_credit_fn,
)
from credit_core.rate_card import rate_card_rate
from credit_core.rounding import round_instalment, round_rate


@dataclass(frozen=True)
class PricingResult:
    offered_amount: float
    term_months: int
    risk_grade: int
    nominal_annual_rate: float | None
    rate_cell_id: str | None
    rate_card_version: str | None
    rate_addon_bps: float
    statutory_ceiling: float
    within_statutory_ceiling: bool
    initiation_fee: float
    initiation_fee_capped: bool
    monthly_service_fee: float
    amount_financed: float
    credit_life_premium: float
    credit_life_cell_id: str | None
    credit_life_capped: bool
    instalment: float
    total_cost_of_credit: float
    effective_annual_rate: float
    priced: bool  # False when no rate cell exists (out of range or above the statutory ceiling)


class RateCardIndex:
    """An in-memory index over the same rows `decider build` loads as the
    `rate_card_flex_loan` `DecisionTableConfig` -- built from that step's own
    `.rows.df` (never regenerated independently), so a refreshed card (00 §9:
    no redeploy) is picked up by the search the moment it is picked up by the table."""

    def __init__(self, rows: list[dict]):
        by_grade: dict[int, list[dict]] = {}
        for r in rows:
            by_grade.setdefault(int(r["grade"]), []).append(r)
        self._bands: dict[int, list[tuple[float, list[dict]]]] = {}
        for grade, grade_rows in by_grade.items():
            by_amount: dict[tuple[float, float], list[dict]] = {}
            for r in grade_rows:
                by_amount.setdefault((r["amt_lo"], r["amt_hi"]), []).append(r)
            amount_bands = sorted(by_amount.items(), key=lambda kv: kv[0][0])
            for _, term_rows in amount_bands:
                term_rows.sort(key=lambda r: r["term_lo"])
            self._bands[grade] = [(lo, term_rows) for (lo, _hi), term_rows in amount_bands]
        self.version = rows[0]["rate_card_version"] if rows else None

    def lookup(self, amount: float, term_months: float, grade: int) -> tuple[float, str] | None:
        bands = self._bands.get(grade)
        if not bands:
            return None
        los = [lo for lo, _ in bands]
        i = bisect.bisect_right(los, amount) - 1
        if i < 0:
            return None
        term_rows = bands[i][1]
        term_los = [r["term_lo"] for r in term_rows]
        j = bisect.bisect_right(term_los, term_months) - 1
        if j < 0:
            return None
        row = term_rows[j]
        return row["rate"], row["cell_id"]

    def band_bounds(self, amount: float, grade: int) -> tuple[float, float] | None:
        """The (amt_lo, amt_hi) of the amount band `amount` falls in, for the solve's
        band-by-band traversal (`solve.py`) -- the same band structure `lookup` uses,
        just returning the edges instead of the priced cell."""
        bands = self._bands.get(grade)
        if not bands:
            return None
        los = [lo for lo, _ in bands]
        i = bisect.bisect_right(los, amount) - 1
        if i < 0:
            return None
        lo = bands[i][0]
        hi = bands[i + 1][0] if i + 1 < len(bands) else float("inf")
        return lo, hi

    @classmethod
    def from_configurable_step(cls, rate_card_flex_loan) -> "RateCardIndex":
        return cls(rate_card_flex_loan.rows.df.to_dicts())


class CreditLifeIndex:
    """Same idea, over project 00's (small, working-depth) credit life table."""

    def __init__(self):
        self._rows = build_credit_life_table().rows.df.to_dicts()

    def lookup(self, applicant_age_years: float, term_months: float, is_joint: bool) -> tuple[float, str]:
        cover_type = JOINT if is_joint else SINGLE
        for r in self._rows:
            if r["cover_type"] != cover_type:
                continue
            if r["age_lo"] <= applicant_age_years < r["age_hi"] and r["term_lo"] <= term_months < r["term_hi"]:
                return r["rate"], r["cell_id"]
        return 0.0, None


class PriceEvaluator:
    """One object, built once per application, holding the two table indexes so
    repeated `evaluate()` calls inside the solve don't re-scan or re-build them."""

    def __init__(self, rate_card_index: RateCardIndex, credit_life_index: CreditLifeIndex,
                 statutory_ceiling: float, credit_life_substitution_declared: bool):
        self.rate_card_index = rate_card_index
        self.credit_life_index = credit_life_index
        self.statutory_ceiling = statutory_ceiling
        self.credit_life_substitution_declared = credit_life_substitution_declared

    def evaluate(self, amount: float, term_months: int, risk_grade: int, applicant_age_years: float,
                 is_joint: bool, rate_addon_bps: float = 0.0) -> PricingResult:
        hit = self.rate_card_index.lookup(amount, term_months, risk_grade)
        if hit is None:
            return PricingResult(
                offered_amount=amount, term_months=term_months, risk_grade=risk_grade,
                nominal_annual_rate=None, rate_cell_id=None, rate_card_version=self.rate_card_index.version,
                rate_addon_bps=rate_addon_bps, statutory_ceiling=self.statutory_ceiling,
                within_statutory_ceiling=False, initiation_fee=0.0, initiation_fee_capped=False,
                monthly_service_fee=0.0, amount_financed=0.0, credit_life_premium=0.0, credit_life_cell_id=None,
                credit_life_capped=False, instalment=0.0, total_cost_of_credit=0.0, effective_annual_rate=0.0,
                priced=False,
            )
        cell_rate_pct, cell_id = hit
        rate = rate_card_rate(cell_rate_pct) + rate_addon_bps / 10_000.0
        rate = round_rate(rate)
        within_ceiling = rate <= self.statutory_ceiling
        if not within_ceiling:
            return PricingResult(
                offered_amount=amount, term_months=term_months, risk_grade=risk_grade,
                nominal_annual_rate=rate, rate_cell_id=cell_id, rate_card_version=self.rate_card_index.version,
                rate_addon_bps=rate_addon_bps, statutory_ceiling=self.statutory_ceiling,
                within_statutory_ceiling=False, initiation_fee=0.0, initiation_fee_capped=False,
                monthly_service_fee=0.0, amount_financed=0.0, credit_life_premium=0.0, credit_life_cell_id=None,
                credit_life_capped=False, instalment=0.0, total_cost_of_credit=0.0, effective_annual_rate=0.0,
                priced=False,
            )

        fee = fees_mod.initiation_fee(amount)
        fee_capped = fees_mod.initiation_fee_capped(fee)
        service_fee = fees_mod.monthly_service_fee(amount)
        amount_financed = round(amount + fee, 2)

        if self.credit_life_substitution_declared:
            cl_premium, cl_cell_id, cl_capped = 0.0, None, False
        else:
            cl_rate, cl_cell_id = self.credit_life_index.lookup(applicant_age_years, term_months, is_joint)
            cl_premium = credit_life_premium(amount_financed, cl_rate)
            cl_capped = credit_life_cap_applied(amount_financed, cl_rate)

        before_fees = instalment_before_fees(amount_financed, term_months, rate)
        total_instalment = round_instalment(instalment_fn(before_fees, service_fee, cl_premium))
        total_cost = total_cost_of_credit_fn(total_instalment, term_months, fee)
        ear = effective_annual_rate_fn(rate)

        return PricingResult(
            offered_amount=amount, term_months=term_months, risk_grade=risk_grade,
            nominal_annual_rate=rate, rate_cell_id=cell_id, rate_card_version=self.rate_card_index.version,
            rate_addon_bps=rate_addon_bps, statutory_ceiling=self.statutory_ceiling, within_statutory_ceiling=True,
            initiation_fee=round(fee, 2), initiation_fee_capped=fee_capped, monthly_service_fee=round(service_fee, 2),
            amount_financed=amount_financed, credit_life_premium=cl_premium, credit_life_cell_id=cl_cell_id,
            credit_life_capped=cl_capped, instalment=total_instalment, total_cost_of_credit=round(total_cost, 2),
            effective_annual_rate=round(ear, 4), priced=True,
        )
