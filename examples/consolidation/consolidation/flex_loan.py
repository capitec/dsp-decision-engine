"""Product 11, Flex Loan Consolidation: an unsecured loan whose proceeds go straight to the creditors.

Priced with `loan_granting`'s `PriceEvaluator` over this product's own rate card, a different card through the same
lookup and fee mechanism.
"""
from __future__ import annotations

from decikit import ParamTable, allow_fallback, param, param_table, step
from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, RateCardIndex

from consolidation.products import Quote
from consolidation.vocab import PRODUCT_FLEX_LOAN_CONSOLIDATION

CODE = PRODUCT_FLEX_LOAN_CONSOLIDATION
REJECT_AMOUNT, REJECT_UNPRICED = "amount_outside_product_limits", "no_rate_cell"

# A client's debt moved off a competitor is cheaper to refinance than the Bank's own book.
MARGIN_BANDS = param_table({"external_from": float, "base_bps": float}, default=[
    {"external_from": 0.0, "base_bps": 45.0}, {"external_from": 0.6, "base_bps": 30.0},
    {"external_from": 0.8, "base_bps": 15.0}, {"external_from": 0.95, "base_bps": 0.0}])
TERMS = param_table({"term_months": int}, default=[{"term_months": m} for m in (12, 24, 36, 48, 60, 72, 84)])


@allow_fallback
@step(output="flex_loan_policy")
def flex_loan_policy(
    min_accounts: int = param(2, ge=1), max_accounts: int = param(8, ge=3, le=15),
    min_amount: float = param(10_000.0, ge=0.0), max_amount: float = param(500_000.0, ge=0.0),
    min_term: int = param(12, ge=1), max_term: int = param(84, ge=1),
    min_relief: float = param(0.10, ge=0.0, le=0.30), anti_harm: float = param(0.15, ge=0.0, le=0.60),
    max_term_extension: int = param(24, ge=0, le=60), rate_ceiling: bool = param(True),
    new_money_pct: float = param(0.25, ge=0.0, le=0.50), new_money_cap: float = param(50_000.0, ge=0.0, le=150_000.0),
    dsr_ceiling: float = param(0.45, ge=0.30, le=0.60), min_external: float = param(0.60, ge=0.0, le=1.0),
    min_discretionary: float = param(850.0, ge=500.0, le=2_500.0),
    margin_per_grade_bps: float = param(2.0, ge=0.0), margin_bands: ParamTable = MARGIN_BANDS,
    candidate_terms: ParamTable = TERMS,
) -> dict:
    """The limits and intervention thresholds product 11's team owns, as one record for the search."""
    return {
        "product_code": CODE, "min_accounts": min_accounts, "max_accounts": max_accounts, "min_amount": min_amount,
        "max_amount": max_amount, "min_term": min_term, "max_term": max_term, "min_relief": min_relief,
        "anti_harm": anti_harm, "max_term_extension": max_term_extension, "rate_ceiling": rate_ceiling,
        "new_money_pct": new_money_pct, "new_money_cap": new_money_cap, "dsr_ceiling": dsr_ceiling,
        "min_external": min_external, "min_discretionary": min_discretionary, "margin_per_grade_bps": margin_per_grade_bps,
        "margin_bands": [{"external_from": f, "base_bps": b}
                         for f, b in zip(margin_bands.external_from.tolist(), margin_bands.base_bps.tolist())],
        "candidate_terms": candidate_terms.term_months.tolist(),
    }


class FlexLoanConsolidation:
    code = CODE

    def __init__(self, rate_card_index: RateCardIndex, credit_life_index: CreditLifeIndex,
                 evaluator: PriceEvaluator | None = None):
        self.rate_card_index, self.credit_life_index, self.evaluator = rate_card_index, credit_life_index, evaluator

    def bind(self, pricing_context: dict) -> FlexLoanConsolidation:
        """This product priced under the statutory ceiling and fee caps in force for one assessment."""
        return FlexLoanConsolidation(self.rate_card_index, self.credit_life_index, PriceEvaluator(
            self.rate_card_index, self.credit_life_index, pricing_context["statutory_ceiling"], False,
            pricing_context["initiation_fee_cap"], pricing_context["monthly_service_fee_cap"]))

    def unavailable(self, client: dict, policy: dict) -> str | None:
        return None

    def carries(self, accounts: list[dict], policy: dict) -> bool:
        return policy["min_accounts"] <= len(accounts)

    def variants(self, policy: dict, longest_remaining_term: float) -> list[tuple[int, int]]:
        """`(term_months, promotional_months)`: the ladder up to the longest term the intervention allows, plus that term."""
        cap = max(policy["min_term"], min(policy["max_term"], longest_remaining_term + policy["max_term_extension"]))
        return [(t, 0) for t in sorted({t for t in policy["candidate_terms"] if policy["min_term"] <= t <= cap} | {int(cap)})]

    def quote(self, policy: dict, proceeds: float, term_months: int, promotional_months: int, risk_grade: int,
              applicant_age_years: float, external_proportion: float, **_) -> Quote | str:
        """The cost of lending `proceeds` (the settlements, their buffer and any new money) over `term_months`.

        The initiation fee is capitalised on top of the proceeds, so there is nothing circular to solve: one pricing.
        Returns the reason it cannot be priced as a string.
        """
        if not policy["min_amount"] <= proceeds <= policy["max_amount"]:
            return REJECT_AMOUNT
        margin = next(b["base_bps"] for b in reversed(policy["margin_bands"]) if external_proportion >= b["external_from"]) \
            + policy["margin_per_grade_bps"] * risk_grade
        price = self.evaluator.evaluate(proceeds, term_months, risk_grade, applicant_age_years, False, rate_addon_bps=margin)
        if not price.priced:
            return REJECT_UNPRICED
        return Quote(proceeds, price.instalment, price.instalment, price.nominal_annual_rate, price.total_cost_of_credit,
                     price.effective_annual_rate, price.rate_cell_id, price.rate_card_version,
                     {"initiation_fee": price.initiation_fee, "monthly_service_fee": price.monthly_service_fee,
                      "credit_life_premium": price.credit_life_premium, "margin_bps": margin})
