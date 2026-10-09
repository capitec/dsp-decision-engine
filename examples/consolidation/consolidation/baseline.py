"""The do-nothing position, and whether the client needs a consolidation at all.

Every later scenario is measured against the baseline. If the request is affordable as made and nothing suggests the
client is better off consolidating, the flow completes as plain granting: the largest affordable Flex Loan, solved with
`loan_granting`'s own solve. The six conditions are each recorded, so a client told "we can only lend you R14 000" can
later be shown why consolidation was not pursued.
"""
from __future__ import annotations

from datetime import date

from decider import DecisionTableConfig, ParamTable, allow_fallback, missing_as, param, param_table, step
from loan_granting.pricing import PriceEvaluator
from loan_granting.rate_tables import RATE_CARD_COLUMNS, RateCardTable, rate_card_rows
from loan_granting.solve import PRODUCT_MAX_AMOUNT, solve_term

from consolidation import vocab
from consolidation.capacity import Capacity, case_record, debt_service_ratio
from consolidation.inventory import num
from consolidation.rate_tables import (
    CREDIT_LIFE,
    CREDIT_LIFE_VERSION_PARAM,
    PRODUCT11_CARD,
    CardTable,
    CreditLifeTable,
)
from consolidation.scenario import remaining_cost, weighted_rate

CONDITIONS = ("request_affordable_as_made", "debt_service_ratio_within_ceiling", "no_account_in_arrears",
              "rate_gap_within_threshold", "fewer_than_six_active_accounts", "no_settleable_account_at_high_rate")


@allow_fallback
@step(output="invariants")
def invariants(
    gross_monthly_income: float, income_source_code: int, income_haircut_applied: float, statutory_deductions: float,
    net_monthly_income: float, living_expenses: float, expense_basis_code: int, dependants_count: float,
    income_verification_tier: int, applicant_income_evidence_gap: bool, bureau_is_stale: bool, minimum_income_tier: int,
    residual_floor_amount: float, affordability_buffer_applied: float,
) -> dict:
    """The income and expense determination, made once. Every scenario reads this record and nothing recomputes it."""
    return {
        "gross_monthly_income": gross_monthly_income, "income_source_code": income_source_code,
        "income_haircut_applied": income_haircut_applied, "statutory_deductions": statutory_deductions,
        "net_monthly_income": net_monthly_income, "living_expenses": living_expenses, "expense_basis_code": expense_basis_code,
        "dependants_count": dependants_count, "income_verification_tier": income_verification_tier,
        "applicant_income_evidence_gap": applicant_income_evidence_gap, "bureau_is_stale": bureau_is_stale,
        "minimum_income_tier": minimum_income_tier, "residual_floor_amount": residual_floor_amount,
        "affordability_buffer_applied": affordability_buffer_applied}


def baseline_step(capacity_documents: tuple[str, str]):
    """The `baseline_measures` step, scoring the client's accounts with the capacity unit built from `capacity_documents`."""

    @allow_fallback
    @step(output="baseline")
    def baseline_measures(
        classified_accounts: list[dict], invariants: dict, decision_date: date, channel_code: int, segment_code: float,
        court_ordered_deductions: float = missing_as(0.0), paydown_horizon_months: int = param(36, ge=1, le=84),
    ) -> dict:
        """Obligations, capacity, average rate, remaining cost, accounts and providers as the client stands today.

        `existing_obligations` is what the assessment costs the accounts at (imputed for cards); `total_instalment` is
        what the client actually pays each month, which is what relief is measured against. A revolving account has no
        natural term, so its remaining cost assumes the balance is amortised over `paydown_horizon_months`.
        """
        case = case_record(decision_date, vocab.PRODUCT_FLEX_LOAN_CONSOLIDATION, channel_code, segment_code,
                           court_ordered_deductions, invariants, invariants["affordability_buffer_applied"])
        now = Capacity(*capacity_documents)(case, classified_accounts, {})
        live = [a for a in classified_accounts if a.get("account_status_code") != vocab.CLOSED_STATUS]
        return {
            "existing_obligations": now["existing_obligations"], "discretionary_income": now["discretionary_income"],
            "max_affordable_instalment": now["max_affordable_instalment"],
            "evidence_sufficiency_code": now["evidence_sufficiency_code"],
            "worst_arrears_months": now["worst_arrears_months"],
            "total_instalment": round(sum(num(a, "instalment") for a in live), 2),
            "debt_service_ratio": debt_service_ratio(now["existing_obligations"], invariants["net_monthly_income"]),
            "weighted_average_rate": round(weighted_rate(live), 4),
            "total_remaining_cost": round(sum(remaining_cost(a, paydown_horizon_months) for a in live), 2),
            "account_count": len(live), "provider_count": len({a.get("provider_code") for a in live}),
            "paydown_horizon_months": paydown_horizon_months,
        }

    return baseline_measures


def plain_granting_step(plain_card: DecisionTableConfig, product11_params: tuple):
    """The `plain_granting` step, pricing from `plain_card` (the Flex Loan's, a `DecisionTableConfig`) and product 11's card.

    The cards are the defaults of table params: `product11_params` is `rate_tables.product11_params(card)`, and the
    Flex Loan's rows are `plain_rate_card`, with its version in `plain_rate_card_version`.
    """
    plain_rows, plain_version = rate_card_rows(plain_card)
    plain_param = param_table(RATE_CARD_COLUMNS, default=plain_rows)
    plain_version_param = param(plain_version)
    consolidation_param, consolidation_version_param = product11_params

    def evaluator(index, context, credit_life):
        return PriceEvaluator(index, credit_life, context["statutory_ceiling"], False, context["initiation_fee_cap"],
                              context["monthly_service_fee_cap"])

    @allow_fallback
    @step(outputs=("plain_offer", "short_circuit_verdicts", "short_circuit_applies"))
    def plain_granting(
        baseline: dict, invariants: dict, classified_accounts: list[dict], settlements: list[dict], pricing_context: dict,
        risk_grade: float, applicant_age_years: float, requested_amount: float = missing_as(0.0),
        reference_term_months: int = param(36, ge=6, le=84), debt_service_ceiling: float = param(0.40, gt=0.0, le=1.0),
        rate_gap_threshold: float = param(0.03, ge=0.0), active_account_limit: int = param(6, ge=1),
        high_rate_threshold: float = param(0.28, gt=0.0), plain_rate_card: ParamTable = plain_param,
        plain_rate_card_version: str = plain_version_param, consolidation_rate_card: ParamTable = consolidation_param,
        consolidation_rate_card_version: str = consolidation_version_param, credit_life: ParamTable = CREDIT_LIFE,
        credit_life_version: str = CREDIT_LIFE_VERSION_PARAM,
    ) -> tuple[dict, dict, bool]:
        """The plain Flex Loan the client could have, and the six conditions that say whether it is enough.

        The offer is the largest affordable amount (up to the amount asked for) at `reference_term_months`.
        """
        grade, age = int(risk_grade), applicant_age_years
        life = CreditLifeTable(credit_life, credit_life_version)
        solved = solve_term(evaluator(RateCardTable(plain_rate_card, plain_rate_card_version), pricing_context, life),
                            reference_term_months, grade, age, False,
                            PRODUCT_MAX_AMOUNT, requested_amount or None, baseline["max_affordable_instalment"])
        offer = {} if solved.amount is None else {
            "offered_amount": solved.amount, "term_months": reference_term_months,
            "binding_constraint_code": solved.binding_constraint_code, "instalment": solved.pricing.instalment,
            "nominal_annual_rate": solved.pricing.nominal_annual_rate,
            "total_cost_of_credit": solved.pricing.total_cost_of_credit, "rate_cell_id": solved.pricing.rate_cell_id}
        settleable = {s["account_ref"] for s in settlements}
        everything = min(max(sum(s["settlement_amount"] for s in settlements), 10_000.0), 500_000.0)
        achievable = evaluator(CardTable(consolidation_rate_card, consolidation_rate_card_version, PRODUCT11_CARD),
                               pricing_context, life).evaluate(everything, reference_term_months, grade, age, False)
        after_plain = baseline["existing_obligations"] + offer.get("instalment", 0.0)
        verdicts = dict(zip(CONDITIONS, (
            bool(offer) and offer["offered_amount"] == requested_amount,
            debt_service_ratio(after_plain, invariants["net_monthly_income"]) <= debt_service_ceiling,
            baseline["worst_arrears_months"] == 0,
            achievable.priced and baseline["weighted_average_rate"] - achievable.nominal_annual_rate <= rate_gap_threshold,
            baseline["account_count"] < active_account_limit,
            not any(num(a, "nominal_annual_rate") >= high_rate_threshold
                    for a in classified_accounts if a["account_ref"] in settleable))))
        return offer, verdicts, bool(requested_amount) and all(verdicts.values())

    return plain_granting
