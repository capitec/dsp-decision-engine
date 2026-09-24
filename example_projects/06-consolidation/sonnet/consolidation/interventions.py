"""Stage 5.7 -- policy interventions inside the search (spec 06 §5.7).

Fourteen interventions declared in the spec; this slice implements the
subset that applies to products 11 and 20 (CON-INT-02, 07, 11, 12 are
enforced earlier, at settleability/eligibility -- §5.7 requirement 6,
"not-applicable is distinct from passed", applies to them here: they are
recorded as not-applicable rather than silently omitted).

A violated intervention invalidates the scenario outright (requirement 1);
every intervention is evaluated, not only until the first failure
(requirement 3); one of them (CON-INT-04, the anti-harm ceiling) is wired
through `credit_core.adjustments` as this project's declared overlay target
(§6.3's worked example: "Threshold adjustment | CON-INT-04 anti-harm ceiling
| 15% -> 12% on product 11 only").
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

from consolidation import vocab

CON_INT_01 = 1  # max accounts settled
CON_INT_02 = 2  # recently opened -- enforced at settleability (inventory.py)
CON_INT_03 = 3  # minimum instalment reduction
CON_INT_04 = 4  # anti-harm: total cost of credit increase ceiling
CON_INT_05 = 5  # maximum term extension
CON_INT_06 = 6  # rate ceiling vs. settled accounts' weighted average
CON_INT_07 = 7  # no consolidation under debt review -- enforced at eligibility
CON_INT_08 = 8  # new money cap
CON_INT_09 = 9  # post-consolidation debt service ratio ceiling
CON_INT_10 = 10  # minimum external-creditor proportion (product 11 only)
CON_INT_11 = 11  # no settlement of a disputed account -- enforced at settleability
CON_INT_12 = 12  # secured accounts require security release/transfer -- enforced at settleability
CON_INT_13 = 13  # max consolidations per rolling 24 months
CON_INT_14 = 14  # minimum post-consolidation discretionary income

NOT_APPLICABLE = "not_applicable"
PASSED = "passed"
FAILED = "failed"

ADJUSTMENT_SET_ID = "AS-06-2026.09"

INTERVENTION_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-06-2026-001", kind="cap_adjustment", target="anti_harm_ceiling_pct",
        effect=AdjustmentEffect("multiply", 0.80), scope={"product_code": vocab.PRODUCT_FLEX_CONSOLIDATION},
        stack_position=1, owner="Credit Risk Policy", approval_reference="CRC-2026-044",
        rationale="Anti-harm ceiling tightened on product 11 pending the Q3 ombud review",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True,
    ),
])

# Default thresholds (§5.7 "Default" column). Tunable ranges are documented but not
# enforced as a separate validation layer at this slice's depth (params.json carries
# the values actually in force).
DEFAULT_THRESHOLDS = {
    "CON-INT-01_max_accounts": 8,
    "CON-INT-03_min_relief_pct": 0.10,
    "CON-INT-04_anti_harm_pct": 0.15,
    "CON-INT-05_max_term_extension_months": 24,
    "CON-INT-06_rate_ceiling_binding": True,
    "CON-INT-08_new_money_pct": 0.25,
    "CON-INT-08_new_money_cap": 50_000.0,
    "CON-INT-09_dsr_ceiling": 0.45,
    "CON-INT-10_min_external_pct": 0.60,
    "CON-INT-13_max_per_24_months": 2,
    "CON-INT-14_min_discretionary_income": 850.0,
}


@dataclass(frozen=True)
class InterventionResult:
    code: int
    name: str
    status: str  # PASSED / FAILED / NOT_APPLICABLE
    actual: float | None
    threshold: float | None
    overlay_id: str | None = None


@dataclass(frozen=True)
class ScenarioFacts:
    product_code: int
    settled_account_count: int
    settled_weighted_rate: float
    new_rate: float
    instalment_relief_pct: float
    total_cost_increase_pct: float
    term_extension_months: float
    new_money: float
    settlement_total: float
    new_dsr: float
    discretionary_income_after: float
    external_proportion: float
    consolidations_last_24_months: int


def evaluate_interventions(
    facts: ScenarioFacts, decision_date: date, thresholds: dict = DEFAULT_THRESHOLDS,
    stack_enabled: bool = True,
) -> list[InterventionResult]:
    results: list[InterventionResult] = []
    provenance = {"product_code": facts.product_code}

    def check(code: int, name: str, actual, threshold, ok: bool, overlay_id: str | None = None) -> None:
        results.append(InterventionResult(code, name, PASSED if ok else FAILED, actual, threshold, overlay_id))

    check(CON_INT_01, "max_accounts_settled", facts.settled_account_count, thresholds["CON-INT-01_max_accounts"],
          facts.settled_account_count <= thresholds["CON-INT-01_max_accounts"])

    results.append(InterventionResult(CON_INT_02, "recently_opened", NOT_APPLICABLE, None, None))

    check(CON_INT_03, "min_instalment_reduction", facts.instalment_relief_pct, thresholds["CON-INT-03_min_relief_pct"],
          facts.instalment_relief_pct >= thresholds["CON-INT-03_min_relief_pct"])

    anti_harm_result = INTERVENTION_ADJUSTMENTS.apply_stack(
        "anti_harm_ceiling_pct", thresholds["CON-INT-04_anti_harm_pct"], provenance, decision_date,
        ADJUSTMENT_SET_ID, stack_enabled=stack_enabled,
    )
    check(CON_INT_04, "anti_harm_ceiling", facts.total_cost_increase_pct, anti_harm_result.adjusted_value,
          facts.total_cost_increase_pct <= anti_harm_result.adjusted_value,
          overlay_id=(anti_harm_result.adjustments_applied[0] if anti_harm_result.adjustments_applied else None))

    check(CON_INT_05, "max_term_extension", facts.term_extension_months,
          thresholds["CON-INT-05_max_term_extension_months"],
          facts.term_extension_months <= thresholds["CON-INT-05_max_term_extension_months"])

    if thresholds["CON-INT-06_rate_ceiling_binding"]:
        check(CON_INT_06, "rate_ceiling", facts.new_rate, facts.settled_weighted_rate,
              facts.new_rate <= facts.settled_weighted_rate)
    else:
        results.append(InterventionResult(CON_INT_06, "rate_ceiling", NOT_APPLICABLE, facts.new_rate,
                                           facts.settled_weighted_rate))

    results.append(InterventionResult(CON_INT_07, "debt_review", NOT_APPLICABLE, None, None))

    new_money_pct = facts.new_money / facts.settlement_total if facts.settlement_total else 0.0
    check(CON_INT_08, "new_money_cap", facts.new_money,
          min(facts.settlement_total * thresholds["CON-INT-08_new_money_pct"], thresholds["CON-INT-08_new_money_cap"]),
          new_money_pct <= thresholds["CON-INT-08_new_money_pct"] and facts.new_money <= thresholds["CON-INT-08_new_money_cap"])

    check(CON_INT_09, "post_consolidation_dsr", facts.new_dsr, thresholds["CON-INT-09_dsr_ceiling"],
          facts.new_dsr <= thresholds["CON-INT-09_dsr_ceiling"])

    if facts.product_code == vocab.PRODUCT_FLEX_CONSOLIDATION:
        check(CON_INT_10, "min_external_proportion", facts.external_proportion,
              thresholds["CON-INT-10_min_external_pct"],
              facts.external_proportion >= thresholds["CON-INT-10_min_external_pct"])
    else:
        results.append(InterventionResult(CON_INT_10, "min_external_proportion", NOT_APPLICABLE,
                                           facts.external_proportion, None))

    results.append(InterventionResult(CON_INT_11, "disputed_account", NOT_APPLICABLE, None, None))
    results.append(InterventionResult(CON_INT_12, "security_release", NOT_APPLICABLE, None, None))

    check(CON_INT_13, "max_consolidations_24m", facts.consolidations_last_24_months,
          thresholds["CON-INT-13_max_per_24_months"],
          facts.consolidations_last_24_months < thresholds["CON-INT-13_max_per_24_months"])

    check(CON_INT_14, "min_discretionary_income", facts.discretionary_income_after,
          thresholds["CON-INT-14_min_discretionary_income"],
          facts.discretionary_income_after >= thresholds["CON-INT-14_min_discretionary_income"])

    return results


def failed(results: list[InterventionResult]) -> list[InterventionResult]:
    return [r for r in results if r.status == FAILED]
