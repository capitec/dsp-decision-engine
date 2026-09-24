"""Stage 5.6 -- per-scenario evaluation (spec 06 §5.6). The reuse-heaviest module in
this project: re-derives obligations (project 00), calls project 02's capacity stage
"in scenario mode", and prices through project 11's own product logic (built on
project 03's `PriceEvaluator` for product 11, from scratch for product 20).

**Invariant across every scenario** (§5.6.1): `net_monthly_income`, `living_expenses`
and every income/expense figure are computed once per assessment
(`baseline.build_evidence`) and passed into every call here unchanged -- this module
never re-derives them. **Varies per scenario**: the obligation set, and everything
downstream of it.

**Calling project 02 "in scenario mode".** Project 02's `pipeline.py` publishes
`evidence_unit()`/`capacity_unit()` as the intended split (its own NOTES.md: "the
only stage project 06's 400-scenario search needs to re-run per scenario"), but
`capacity_unit()` is a `decider` `dag`, and running a `dag` through `Engine.score()`
400 times inside a 900 ms budget is the exact per-call overhead project 03's own
NOTES.md measured as too slow for its own (far cheaper) solve. What *is* cheap is
02's underlying plain-Python arithmetic: `capacity.max_affordable_instalment_before_overlay`
and `verdict.affordability_verdict_code`/`evidence_sufficiency_code` are already plain
functions (like every unit in project 00). The two small **tables** underneath them
-- the buffer grid and the residual floor -- are not published as plain callables,
only as `DecisionTableConfig` documents; this module calls their private
`_buffer_pct`/`_RESIDUAL_BANDS`/`_RESIDUAL_AMOUNTS` implementation directly rather
than re-derive the same numbers from a second copy. See NOTES.md "Framework
friction" and "Gaps in what I consumed" -- this is the single most consequential
reuse gap in the whole slice.
"""
from __future__ import annotations

from dataclasses import dataclass

from credit_core.obligations import _process as obligations_process

from assessment import capacity as capacity02
from assessment import verdict as verdict02

from consolidation import interventions, pricing11, pricing20, reasons, settlement
from consolidation.objective import ScenarioMeasures
from consolidation.rate_cards import Product20CardIndex
from consolidation import vocab

COST_OF_FUNDS = 0.08
EXPECTED_LOSS_RATE = 0.02
STANDARD_TERM_MONTHS = 36  # revolving remaining-cost horizon, matching credit_core.obligations


def buffer_pct(risk_grade: int, product_code: int) -> float:
    """02's own buffer grid, called directly (see module docstring)."""
    return capacity02._buffer_pct(risk_grade, product_code)


def residual_floor(dependants_count: int) -> float:
    for (lo, hi), amount in zip(capacity02._RESIDUAL_BANDS, capacity02._RESIDUAL_AMOUNTS):
        if lo <= dependants_count < hi:
            return amount
    return capacity02._RESIDUAL_AMOUNTS[-1]


@dataclass(frozen=True)
class EvidenceUnit:
    """§5.6.1's invariant: computed once (`baseline.py`), read by every scenario.
    `evidence_sufficiency_code` is itself invariant (02's own function depends only on
    income evidence quality, never on obligations -- verified against
    `assessment/verdict.py`'s signature), so it too is computed once here rather than
    per scenario."""
    net_monthly_income: float
    living_expenses: float
    dependants_count: int
    court_ordered_deductions: float
    risk_grade: int
    applicant_age_years: float
    evidence_sufficiency_code: int


@dataclass(frozen=True)
class BaselineFacts:
    weighted_average_rate: float
    total_instalment: float
    total_remaining_cost: float
    account_count: int
    provider_count: int


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: int
    settlement_set: frozenset
    generating_rule: str
    product_code: int
    term_months: int
    viable: bool
    rejection_reason_codes: list[int]
    intervention_results: list[interventions.InterventionResult]
    offered_amount: float | None
    nominal_annual_rate: float | None
    nominal_annual_rate_unadjusted: float | None
    rate_cell_id: str | None
    rate_card_version: str | None
    instalment: float | None
    total_cost_of_credit: float | None
    effective_annual_rate: float | None
    solve_iterations: int
    existing_obligations_after: float | None
    discretionary_income: float | None
    max_affordable_instalment: float | None
    affordability_verdict_code: int | None
    measures: ScenarioMeasures | None
    instalment_relief: float | None
    total_cost_delta: float | None


def _remaining_cost(account: dict) -> float:
    """Baseline "total remaining cost" (§5.4): remaining instalments x instalment for a
    term account, or the standard-horizon amortised cost for a revolving one."""
    term = account.get("remaining_term_months")
    instalment = account.get("instalment") or 0.0
    if term:
        return instalment * term
    return instalment * STANDARD_TERM_MONTHS


def compute_baseline_facts(accounts: list[dict]) -> BaselineFacts:
    total_balance = sum((a.get("balance") or 0.0) for a in accounts) or 1.0
    weighted_rate = sum((a.get("balance") or 0.0) * (a.get("nominal_annual_rate") or 0.0) for a in accounts) / total_balance
    return BaselineFacts(
        weighted_average_rate=round(weighted_rate, 4),
        total_instalment=round(sum((a.get("instalment") or 0.0) for a in accounts), 2),
        total_remaining_cost=round(sum(_remaining_cost(a) for a in accounts), 2),
        account_count=len(accounts),
        provider_count=len({a.get("provider_code") for a in accounts if not a.get("is_internal")}),
    )


def _reduce_inventory(accounts: list[dict], settlement_set: frozenset, settleability_by_ref: dict[int, int]) -> list[dict]:
    """§5.6.2 step 1: remove settled accounts; partially-settleable revolving accounts
    are zeroed rather than removed (`core.obligations` must see the post-action
    state)."""
    reduced = []
    for a in accounts:
        ref = a.get("account_ref")
        if ref not in settlement_set:
            reduced.append(a)
            continue
        if settleability_by_ref.get(ref) == vocab.SETTLE_PARTIAL_REVOLVING:
            zeroed = dict(a)
            zeroed["balance"], zeroed["limit"], zeroed["credit_limit"], zeroed["instalment"] = 0.0, 0.0, 0.0, 0.0
            reduced.append(zeroed)
        # else: fully settled and closed -- dropped from the inventory entirely.
    return reduced


def _to_obligations_shape(account: dict) -> dict:
    """00's `_process` expects `limit`/`closed`; 06's request shape carries
    `credit_limit`/`account_status_code` (§4.2). A field-name adapter, not a fork."""
    closed_statuses = {2}  # 06's local vocabulary: 2 = closed
    # No `settlement_quote` key: 00's `_process` reads it with `.get(...)` (defaults to
    # `None`), and 06 never has a real value for it -- omitting the key rather than
    # always sending `None` avoids a struct field that is null in every row of the
    # list, which `decider`'s arrow import cannot type (see NOTES.md "Framework
    # friction": `ArrowImportError: Expected array with 0 buffer(s) but found 1
    # buffer(s)`, the struct-field analogue of the documented empty-list-column crash).
    return {
        "account_type_code": account.get("account_type_code"), "balance": account.get("balance"),
        "limit": account.get("credit_limit") or account.get("limit"), "instalment": account.get("instalment"),
        "months_in_arrears": account.get("months_in_arrears"),
        "closed": account.get("account_status_code") in closed_statuses,
        "is_internal": account.get("is_internal"),
    }


def rederive_obligations(accounts: list[dict], settlement_set: frozenset, settleability_by_ref: dict[int, int]) -> dict:
    reduced = _reduce_inventory(accounts, settlement_set, settleability_by_ref)
    bureau = [_to_obligations_shape(a) for a in reduced if not a.get("is_internal")]
    internal = [_to_obligations_shape(a) for a in reduced if a.get("is_internal")]
    return obligations_process(bureau, internal)


def _external_proportion(accounts_by_ref: dict, settlement_set: frozenset, settlement_amounts_by_ref: dict) -> float:
    total = sum(settlement_amounts_by_ref.get(r, 0.0) for r in settlement_set) or 1.0
    external = sum(
        settlement_amounts_by_ref.get(r, 0.0) for r in settlement_set
        if not accounts_by_ref.get(r, {}).get("is_internal")
    )
    return round(external / total, 4)


def _bank_expected_value(
    new_amount: float, new_rate: float, accounts_by_ref: dict, settlement_set: frozenset,
) -> float:
    """OBJ-04 (§5.8): incremental margin less expected loss, less margin forgone on the
    Bank's own settled accounts -- "a consolidation that refinances the Bank's own 26%
    loan at 19% destroys value" is exactly what the third term prevents."""
    incremental_margin = new_amount * max(0.0, new_rate - COST_OF_FUNDS - EXPECTED_LOSS_RATE)
    forgone = sum(
        (accounts_by_ref[r].get("balance") or 0.0) * max(0.0, (accounts_by_ref[r].get("nominal_annual_rate") or 0.0) - COST_OF_FUNDS)
        for r in settlement_set if accounts_by_ref.get(r, {}).get("is_internal")
    )
    return round(incremental_margin - forgone, 2)


def evaluate_scenario(
    scenario: dict, accounts_by_ref: dict[int, dict], settleability_by_ref: dict[int, int],
    settlement_amounts_by_ref: dict[int, float], evidence: EvidenceUnit, baseline: BaselineFacts,
    evaluator11, card_index20: Product20CardIndex, requested_amount: float, channel_code: int,
    consolidations_last_24_months: int, decision_date, thresholds: dict, stack_enabled: bool = True,
) -> ScenarioOutcome:
    settlement_set: frozenset = scenario["settlement_set"]
    product_code = scenario["product_code"]
    term_months = scenario["term_months"]

    obligations_after = rederive_obligations(list(accounts_by_ref.values()), settlement_set, settleability_by_ref)
    discretionary_income = round(
        evidence.net_monthly_income - evidence.living_expenses - obligations_after["existing_obligations"]
        - evidence.court_ordered_deductions, 2,
    )
    buf = buffer_pct(evidence.risk_grade, product_code)
    floor = residual_floor(evidence.dependants_count)
    max_instalment, binding_code = capacity02.max_affordable_instalment_before_overlay(discretionary_income, buf, floor)

    settlement_total = round(sum(settlement_amounts_by_ref.get(r, 0.0) for r in settlement_set), 2)
    # New money is not itself a search dimension this slice (SCOPE.md does not ask for a
    # new-money bracket alongside settlement set/product/term). It is clamped to what
    # CON-INT-08's base policy would allow for *this* settlement set -- otherwise every
    # scenario carries the client's full request regardless of set size, which makes a
    # R6 800 settlement carry a R60 000 advance and fail CON-INT-08 by construction
    # rather than on its own merits. CON-INT-08 is still evaluated (an overlay can
    # tighten its cap below this clamp) -- see NOTES.md "Spec problems".
    new_money = min(requested_amount or 0.0, settlement_total * thresholds["CON-INT-08_new_money_pct"],
                     thresholds["CON-INT-08_new_money_cap"])
    external_proportion = _external_proportion(accounts_by_ref, settlement_set, settlement_amounts_by_ref)
    longest_settled_term = max((accounts_by_ref[r].get("remaining_term_months") or 0) for r in settlement_set) if settlement_set else 0
    settled_balance = sum((accounts_by_ref[r].get("balance") or 0.0) for r in settlement_set) or 1.0
    settled_weighted_rate = sum(
        (accounts_by_ref[r].get("balance") or 0.0) * (accounts_by_ref[r].get("nominal_annual_rate") or 0.0)
        for r in settlement_set
    ) / settled_balance
    settled_instalment_total = sum((accounts_by_ref[r].get("instalment") or 0.0) for r in settlement_set)
    settled_remaining_cost = sum(_remaining_cost(accounts_by_ref[r]) for r in settlement_set)

    rejections: list[int] = []
    solve_iterations = 0
    result_amount = result_rate = result_rate_unadj = result_cell = result_version = None
    result_instalment = result_total_cost = result_ear = None
    new_dsr = None
    measures = None
    verdict_code = verdict02.INDETERMINATE

    if product_code == vocab.PRODUCT_FLEX_CONSOLIDATION:
        base_amount = round(settlement_total + settlement.settlement_buffer(settlement_total) + new_money, 2)
        solved = pricing11.solve_required_advance(
            evaluator11, base_amount, term_months, evidence.risk_grade, evidence.applicant_age_years,
            external_proportion,
        )
        solve_iterations = solved.iterations
        if not solved.pricing.priced:
            rejections.append(reasons.REJ_NOT_PRICED)
        elif not solved.converged:
            rejections.append(reasons.REJ_SOLVE_NOT_CONVERGED)
        else:
            result_amount, result_instalment = solved.offered_amount, solved.pricing.instalment
            result_rate, result_rate_unadj = solved.pricing.nominal_annual_rate, solved.pricing.nominal_annual_rate
            result_cell, result_version = solved.pricing.rate_cell_id, solved.pricing.rate_card_version
            result_total_cost, result_ear = solved.pricing.total_cost_of_credit, solved.pricing.effective_annual_rate
            new_total_commitment = result_instalment + (obligations_after["existing_obligations"])
            new_dsr = round(new_total_commitment / evidence.net_monthly_income, 4) if evidence.net_monthly_income else None
            verdict_code = verdict02.affordability_verdict_code(
                evidence.evidence_sufficiency_code, max_instalment, proposed_instalment=result_instalment,
            )
            if verdict_code == verdict02.FAIL:
                rejections.append(reasons.REJ_AFFORDABILITY_FAIL)

    elif product_code == vocab.PRODUCT_BALANCE_TRANSFER:
        priced20 = pricing20.price_product20(card_index20, settlement_total, evidence.risk_grade, term_months)
        if not priced20.priced:
            rejections.append(reasons.REJ_NOT_PRICED)
        else:
            result_amount, result_instalment = priced20.approved_limit, priced20.monthly_payment
            result_rate, result_rate_unadj = priced20.promo_rate, priced20.promo_rate
            result_cell, result_version = priced20.promo_cell_id, "rc-p20-2026.09"
            result_total_cost, result_ear = priced20.total_cost_of_credit, priced20.reversion_rate
            new_total_commitment = priced20.stressed_payment + obligations_after["existing_obligations"]
            new_dsr = round(new_total_commitment / evidence.net_monthly_income, 4) if evidence.net_monthly_income else None
            # §5.6.6: affordability is tested against the *stressed* payment, never the
            # promotional minimum -- the one place this product's verdict input differs.
            verdict_code = verdict02.affordability_verdict_code(
                evidence.evidence_sufficiency_code, max_instalment, proposed_instalment=priced20.stressed_payment,
            )
            if verdict_code == verdict02.FAIL:
                rejections.append(reasons.REJ_AFFORDABILITY_FAIL)
    else:
        rejections.append(reasons.REJ_NOT_ROUTABLE)

    intervention_results: list[interventions.InterventionResult] = []
    instalment_relief = total_cost_delta = None
    if result_instalment is not None:
        instalment_relief = round(settled_instalment_total - result_instalment, 2)
        total_cost_delta = round((result_total_cost or 0.0) - settled_remaining_cost, 2)
        facts = interventions.ScenarioFacts(
            product_code=product_code, settled_account_count=len(settlement_set),
            settled_weighted_rate=round(settled_weighted_rate, 4), new_rate=result_rate or 0.0,
            instalment_relief_pct=round(instalment_relief / settled_instalment_total, 4) if settled_instalment_total else 0.0,
            total_cost_increase_pct=round(total_cost_delta / settled_remaining_cost, 4) if settled_remaining_cost else 0.0,
            term_extension_months=max(0.0, term_months - longest_settled_term),
            new_money=new_money, settlement_total=settlement_total, new_dsr=new_dsr or 0.0,
            discretionary_income_after=round(discretionary_income - (result_instalment or 0.0), 2),
            external_proportion=external_proportion,
            consolidations_last_24_months=consolidations_last_24_months,
        )
        intervention_results = interventions.evaluate_interventions(facts, decision_date, thresholds, stack_enabled)
        rejections.extend(1000 + r.code for r in interventions.failed(intervention_results))

        if not rejections:
            bank_value = _bank_expected_value(result_amount or 0.0, result_rate or 0.0, accounts_by_ref, settlement_set)
            new_weighted_rate = (
                ((result_amount or 0.0) * (result_rate or 0.0) + (baseline.weighted_average_rate * sum(
                    (a.get("balance") or 0.0) for r, a in accounts_by_ref.items() if r not in settlement_set)))
                / max(1.0, (result_amount or 0.0) + sum(
                    (a.get("balance") or 0.0) for r, a in accounts_by_ref.items() if r not in settlement_set))
            )
            providers_exited = len({accounts_by_ref[r].get("provider_code") for r in settlement_set if not accounts_by_ref[r].get("is_internal")})
            measures = ScenarioMeasures(
                scenario_id=scenario["scenario_id"], new_money=new_money,
                instalment_relief_ratio=round(instalment_relief / baseline.total_instalment, 4) if baseline.total_instalment else 0.0,
                total_cost_increase_ratio=round(total_cost_delta / baseline.total_remaining_cost, 4) if baseline.total_remaining_cost else 0.0,
                bank_value_ratio=round(bank_value / settlement_total, 4) if settlement_total else 0.0,
                accounts_exited_proportion=round(len(settlement_set) / baseline.account_count, 4) if baseline.account_count else 0.0,
                weighted_rate_reduction=round(baseline.weighted_average_rate - new_weighted_rate, 4),
                providers_exited_proportion=round(providers_exited / baseline.provider_count, 4) if baseline.provider_count else 0.0,
                requested_amount=requested_amount or 0.0,
            )

    return ScenarioOutcome(
        scenario_id=scenario["scenario_id"], settlement_set=settlement_set,
        generating_rule=scenario["generating_rule"], product_code=product_code, term_months=term_months,
        viable=not rejections, rejection_reason_codes=sorted(set(rejections)),
        intervention_results=intervention_results, offered_amount=result_amount, nominal_annual_rate=result_rate,
        nominal_annual_rate_unadjusted=result_rate_unadj, rate_cell_id=result_cell, rate_card_version=result_version,
        instalment=result_instalment, total_cost_of_credit=result_total_cost, effective_annual_rate=result_ear,
        solve_iterations=solve_iterations, existing_obligations_after=obligations_after["existing_obligations"],
        discretionary_income=discretionary_income, max_affordable_instalment=max_instalment,
        affordability_verdict_code=verdict_code, measures=measures,
        instalment_relief=instalment_relief, total_cost_delta=total_cost_delta,
    )
