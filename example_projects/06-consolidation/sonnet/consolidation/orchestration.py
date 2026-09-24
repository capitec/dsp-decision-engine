"""The top-level orchestration: baseline -> (short-circuit to 03, or) candidate
generation (§5.5) -> per-scenario evaluation (§5.6) -> objective and selection
(§5.8) -> outcome.

One `frame_step`, like `waterfall.py`/`solve.py` in project 03 and every
per-record search in this codebase: the search has its own control flow (early
exit on the short circuit, a budget loop, an objective pass) that does not map
onto `decider`'s `dag`/`branch`/`loop` primitives cleanly, and a single,
directly unit-testable Python function is the shorter, clearer path to the
900 ms / 400-scenario correctness bar than decomposing it into framework state
(the same judgement 03's NOTES.md makes for its own solve, "made for the same
reason").

Per-scenario detail is emitted as **parallel primitive lists**, one entry per
evaluated scenario, never a `list[struct]` -- the same `frame_step` output
limitation documented in 00/02/03's NOTES.md (`ValueError: cannot parse numpy
data type dtype('O') into Polars data type`). A scenario's rejection reasons
(0..n per scenario) are joined into one comma-separated string per scenario
rather than a nested `list[list[int]]` column, for the same reason.
"""
from __future__ import annotations

import time

from decider import frame_step

import polars as pl

from consolidation import baseline as baseline_mod
from consolidation import objective as objective_mod, pricing11, reasons, search
from consolidation.interventions import DEFAULT_THRESHOLDS
from consolidation.rate_cards import Product20CardIndex
from consolidation.scenario_eval import evaluate_scenario
from consolidation import vocab

INTERACTIVE_BUDGET = 400
INTERACTIVE_TIME_BUDGET_MS = 900
STATUTORY_CEILING = 0.28  # NCA margin-over-repo ceiling, same convention as project 03


def _csv(values) -> str:
    return ",".join(str(v) for v in values)


def _run(row: dict, rate_card_flex_loan, product11_rows: list[dict] | None) -> dict:
    decision_date = row["decision_date"]
    channel_code = row["channel_code"]

    evidence, baseline_facts, out02 = baseline_mod.build_evidence_and_baseline(row)

    evaluator11 = pricing11.build_product11_evaluator(STATUTORY_CEILING, product11_rows)
    card_index20 = Product20CardIndex.build()

    # Achievable consolidation rate, for the short-circuit's rate-gap test: product
    # 11's own card, at the client's grade, at the reference term (§5.4's own
    # comparator, not a scenario -- computed once, cheaply).
    achievable_rate = evaluator11.rate_card_index.lookup(
        max(50_000.0, baseline_facts.total_instalment * 12), baseline_mod.REFERENCE_TERM_MONTHS, evidence.risk_grade,
    )
    achievable_consolidation_rate = (achievable_rate[0] / 100.0) if achievable_rate else 0.20

    sc = baseline_mod.evaluate_short_circuit(row, evidence, baseline_facts, out02, achievable_consolidation_rate)

    result = {
        "existing_obligations_baseline": out02["existing_obligations"],
        "baseline_total_instalment": baseline_facts.total_instalment,
        "baseline_weighted_average_rate": baseline_facts.weighted_average_rate,
        "baseline_total_remaining_cost": baseline_facts.total_remaining_cost,
        "baseline_account_count": baseline_facts.account_count,
        "baseline_provider_count": baseline_facts.provider_count,
        "achievable_consolidation_rate": round(achievable_consolidation_rate, 4),
        "short_circuit_applies": sc["short_circuit_applies"],
        "short_circuit_condition_names": _csv(sc["short_circuit_conditions"].keys()),
        "short_circuit_condition_verdicts": _csv(sc["short_circuit_conditions"].values()),
        "post_advance_debt_service_ratio": sc["post_advance_debt_service_ratio"],
        "net_monthly_income": evidence.net_monthly_income, "living_expenses": evidence.living_expenses,
        "evidence_sufficiency_code": evidence.evidence_sufficiency_code,
    }

    if sc["short_circuit_applies"] and row.get("requested_amount"):
        plain = baseline_mod.plain_grant_via_project03(row, evidence, rate_card_flex_loan)
        result.update({
            "search_outcome_code": vocab.OUTCOME_APPROVE if plain.get("outcome_code") == "approve" else vocab.OUTCOME_REFER,
            "primary_reason_code": None, "search_decline_reason_codes": [],
            "reason_registry_version": reasons.DECLINE_REGISTRY.version,
            "plain_grant_via_project03": True,
            "chosen_product_code": 10, "chosen_offered_amount": plain.get("recommended_amount"),
            "chosen_term_months": plain.get("recommended_term"), "chosen_instalment": plain.get("recommended_instalment"),
            "chosen_total_cost_of_credit": plain.get("recommended_total_cost_of_credit"),
            "chosen_settlement_set": "", "chosen_scenario_id": -1, "chosen_nominal_annual_rate": None,
            "chosen_instalment_relief": 0.0, "chosen_total_cost_delta": 0.0, "chosen_objective_score": None,
            "search_budget": 0, "search_consumption": 0, "search_termination_cause": "short_circuited",
            "objective_id": None, "objective_weight_names": "", "objective_weight_values": "",
            "objective_overlay_ids": "", "indifference_band_triggered": False,
            "runner_up_scenario_ids": [], "runner_up_objective_scores": [], "runner_up_product_codes": [],
            "shadow_best_scenario_id": -1, "shadow_best_objective_score": None,
            "scenario_ids": [], "scenario_settlement_sets": [], "scenario_generating_rules": [],
            "scenario_product_codes": [], "scenario_term_months": [], "scenario_viable": [],
            "scenario_rejection_reason_codes": [], "scenario_offered_amounts": [], "scenario_nominal_rates": [],
            "scenario_instalments": [], "scenario_total_costs_of_credit": [], "scenario_objective_scores": [],
            "ordering_rule_set_version": search.ORDERING_RULE_SET_VERSION,
        })
        return result

    accounts = row.get("accounts") or []
    accounts_by_ref = {a["account_ref"]: a for a in accounts}
    settleability_by_ref = dict(zip(row["settleability_account_refs"], row["settleability_codes"]))
    settlement_amounts_by_ref = dict(zip(row.get("settlement_account_refs", []), row.get("settlement_amounts", [])))

    settleable = [
        search.SettleableAccount(
            account_ref=ref, settlement_amount=settlement_amounts_by_ref.get(ref, 0.0),
            instalment=accounts_by_ref[ref].get("instalment") or 0.0,
            nominal_annual_rate=accounts_by_ref[ref].get("nominal_annual_rate") or 0.0,
            remaining_term_months=accounts_by_ref[ref].get("remaining_term_months"),
            provider_code=accounts_by_ref[ref].get("provider_code"), is_internal=bool(accounts_by_ref[ref].get("is_internal")),
            months_in_arrears=accounts_by_ref[ref].get("months_in_arrears") or 0,
            account_type_code=accounts_by_ref[ref].get("account_type_code"),
            is_mandatory=ref in (row.get("mandatory_account_refs") or []),
        )
        for ref in (row.get("settleable_account_refs") or []) if ref in accounts_by_ref
    ]

    thresholds = dict(DEFAULT_THRESHOLDS)
    start = time.monotonic()
    candidates, termination_cause = search.generate_scenarios(
        settleable, row.get("mandatory_account_refs") or [], thresholds["CON-INT-05_max_term_extension_months"],
        INTERACTIVE_BUDGET,
    )
    for i, c in enumerate(candidates):
        c["scenario_id"] = i + 1

    outcomes = []
    for c in candidates:
        if (time.monotonic() - start) * 1000 > INTERACTIVE_TIME_BUDGET_MS:
            termination_cause = vocab.TERMINATION_BUDGET
            break
        outcomes.append(evaluate_scenario(
            c, accounts_by_ref, settleability_by_ref, settlement_amounts_by_ref, evidence, baseline_facts,
            evaluator11, card_index20, row.get("requested_amount") or 0.0, channel_code,
            row.get("consolidations_last_24_months", 0), decision_date, thresholds,
        ))

    viable_measures = [o.measures for o in outcomes if o.viable and o.measures is not None]
    weights, objective_overlay_ids = objective_mod.resolve_weights(
        row.get("objective_id", vocab.OBJ_MIN_COMMITMENT), channel_code, decision_date,
    )
    settlement_sets = {o.scenario_id: o.settlement_set for o in outcomes}
    product_codes = {o.scenario_id: o.product_code for o in outcomes}
    ranked, indifferent = objective_mod.rank_and_select(viable_measures, weights, settlement_sets, product_codes)
    shadow = objective_mod.shadow_best(viable_measures)

    outcome_by_id = {o.scenario_id: o for o in outcomes}
    winner = ranked[0] if ranked else None
    runners_up = ranked[1:3]

    all_rejection_codes = sorted({code for o in outcomes for code in o.rejection_reason_codes})
    ranked_rejections, primary_rejection = (
        reasons.REJECTION_REGISTRY.rank(all_rejection_codes) if all_rejection_codes else ([], None)
    )

    if winner is not None:
        w_out = outcome_by_id[winner.measures.scenario_id]
        search_outcome_code = vocab.OUTCOME_APPROVE
        decline_codes, primary_reason, registry_version = [], None, reasons.DECLINE_REGISTRY.version
    else:
        search_outcome_code = vocab.OUTCOME_DECLINE
        decline_codes, primary_reason = reasons.DECLINE_REGISTRY.rank([reasons.D_NO_VIABLE_SCENARIO])
        registry_version = reasons.DECLINE_REGISTRY.version
        w_out = None

    result.update({
        "search_outcome_code": search_outcome_code, "primary_reason_code": primary_reason, "search_decline_reason_codes": decline_codes,
        "reason_registry_version": registry_version, "plain_grant_via_project03": False,
        "search_budget": INTERACTIVE_BUDGET, "search_consumption": len(outcomes),
        "search_termination_cause": termination_cause, "ordering_rule_set_version": search.ORDERING_RULE_SET_VERSION,
        "objective_id": row.get("objective_id", vocab.OBJ_MIN_COMMITMENT),
        "objective_weight_names": _csv(weights.keys()), "objective_weight_values": _csv(round(v, 4) for v in weights.values()),
        "objective_overlay_ids": _csv(objective_overlay_ids), "indifference_band_triggered": indifferent,
        "shadow_best_scenario_id": shadow.scenario_id if shadow else -1,
        "shadow_best_objective_score": objective_mod.objective_score(shadow, weights) if shadow else None,
        "scenario_ids": [o.scenario_id for o in outcomes],
        "scenario_settlement_sets": [_csv(sorted(o.settlement_set)) for o in outcomes],
        "scenario_generating_rules": [o.generating_rule for o in outcomes],
        "scenario_product_codes": [o.product_code for o in outcomes],
        "scenario_term_months": [o.term_months for o in outcomes],
        "scenario_viable": [o.viable for o in outcomes],
        "scenario_rejection_reason_codes": [_csv(o.rejection_reason_codes) for o in outcomes],
        "scenario_offered_amounts": [o.offered_amount or 0.0 for o in outcomes],
        "scenario_nominal_rates": [o.nominal_annual_rate or 0.0 for o in outcomes],
        "scenario_instalments": [o.instalment or 0.0 for o in outcomes],
        "scenario_total_costs_of_credit": [o.total_cost_of_credit or 0.0 for o in outcomes],
        "scenario_objective_scores": [
            objective_mod.objective_score(o.measures, weights) if o.measures else 0.0 for o in outcomes
        ],
        "runner_up_scenario_ids": [r.measures.scenario_id for r in runners_up],
        "runner_up_objective_scores": [r.score for r in runners_up],
        "runner_up_product_codes": [product_codes[r.measures.scenario_id] for r in runners_up],
    })

    if w_out is not None:
        result.update({
            "chosen_scenario_id": w_out.scenario_id, "chosen_settlement_set": _csv(sorted(w_out.settlement_set)),
            "chosen_product_code": w_out.product_code, "chosen_term_months": w_out.term_months,
            "chosen_offered_amount": w_out.offered_amount, "chosen_nominal_annual_rate": w_out.nominal_annual_rate,
            "chosen_instalment": w_out.instalment, "chosen_total_cost_of_credit": w_out.total_cost_of_credit,
            "chosen_instalment_relief": w_out.instalment_relief, "chosen_total_cost_delta": w_out.total_cost_delta,
            "chosen_objective_score": winner.score,
        })
    else:
        result.update({
            "chosen_scenario_id": -1, "chosen_settlement_set": "", "chosen_product_code": None,
            "chosen_term_months": None, "chosen_offered_amount": None, "chosen_nominal_annual_rate": None,
            "chosen_instalment": None, "chosen_total_cost_of_credit": None, "chosen_instalment_relief": 0.0,
            "chosen_total_cost_delta": 0.0, "chosen_objective_score": None,
        })

    return result


_READS = [
    "decision_id", "decision_date", "channel_code", "requested_amount", "risk_grade", "accounts",
    "client_nominated_settle", "client_excluded_settle", "dependants_count", "employment_type_code",
    "payslip_income", "variable_pay_history", "declared_expenses", "statement_expenses", "bureau_as_of_date",
    "court_ordered_deductions", "applicant_age_years", "segment_code", "objective_id",
    "consolidations_last_24_months", "settleability_account_refs", "settleability_codes",
    "settleable_account_refs", "mandatory_account_refs", "settlement_account_refs", "settlement_amounts",
]
_WRITES = [
    "existing_obligations_baseline", "baseline_total_instalment", "baseline_weighted_average_rate",
    "baseline_total_remaining_cost", "baseline_account_count", "baseline_provider_count",
    "achievable_consolidation_rate", "short_circuit_applies", "short_circuit_condition_names",
    "short_circuit_condition_verdicts", "post_advance_debt_service_ratio", "net_monthly_income",
    "living_expenses", "evidence_sufficiency_code", "search_outcome_code", "primary_reason_code",
    "search_decline_reason_codes", "reason_registry_version", "plain_grant_via_project03", "search_budget",
    "search_consumption", "search_termination_cause", "ordering_rule_set_version", "objective_id",
    "objective_weight_names", "objective_weight_values", "objective_overlay_ids",
    "indifference_band_triggered", "shadow_best_scenario_id", "shadow_best_objective_score", "scenario_ids",
    "scenario_settlement_sets", "scenario_generating_rules", "scenario_product_codes", "scenario_term_months",
    "scenario_viable", "scenario_rejection_reason_codes", "scenario_offered_amounts", "scenario_nominal_rates",
    "scenario_instalments", "scenario_total_costs_of_credit", "scenario_objective_scores",
    "runner_up_scenario_ids", "runner_up_objective_scores", "runner_up_product_codes", "chosen_scenario_id",
    "chosen_settlement_set", "chosen_product_code", "chosen_term_months", "chosen_offered_amount",
    "chosen_nominal_annual_rate", "chosen_instalment", "chosen_total_cost_of_credit",
    "chosen_instalment_relief", "chosen_total_cost_delta", "chosen_objective_score",
]


def build_orchestration_step(rate_card_flex_loan, product11_rows: list[dict] | None = None):
    def run(df: pl.DataFrame) -> pl.DataFrame:
        results = [_run(row, rate_card_flex_loan, product11_rows) for row in df.select(_READS).to_dicts()]
        return df.with_columns(pl.DataFrame(results))

    return frame_step(run, reads=_READS, writes=_WRITES)
