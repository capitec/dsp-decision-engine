"""Which scenario wins, which are shown as alternatives, and why.

The Bank has five objectives and blends them with a weight vector that is configuration: a row per channel in the
`objective_weights` table, which Credit Committee edits without a release. The vector is also an overlay target: an
approved, expiring re-weight changes the winner with no rule changing, and its id is recorded. Every component is a
ratio to the baseline, never scaled between the best and worst scenario evaluated, so the winner does not depend on
which losers the budget happened to reach.
"""
from __future__ import annotations

from datetime import date

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from decider import ParamTable, allow_fallback, param, param_table, step

from consolidation import vocab

OBJECTIVES = vocab.OBJECTIVES
ADJUSTMENT_SET_ID = "AS-CON-OBJ-2026.09"

# channel 0 is the row used by any channel without its own
OBJECTIVE_WEIGHTS = param_table({"channel_code": int, **dict.fromkeys(OBJECTIVES, float)}, default=[
    {"channel_code": 0, "new_money": 0.0, "min_commitment": 1.0, "min_total_cost": 0.0, "bank_value": 0.0, "client_outcome": 0.0},
    {"channel_code": 1, "new_money": 0.0, "min_commitment": 1.0, "min_total_cost": 0.0, "bank_value": 0.0, "client_outcome": 0.0},
    {"channel_code": 2, "new_money": 0.0, "min_commitment": 0.0, "min_total_cost": 0.0, "bank_value": 0.0, "client_outcome": 1.0},
    {"channel_code": 4, "new_money": 1.0, "min_commitment": 0.0, "min_total_cost": 0.0, "bank_value": 0.0, "client_outcome": 0.0},
])


def _reweight(objective: str, value: float) -> Adjustment:
    return Adjustment(
        adjustment_id=f"ADJ-CON-OBJ-2026-{'001' if objective == 'min_commitment' else '002'}", kind="objective_reweight",
        target=f"weight_{objective}", effect=AdjustmentEffect("set", value), scope={"channel_code": 1}, stack_position=1,
        owner="Credit Committee", approval_reference="CC-2026-014",
        rationale="Branch channel weighted towards instalment relief this quarter",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31))


OBJECTIVE_OVERLAYS = AdjustmentRegister([_reweight("min_commitment", 0.6), _reweight("new_money", 0.4)])


def weights_in_force(table: ParamTable, channel_code: int, decision_date: date, stack_enabled: bool) -> tuple[dict, list[str]]:
    """The normalised weight vector for a channel, after any re-weight overlay, and the ids of the overlays that applied."""
    channels = table.channel_code.tolist()
    row = channels.index(channel_code) if channel_code in channels else channels.index(0)
    weights, applied = {}, []
    for name in OBJECTIVES:
        result = OBJECTIVE_OVERLAYS.apply_stack(
            f"weight_{name}", float(getattr(table, name)[row]), {"channel_code": channel_code}, decision_date,
            ADJUSTMENT_SET_ID, stack_enabled=stack_enabled)
        weights[name] = result.adjusted_value
        applied += result.adjustments_applied
    total = sum(weights.values())
    return {name: round(w / total, 6) for name, w in weights.items()}, sorted(set(applied))


def components(measures: dict, outcome_weights: dict) -> dict:
    """One score per objective. `client_outcome` blends relief, cost, exits, rate and providers by `outcome_weights`."""
    w = outcome_weights
    return {
        "new_money": measures["new_money_ratio"], "min_commitment": measures["instalment_relief_ratio"],
        "min_total_cost": -measures["total_cost_increase_ratio"], "bank_value": measures["bank_value_ratio"],
        "client_outcome": w["relief"] * measures["instalment_relief_ratio"] - w["cost"] * measures["total_cost_increase_ratio"]
        + w["exited"] * measures["accounts_exited_proportion"] + w["rate"] * measures["weighted_rate_reduction"]
        + w["providers"] * measures["providers_exited_proportion"],
    }


def rank(scenarios: list[dict], weights: dict, outcome_weights: dict, indifference_band: float) -> dict:
    """Score every viable scenario and pick the winner and up to two runners-up.

    A rejected scenario is never a runner-up, and the three are distinct: at most one per settlement set and product.
    Ties break on the lowest total cost of credit, then the fewest accounts settled, then the lowest scenario id.
    Returns `scored`, `top` (scenario ids, winner first), `indifferent`, `versus` and `shadow`.
    """
    by_id = {s["scenario_id"]: s for s in scenarios}
    scored = []
    for s in (s for s in scenarios if s["viable"]):
        parts = components(s["measures"], outcome_weights)
        score = round(sum(weights[n] * parts[n] for n in OBJECTIVES), 6)
        scored.append({"scenario_id": s["scenario_id"], "objective_score": score,
                       "components": {n: round(parts[n], 6) for n in OBJECTIVES}})
    scored.sort(key=lambda r: (-r["objective_score"], by_id[r["scenario_id"]]["total_cost_of_credit"],
                               len(by_id[r["scenario_id"]]["settlement_set"]), r["scenario_id"]))
    top, shown = [], set()
    for r in scored:
        key = (tuple(by_id[r["scenario_id"]]["settlement_set"]), by_id[r["scenario_id"]]["product_code"])
        if key not in shown and len(top) < 3:
            shown.add(key)
            top.append(r)
    indifferent = len(top) > 1 and top[0]["objective_score"] != 0 and (
        abs(top[0]["objective_score"] - top[1]["objective_score"]) / abs(top[0]["objective_score"]) < indifference_band)
    versus = []
    for r in top[1:]:
        gap = {n: round(weights[n] * (top[0]["components"][n] - r["components"][n]), 6) for n in OBJECTIVES}
        versus.append({"winner": top[0]["scenario_id"], "runner_up": r["scenario_id"],
                       "score_difference": round(top[0]["objective_score"] - r["objective_score"], 6),
                       "weighted_component_differences": gap, "decisive_component": max(gap, key=lambda n: abs(gap[n]))})
    best_for_client = max(scored, key=lambda r: (r["components"][vocab.CLIENT_OUTCOME], -r["scenario_id"]), default=None)
    shadow = {"scenario_id": best_for_client["scenario_id"],
              "client_outcome_score": best_for_client["components"][vocab.CLIENT_OUTCOME]} if best_for_client else {}
    return {"scored": scored, "top": top, "indifferent": indifferent, "versus": versus, "shadow": shadow}


@allow_fallback
@step(outputs=("objective_weights", "objective_overlay_ids", "scored_scenarios", "chosen_scenario", "runners_up",
               "indifference_band_triggered", "winner_vs_runners_up", "shadow_best_scenario"))
def selection(
    scenarios: list[dict], channel_code: int, decision_date: date, objective_weights: ParamTable = OBJECTIVE_WEIGHTS,
    outcome_weight_relief: float = param(0.35, ge=0.0), outcome_weight_cost: float = param(0.25, ge=0.0),
    outcome_weight_exited: float = param(0.15, ge=0.0), outcome_weight_rate: float = param(0.15, ge=0.0),
    outcome_weight_providers: float = param(0.10, ge=0.0), indifference_band: float = param(0.02, ge=0.0, le=0.2),
    overlay_stack_enabled: bool = param(True, shared_key="overlay_stack_enabled"),
) -> tuple[dict, list[str], list[dict], dict, list[dict], bool, list[dict], dict]:
    """The weights in force, every viable scenario scored, the winner and runners-up, and why the winner won.

    `winner_vs_runners_up` names the component that accounts for each gap, and `shadow_best_scenario` is the best
    for the client whatever objective is in force: the answer to "what was the best outcome available to me".
    """
    weights, overlay_ids = weights_in_force(objective_weights, channel_code, decision_date, overlay_stack_enabled)
    outcome = {"relief": outcome_weight_relief, "cost": outcome_weight_cost, "exited": outcome_weight_exited,
               "rate": outcome_weight_rate, "providers": outcome_weight_providers}
    ranked = rank(scenarios, weights, outcome, indifference_band)
    by_id = {s["scenario_id"]: s for s in scenarios}
    top = [{**by_id[r["scenario_id"]], "objective_score": r["objective_score"]} for r in ranked["top"]]
    return (weights, overlay_ids, ranked["scored"], top[0] if top else {}, top[1:], ranked["indifferent"], ranked["versus"],
            ranked["shadow"])
