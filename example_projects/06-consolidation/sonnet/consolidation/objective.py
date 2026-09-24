"""Stage 5.8 -- objective and selection (spec 06 §5.8).

The Bank has five objectives (OBJ-01..05) and blends them by a weight vector
that "is itself an overlay target" (§5.8 requirement 2). Every component is
expressed as a **ratio to the baseline** (requirement 3: "not scaled between
the best and worst... among the evaluated scenarios", so the winner cannot
depend on which losers the budget happened to evaluate) -- applied uniformly
across all five measures here, not only OBJ-05's own five sub-components,
because a blend across OBJ-01..05 cannot be summed sensibly otherwise (rands
against a debt-service ratio against a probability-weighted margin have no
common scale; NOTES.md "Spec problems" records this as a declared reading,
not a literal instruction).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

from consolidation import vocab

INDIFFERENCE_BAND = 0.02  # 2% (§5.8 requirement 5)

OBJECTIVE_WEIGHT_TARGETS = {
    vocab.OBJ_NEW_MONEY: "objective_weight_new_money",
    vocab.OBJ_MIN_COMMITMENT: "objective_weight_min_commitment",
    vocab.OBJ_MIN_TOTAL_COST: "objective_weight_min_total_cost",
    vocab.OBJ_BANK_VALUE: "objective_weight_bank_value",
    vocab.OBJ_CLIENT_OUTCOME: "objective_weight_client_outcome",
}

# One-hot default weight vectors: running `objective_id=OBJ-0n` alone is that
# objective's own pure measure; a blend overrides the vector via the overlay below.
DEFAULT_WEIGHTS = {
    vocab.OBJ_NEW_MONEY: {vocab.OBJ_NEW_MONEY: 1.0},
    vocab.OBJ_MIN_COMMITMENT: {vocab.OBJ_MIN_COMMITMENT: 1.0},
    vocab.OBJ_MIN_TOTAL_COST: {vocab.OBJ_MIN_TOTAL_COST: 1.0},
    vocab.OBJ_BANK_VALUE: {vocab.OBJ_BANK_VALUE: 1.0},
    vocab.OBJ_CLIENT_OUTCOME: {vocab.OBJ_CLIENT_OUTCOME: 1.0},
}

ADJUSTMENT_SET_ID = "AS-06-OBJ-2026.09"

# §6.3's worked example: "Objective re-weight | The blend weights of §5.8 | Instalment
# relief 0.4 -> 0.6 for a quarter" -- one `set` adjustment per weight component,
# scoped to channel, composed at the same stack position so the five always move
# together as one declared blend.
OBJECTIVE_REWEIGHT = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-06-OBJ-2026-001", kind="objective_reweight",
        target="objective_weight_min_commitment", effect=AdjustmentEffect("set", 0.6),
        scope={"channel_code": 1}, stack_position=1, owner="Credit Committee",
        approval_reference="CC-2026-014", rationale="Branch channel weighted to instalment relief this quarter",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31),
    ),
    Adjustment(
        adjustment_id="ADJ-06-OBJ-2026-002", kind="objective_reweight",
        target="objective_weight_new_money", effect=AdjustmentEffect("set", 0.4),
        scope={"channel_code": 1}, stack_position=1, owner="Credit Committee",
        approval_reference="CC-2026-014", rationale="Branch channel weighted to instalment relief this quarter",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31),
    ),
])


@dataclass(frozen=True)
class ScenarioMeasures:
    scenario_id: int
    new_money: float
    instalment_relief_ratio: float          # (baseline_instalment - new_total_commitment) / baseline_instalment
    total_cost_increase_ratio: float        # (new_total_cost - baseline_remaining_cost) / baseline_remaining_cost
    bank_value_ratio: float                 # bank_expected_value / settlement_total
    accounts_exited_proportion: float
    weighted_rate_reduction: float          # baseline_rate - new_weighted_rate (fraction, e.g. 0.04 = 4pp)
    providers_exited_proportion: float
    requested_amount: float


def _client_outcome_score(m: ScenarioMeasures, weights: dict[str, float] | None = None) -> float:
    w = weights or {"relief": 0.35, "cost": 0.25, "exited": 0.15, "rate": 0.15, "providers": 0.10}
    return (
        w["relief"] * m.instalment_relief_ratio
        - w["cost"] * m.total_cost_increase_ratio
        + w["exited"] * m.accounts_exited_proportion
        + w["rate"] * m.weighted_rate_reduction
        + w["providers"] * m.providers_exited_proportion
    )


def component_scores(m: ScenarioMeasures) -> dict[str, float]:
    """One normalised (ratio-to-baseline) score per OBJ-01..05, always computed -- OBJ-05's
    is the "shadow" result §5.8 requirement 8 needs even when a different objective is in
    force."""
    new_money_ratio = (m.new_money / m.requested_amount) if m.requested_amount else 0.0
    return {
        vocab.OBJ_NEW_MONEY: new_money_ratio,
        vocab.OBJ_MIN_COMMITMENT: m.instalment_relief_ratio,
        vocab.OBJ_MIN_TOTAL_COST: -m.total_cost_increase_ratio,
        vocab.OBJ_BANK_VALUE: m.bank_value_ratio,
        vocab.OBJ_CLIENT_OUTCOME: _client_outcome_score(m),
    }


def resolve_weights(
    objective_id: str, channel_code: int, decision_date: date, stack_enabled: bool = True,
) -> tuple[dict[str, float], list[str]]:
    """§5.8 requirement 1: the objective in force is a parameter; requirement 2: the
    blend weight vector is itself an overlay target. Returns the normalised weight
    vector and the overlay ids that fired."""
    base = dict(DEFAULT_WEIGHTS[objective_id])
    for obj_id, target in OBJECTIVE_WEIGHT_TARGETS.items():
        base.setdefault(obj_id, 0.0)
    applied_ids: list[str] = []
    for obj_id, target in OBJECTIVE_WEIGHT_TARGETS.items():
        result = OBJECTIVE_REWEIGHT.apply_stack(
            target, base[obj_id], {"channel_code": channel_code}, decision_date, ADJUSTMENT_SET_ID,
            stack_enabled=stack_enabled,
        )
        base[obj_id] = result.adjusted_value
        applied_ids.extend(result.adjustments_applied)
    total = sum(base.values()) or 1.0
    normalised = {k: v / total for k, v in base.items()}
    return normalised, sorted(set(applied_ids))


def objective_score(m: ScenarioMeasures, weights: dict[str, float]) -> float:
    scores = component_scores(m)
    return round(sum(weights.get(obj_id, 0.0) * scores.get(obj_id, 0.0) for obj_id in weights), 6)


@dataclass(frozen=True)
class RankedScenario:
    measures: ScenarioMeasures
    score: float
    component_scores: dict[str, float]
    settlement_set_key: tuple[frozenset, int]  # (settlement set, product_code) -- distinctness key


def rank_and_select(
    viable: list[ScenarioMeasures], weights: dict[str, float], settlement_sets: dict[int, frozenset],
    product_codes: dict[int, int],
) -> tuple[list[RankedScenario], bool]:
    """§5.8 requirements 4 (tie-break), 5 (indifference band), 6 (distinct top three).
    Returns the ranked, deduplicated list (winner first) and whether the winner and
    runner-up are within the indifference band."""
    ranked = [
        RankedScenario(m, objective_score(m, weights), component_scores(m),
                        (settlement_sets[m.scenario_id], product_codes[m.scenario_id]))
        for m in viable
    ]
    # Tie-break (requirement 4): highest score first; ties broken by lowest total cost
    # increase, then fewest accounts settled, then lowest scenario_id.
    ranked.sort(key=lambda r: (
        -r.score, r.measures.total_cost_increase_ratio, -r.measures.accounts_exited_proportion,
        r.measures.scenario_id,
    ))
    distinct: list[RankedScenario] = []
    seen_keys: set = set()
    for r in ranked:
        if r.settlement_set_key in seen_keys:
            continue
        seen_keys.add(r.settlement_set_key)
        distinct.append(r)
        if len(distinct) == 3:
            break
    indifferent = False
    if len(distinct) >= 2 and distinct[0].score != 0:
        indifferent = abs(distinct[0].score - distinct[1].score) / abs(distinct[0].score) < INDIFFERENCE_BAND
    return distinct, indifferent


def shadow_best(viable: list[ScenarioMeasures]) -> ScenarioMeasures | None:
    """§5.8 requirement 8: the best scenario under OBJ-05 (client outcome), recorded
    regardless of which objective is actually in force."""
    if not viable:
        return None
    return max(viable, key=lambda m: component_scores(m)[vocab.OBJ_CLIENT_OUTCOME])
