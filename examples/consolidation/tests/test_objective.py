"""The objective is configuration: weights per channel, an overlay that re-weights, a ranking independent of the budget."""
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from consolidation import objective

OUTCOME = {"relief": 0.35, "cost": 0.25, "exited": 0.15, "rate": 0.15, "providers": 0.10}
TABLE = SimpleNamespace(channel_code=np.array([0, 1]), new_money=np.array([0.0, 0.0]), min_commitment=np.array([1.0, 1.0]),
                        min_total_cost=np.array([0.0, 0.0]), bank_value=np.array([0.0, 0.0]), client_outcome=np.array([0.0, 0.0]))


def scenario(scenario_id, relief=0.1, cost=0.05, accounts=(1,), product=11, total_cost=1_000.0, **measures):
    base = {"new_money_ratio": 0.0, "instalment_relief_ratio": relief, "total_cost_increase_ratio": cost, "bank_value_ratio": 0.0,
            "accounts_exited_proportion": 0.3, "weighted_rate_reduction": 0.02, "providers_exited_proportion": 0.3}
    return {"scenario_id": scenario_id, "viable": True, "settlement_set": list(accounts), "product_code": product,
            "total_cost_of_credit": total_cost, "measures": {**base, **measures}}


def weights(**given):
    return {name: given.get(name, 0.0) for name in objective.OBJECTIVES}


def rank(scenarios, w=None, band=0.02):
    return objective.rank(scenarios, w or weights(min_commitment=1.0), OUTCOME, band)


def test_the_default_weights_are_one_objective_and_a_channel_without_a_row_uses_the_default(assess):
    out = assess(channel_code=3)
    assert out["objective_weights"] == {"new_money": 0.0, "min_commitment": 1.0, "min_total_cost": 0.0, "bank_value": 0.0,
                                        "client_outcome": 0.0}
    assert out["objective_overlay_ids"] == []


def test_each_channel_has_its_own_objective(assess):
    assert assess(channel_code=2)["objective_weights"]["client_outcome"] == 1.0
    assert assess(channel_code=4)["objective_weights"]["new_money"] == 1.0


def test_the_branch_reweight_overlay_blends_relief_and_new_money_and_names_itself(assess):
    out = assess(channel_code=1)
    assert out["objective_weights"] == {"new_money": 0.4, "min_commitment": 0.6, "min_total_cost": 0.0, "bank_value": 0.0,
                                        "client_outcome": 0.0}
    assert out["objective_overlay_ids"] == ["ADJ-CON-OBJ-2026-001", "ADJ-CON-OBJ-2026-002"]


def test_the_overlay_does_not_apply_outside_its_channel_or_before_it_takes_effect():
    assert objective.weights_in_force(TABLE, 3, date(2026, 9, 24), True)[1] == []
    assert objective.weights_in_force(TABLE, 1, date(2025, 6, 1), True)[1] == []
    assert objective.weights_in_force(TABLE, 1, date(2026, 9, 24), True)[1] == ["ADJ-CON-OBJ-2026-001", "ADJ-CON-OBJ-2026-002"]


def test_with_the_stack_off_the_branch_gets_the_unblended_objective(assess, params):
    out = assess(params_document={**params, "shared": {"overlay_stack_enabled": False}})
    assert out["objective_weights"]["min_commitment"] == 1.0 and out["objective_overlay_ids"] == []


def test_a_weights_table_row_edited_in_the_params_changes_the_objective(assess, params, with_params):
    table = [{"channel_code": 0, "new_money": 0.0, "min_commitment": 0.0, "min_total_cost": 1.0, "bank_value": 0.0,
              "client_outcome": 0.0}]
    out = assess(channel_code=3, params_document=with_params(params, "selection", objective_weights=table))
    assert out["objective_weights"]["min_total_cost"] == 1.0


def test_the_scenario_with_the_most_relief_wins_under_the_relief_objective():
    ranked = rank([scenario(1, relief=0.1, accounts=(1,)), scenario(2, relief=0.5, accounts=(2,))])
    assert [r["scenario_id"] for r in ranked["top"]] == [2, 1]


def test_an_equal_score_goes_to_the_lowest_total_cost_then_the_fewest_accounts_then_the_lowest_id():
    cheaper = rank([scenario(1, total_cost=900.0, accounts=(1, 2)), scenario(2, total_cost=800.0, accounts=(3, 4))])
    assert cheaper["top"][0]["scenario_id"] == 2
    fewer = rank([scenario(1, accounts=(1, 2, 3)), scenario(2, accounts=(4,))])
    assert fewer["top"][0]["scenario_id"] == 2
    lowest = rank([scenario(5, accounts=(1,)), scenario(3, accounts=(2,))])
    assert lowest["top"][0]["scenario_id"] == 3


def test_the_three_shown_are_distinct_by_settlement_set_and_product():
    ranked = rank([scenario(1, relief=0.30, accounts=(1, 2)), scenario(2, relief=0.29, accounts=(1, 2)),
                   scenario(3, relief=0.28, accounts=(1, 2), product=20), scenario(4, relief=0.27, accounts=(3,)),
                   scenario(5, relief=0.26, accounts=(4,))])
    assert [r["scenario_id"] for r in ranked["top"]] == [1, 3, 4]


def test_a_rejected_scenario_is_never_a_runner_up():
    rejected = {**scenario(1, relief=0.9), "viable": False}
    ranked = rank([rejected, scenario(2, relief=0.1, accounts=(2,))])
    assert [r["scenario_id"] for r in ranked["top"]] == [2]


def test_scores_within_the_indifference_band_are_flagged_as_materially_equivalent():
    close = rank([scenario(1, relief=0.500, accounts=(1,)), scenario(2, relief=0.495, accounts=(2,))])
    apart = rank([scenario(1, relief=0.500, accounts=(1,)), scenario(2, relief=0.400, accounts=(2,))])
    assert close["indifferent"] is True and apart["indifferent"] is False


def test_the_winner_is_explained_by_the_component_that_accounts_for_the_gap():
    ranked = rank([scenario(1, relief=0.6, accounts=(1,)), scenario(2, relief=0.2, accounts=(2,))],
                  weights(min_commitment=0.6, min_total_cost=0.4))
    gap = ranked["versus"][0]
    assert (gap["winner"], gap["runner_up"], gap["decisive_component"]) == (1, 2, "min_commitment")
    assert gap["score_difference"] == pytest.approx(0.6 * 0.4, abs=1e-6)


def test_the_shadow_result_is_the_best_for_the_client_whatever_objective_is_in_force():
    ranked = rank([scenario(1, relief=0.9, cost=0.9, accounts=(1,)),  # best relief, harms the client
                   scenario(2, relief=0.2, cost=0.0, accounts=(2,), accounts_exited_proportion=0.9)])
    assert ranked["top"][0]["scenario_id"] == 1 and ranked["shadow"]["scenario_id"] == 2


def test_components_are_ratios_to_the_baseline_so_the_budget_does_not_change_the_winner(assess, params, with_params):
    full = assess()
    cut = assess(params_document=with_params(params, "search", max_scenarios=full["chosen_offer"]["scenario_id"]))
    assert cut["chosen_offer"]["scenario_id"] == full["chosen_offer"]["scenario_id"]
    assert cut["chosen_offer"]["objective_score"] == full["chosen_offer"]["objective_score"]


def test_the_sample_records_the_gap_to_each_runner_up(assess):
    out = assess()
    assert [v["runner_up"] for v in out["winner_vs_runners_up"]] == [r["scenario_id"] for r in out["runners_up"]]
    assert all(v["score_difference"] > 0 for v in out["winner_vs_runners_up"])
    assert out["shadow_best_scenario"]["scenario_id"] > 0
