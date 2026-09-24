from datetime import date

from consolidation import objective, vocab


def _measures(scenario_id, **kw):
    base = dict(
        new_money=0.0, instalment_relief_ratio=0.1, total_cost_increase_ratio=0.05, bank_value_ratio=0.02,
        accounts_exited_proportion=0.3, weighted_rate_reduction=0.02, providers_exited_proportion=0.3,
        requested_amount=10_000.0,
    )
    base.update(kw)
    return objective.ScenarioMeasures(scenario_id=scenario_id, **base)


def test_resolve_weights_default_objective_is_one_hot():
    weights, overlays = objective.resolve_weights(vocab.OBJ_MIN_TOTAL_COST, channel_code=9, decision_date=date(2026, 9, 24))
    assert weights[vocab.OBJ_MIN_TOTAL_COST] == 1.0
    assert weights[vocab.OBJ_NEW_MONEY] == 0.0
    assert overlays == []


def test_reweight_overlay_blends_new_money_and_min_commitment_on_branch_channel():
    weights, overlays = objective.resolve_weights(vocab.OBJ_MIN_COMMITMENT, channel_code=1, decision_date=date(2026, 9, 24))
    assert weights[vocab.OBJ_MIN_COMMITMENT] == 0.6
    assert weights[vocab.OBJ_NEW_MONEY] == 0.4
    assert set(overlays) == {"ADJ-06-OBJ-2026-001", "ADJ-06-OBJ-2026-002"}


def test_reweight_overlay_does_not_apply_outside_its_scoped_channel():
    weights, overlays = objective.resolve_weights(vocab.OBJ_MIN_COMMITMENT, channel_code=3, decision_date=date(2026, 9, 24))
    assert weights[vocab.OBJ_MIN_COMMITMENT] == 1.0
    assert overlays == []


def test_higher_instalment_relief_wins_under_min_commitment_objective():
    weak = _measures(1, instalment_relief_ratio=0.1)
    strong = _measures(2, instalment_relief_ratio=0.5)
    weights = {vocab.OBJ_MIN_COMMITMENT: 1.0}
    settlement_sets = {1: frozenset({1}), 2: frozenset({2})}
    products = {1: 11, 2: 11}
    ranked, _ = objective.rank_and_select([weak, strong], weights, settlement_sets, products)
    assert ranked[0].measures.scenario_id == 2


def test_top_three_are_distinct_by_settlement_set_and_product():
    same_set_twice = [
        _measures(1, instalment_relief_ratio=0.3),
        _measures(2, instalment_relief_ratio=0.29),  # same (set, product) as #1's -- must be excluded
        _measures(3, instalment_relief_ratio=0.28),
    ]
    settlement_sets = {1: frozenset({1, 2}), 2: frozenset({1, 2}), 3: frozenset({3})}
    products = {1: 11, 2: 11, 3: 20}
    weights = {vocab.OBJ_MIN_COMMITMENT: 1.0}
    ranked, _ = objective.rank_and_select(same_set_twice, weights, settlement_sets, products)
    assert [r.measures.scenario_id for r in ranked] == [1, 3]


def test_indifference_band_flags_near_ties():
    a = _measures(1, instalment_relief_ratio=0.500)
    b = _measures(2, instalment_relief_ratio=0.495)
    settlement_sets = {1: frozenset({1}), 2: frozenset({2})}
    products = {1: 11, 2: 20}
    weights = {vocab.OBJ_MIN_COMMITMENT: 1.0}
    _, indifferent = objective.rank_and_select([a, b], weights, settlement_sets, products)
    assert indifferent is True


def test_shadow_best_is_always_the_top_client_outcome_scenario_regardless_of_objective_in_force():
    weak_outcome = _measures(1, instalment_relief_ratio=0.9, total_cost_increase_ratio=0.9)  # great OBJ-02, bad OBJ-05
    good_outcome = _measures(2, instalment_relief_ratio=0.2, total_cost_increase_ratio=0.0, accounts_exited_proportion=0.9)
    shadow = objective.shadow_best([weak_outcome, good_outcome])
    assert shadow.scenario_id == 2
