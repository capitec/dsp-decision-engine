"""Which settlement sets are generated, in what order, and under which product."""
from conftest import with_params

CARDS = {2, 4, 5, 7}


def sets(out, rule=None):
    seen = []
    for s in out["scenarios"]:
        if (rule is None or s["generating_rule"] == rule) and s["settlement_set"] not in seen:
            seen.append(s["settlement_set"])
    return seen


def with_rules(params, rules):
    return with_params(params, "search", ordering_rules=[{"rule": r} for r in rules])


def test_the_prefixes_of_the_highest_rate_ordering_are_the_first_candidates(assess):
    # rates: 3 at 29.9%, 1 at 29.5%, 6 at 28.9%, 4 at 24.9%
    assert sets(assess(), "H1_highest_rate_first")[:3] == [[1, 3], [1, 3, 6], [1, 3, 4, 6]]


def test_a_single_account_is_never_routed_to_the_consolidation_loan(assess):
    out = assess()
    assert all(len(s["settlement_set"]) >= 2 for s in out["scenarios"] if s["product_code"] == 11)


def test_the_balance_transfer_card_only_carries_sets_made_of_cards(assess):
    card_scenarios = [s for s in assess()["scenarios"] if s["product_code"] == 20]
    assert card_scenarios and all(set(s["settlement_set"]) <= CARDS for s in card_scenarios)


def test_a_set_of_cards_is_carried_by_both_products_and_each_is_priced_from_its_own_card(assess):
    by_product = {}
    for s in assess()["scenarios"]:
        if s["settlement_set"] == [2, 4, 7]:
            by_product.setdefault(s["product_code"], s)
    assert set(by_product) == {11, 20}
    assert by_product[11]["rate_cell_id"].startswith("rate_card.product11@")
    assert by_product[20]["rate_cell_id"].startswith("rate_card.card_promotional@")


def test_the_candidates_come_in_the_same_order_every_time(assess):
    first = [(s["scenario_id"], s["settlement_set"], s["product_code"], s["term_months"], s["promotional_months"])
             for s in assess()["scenarios"]]
    for _ in range(3):
        again = [(s["scenario_id"], s["settlement_set"], s["product_code"], s["term_months"], s["promotional_months"])
                 for s in assess()["scenarios"]]
        assert again == first


def test_no_set_is_generated_twice_and_ids_follow_generation_order(assess):
    out = assess()
    keys = [(tuple(s["settlement_set"]), s["product_code"], s["term_months"], s["promotional_months"]) for s in out["scenarios"]]
    assert len(set(keys)) == len(keys)
    assert [s["scenario_id"] for s in out["scenarios"]] == list(range(1, len(keys) + 1))


def test_the_full_settleable_set_is_priced_so_it_can_be_shown_to_have_been(assess):
    out = assess()
    full = [s for s in out["scenarios"] if s["settlement_set"] == out["settleable_account_refs"]]
    assert full and full[0]["generating_rule"] in {"full_settleable_set", "H1_highest_rate_first", "H2_most_relief_per_rand"}


def test_a_retired_ordering_rule_produces_no_candidates(assess, params):
    out = assess(params_document=with_rules(params, [1]))
    assert {s["generating_rule"] for s in out["scenarios"]} <= {"H1_highest_rate_first", "full_settleable_set"}
    assert out["search_record"]["ordering_rules"] == ["H1_highest_rate_first"]


def test_reordering_the_rules_reorders_the_candidates(assess, params):
    first = assess(params_document=with_rules(params, [8, 1]))["scenarios"][0]
    assert first["generating_rule"] == "H8_highest_reaccumulation_risk_first"
    assert all(a in CARDS for a in first["settlement_set"])


def test_the_nominated_set_is_a_candidate_and_every_set_contains_it(assess):
    out = assess(client_nominated_settle=[5])
    assert out["scenarios"][0]["generating_rule"] == "client_nominated"
    assert out["scenarios"][0]["settlement_set"] == [5]
    assert all(5 in s["settlement_set"] for s in out["scenarios"])


def test_a_provider_the_client_is_in_arrears_with_comes_first_under_h5(assess, params):
    rules = with_rules(params, [5])
    first = assess(params_document=rules)["scenarios"][0]
    assert first["generating_rule"] == "H5_provider_in_arrears_first"
    assert 3 in first["settlement_set"]  # provider 105 is the one the client is in arrears with


def test_a_held_quotation_comes_first_under_h6_not_the_highest_rate(assess, params):
    ordered = [s for s in assess(params_document=with_rules(params, [6]))["scenarios"] if s["generating_rule"].startswith("H6")]
    assert 3 in ordered[0]["settlement_set"]  # the only account with a quotation in hand
