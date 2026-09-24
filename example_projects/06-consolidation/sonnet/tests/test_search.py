from consolidation import search, vocab


def _acc(ref, rate=0.2, instalment=100.0, settlement=1000.0, term=12, provider=1, internal=False, arrears=0, type_code=20):
    return search.SettleableAccount(
        account_ref=ref, settlement_amount=settlement, instalment=instalment, nominal_annual_rate=rate,
        remaining_term_months=term, provider_code=provider, is_internal=internal, months_in_arrears=arrears,
        account_type_code=type_code, is_mandatory=False,
    )


def test_empty_set_is_always_generated_first():
    accounts = [_acc(1), _acc(2)]
    candidates = search.generate_candidate_sets(accounts, [])
    assert candidates[0][0] == frozenset()


def test_full_settleable_set_is_generated():
    accounts = [_acc(1), _acc(2), _acc(3)]
    candidates = search.generate_candidate_sets(accounts, [])
    all_sets = {c[0] for c in candidates}
    assert frozenset({1, 2, 3}) in all_sets


def test_mandatory_account_is_present_in_every_non_empty_candidate():
    accounts = [_acc(1), _acc(2), _acc(3)]
    candidates = search.generate_candidate_sets(accounts, mandatory_refs=[2])
    for s, rule in candidates:
        if s:
            assert 2 in s


def test_ordering_is_deterministic_across_repeated_calls():
    accounts = [_acc(i, rate=0.1 * i) for i in range(1, 6)]
    first = search.generate_candidate_sets(accounts, [])
    second = search.generate_candidate_sets(accounts, [])
    assert first == second


def test_product_routing_card_only_set_gets_both_products():
    cards = [_acc(1, type_code=10), _acc(2, type_code=11)]
    assert set(search.route_products(cards)) == {vocab.PRODUCT_FLEX_CONSOLIDATION, vocab.PRODUCT_BALANCE_TRANSFER}


def test_product_routing_mixed_set_only_gets_flex_consolidation():
    mixed = [_acc(1, type_code=10), _acc(2, type_code=20)]
    assert search.route_products(mixed) == [vocab.PRODUCT_FLEX_CONSOLIDATION]


def test_budget_is_never_exceeded():
    accounts = [_acc(i) for i in range(1, 15)]
    scenarios, cause = search.generate_scenarios(accounts, [], max_term_extension=24, budget=10)
    assert len(scenarios) <= 10
    assert cause == vocab.TERMINATION_BUDGET
