from datetime import date

from consolidation import interventions, vocab


def _facts(**kw):
    base = dict(
        product_code=vocab.PRODUCT_FLEX_CONSOLIDATION, settled_account_count=3, settled_weighted_rate=0.25,
        new_rate=0.20, instalment_relief_pct=0.2, total_cost_increase_pct=0.05, term_extension_months=0,
        new_money=5000.0, settlement_total=50000.0, new_dsr=0.30, discretionary_income_after=1500.0,
        external_proportion=0.8, consolidations_last_24_months=0,
    )
    base.update(kw)
    return interventions.ScenarioFacts(**base)


def test_all_pass_when_every_measure_is_within_threshold():
    results = interventions.evaluate_interventions(_facts(), date(2026, 9, 24))
    assert interventions.failed(results) == []


def test_anti_harm_ceiling_rejects_when_exceeded():
    results = interventions.evaluate_interventions(_facts(total_cost_increase_pct=0.40), date(2026, 9, 24))
    codes = {r.code for r in interventions.failed(results)}
    assert interventions.CON_INT_04 in codes


def test_anti_harm_overlay_tightens_the_threshold_on_product_11_only():
    facts = _facts(total_cost_increase_pct=0.13)  # passes the base 15% but not a tightened 12%
    base = interventions.evaluate_interventions(facts, date(2025, 6, 1))  # before the overlay's effective_from
    assert interventions.failed(base) == []

    overlaid = interventions.evaluate_interventions(facts, date(2026, 6, 1))
    anti_harm = next(r for r in overlaid if r.code == interventions.CON_INT_04)
    assert anti_harm.status == interventions.FAILED
    assert anti_harm.overlay_id == "ADJ-06-2026-001"


def test_overlay_does_not_apply_to_product_20():
    facts = _facts(product_code=vocab.PRODUCT_BALANCE_TRANSFER, total_cost_increase_pct=0.13)
    results = interventions.evaluate_interventions(facts, date(2026, 6, 1))
    assert interventions.failed(results) == []


def test_stack_disabled_reverts_to_base_threshold():
    facts = _facts(total_cost_increase_pct=0.13)
    results = interventions.evaluate_interventions(facts, date(2026, 6, 1), stack_enabled=False)
    assert interventions.failed(results) == []


def test_max_accounts_settled_rejects_over_the_cap():
    results = interventions.evaluate_interventions(_facts(settled_account_count=9), date(2026, 9, 24))
    codes = {r.code for r in interventions.failed(results)}
    assert interventions.CON_INT_01 in codes


def test_all_fourteen_interventions_are_evaluated_not_only_until_first_failure():
    results = interventions.evaluate_interventions(_facts(), date(2026, 9, 24))
    assert len(results) == 14
