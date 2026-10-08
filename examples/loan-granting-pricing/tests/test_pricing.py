"""Pricing one candidate: the rate cell, the capitalised fee, the premium on the amount financed, the instalment."""
from loan_granting.pricing import INF, CreditLifeIndex, PriceEvaluator, RateCardIndex


def card(*bands, grade=9):
    return RateCardIndex([{"amt_lo": -INF if i == 0 else edge, "term_lo": -INF, "grade": grade, "rate": rate,
                           "cell_id": f"cell-{i}", "rate_card_version": "v1"} for i, (edge, rate) in enumerate(bands)])


def evaluator(index, ceiling=0.32, own_credit_life=False, **caps):
    return PriceEvaluator(index, CreditLifeIndex(), ceiling, own_credit_life, **caps)


def annuity(principal, annual_rate, months):
    monthly = annual_rate / 12
    return principal * monthly / (1 - (1 + monthly) ** -months)


INDEX = card((0, 19.75), (50_000.0, 18.25))


def test_the_instalment_is_the_annuity_on_the_amount_financed_plus_service_fee_and_premium():
    priced = evaluator(INDEX).evaluate(50_000.0, 60, 9, 38.0, False)
    financed = 50_000.0 + 5_050.0  # the initiation fee is capitalised: R1 050 plus 10% over R10 000
    assert (priced.initiation_fee, priced.amount_financed) == (5_050.0, financed)
    assert priced.credit_life_premium == 308.28  # R5.60 per R1 000 of the amount financed, not of the advance
    assert priced.instalment == round(annuity(financed, 0.1825, 60) + 69.0 + 308.28, 2) == 1_782.69
    assert (priced.rate_cell_id, priced.rate_card_version) == ("cell-1", "v1")
    assert priced.total_cost_of_credit == round(1_782.69 * 60 + 5_050.0, 2)


def test_the_rate_steps_down_at_the_band_edge_so_a_smaller_amount_can_cost_more():
    ev = evaluator(INDEX, own_credit_life=True)
    assert ev.evaluate(49_900.0, 60, 9, 38.0, False).instalment > ev.evaluate(50_000.0, 60, 9, 38.0, False).instalment


def test_a_declared_own_credit_life_policy_removes_the_premium():
    with_cover = evaluator(INDEX).evaluate(50_000.0, 60, 9, 38.0, False)
    own_cover = evaluator(INDEX, own_credit_life=True).evaluate(50_000.0, 60, 9, 38.0, False)
    assert own_cover.credit_life_premium == 0.0
    assert round(with_cover.instalment - own_cover.instalment, 2) == 308.28


def test_a_joint_application_pays_the_joint_premium():
    single = evaluator(INDEX).evaluate(50_000.0, 60, 9, 38.0, False)
    joint = evaluator(INDEX).evaluate(50_000.0, 60, 9, 38.0, True)
    assert joint.credit_life_premium > single.credit_life_premium


def test_the_initiation_fee_stops_at_the_cap_in_force():
    big = evaluator(INDEX).evaluate(200_000.0, 60, 9, 38.0, False)
    assert (big.initiation_fee, big.initiation_fee_capped) == (6_500.0, True)
    older_cap = evaluator(INDEX, initiation_fee_cap=6_200.0).evaluate(200_000.0, 60, 9, 38.0, False)
    assert older_cap.initiation_fee == 6_200.0


def test_a_cell_above_the_statutory_ceiling_is_not_priced():
    result = evaluator(INDEX, ceiling=0.18).evaluate(50_000.0, 60, 9, 38.0, False)
    assert (result.priced, result.instalment, result.nominal_annual_rate) == (False, 0.0, 0.1825)
    assert evaluator(INDEX, ceiling=0.18).evaluate(10_000.0, 60, 9, 38.0, False).priced is False


def test_a_grade_the_card_does_not_price_is_not_priced():
    assert evaluator(INDEX).evaluate(50_000.0, 60, 4, 38.0, False).priced is False


def test_the_rate_add_on_moves_the_rate_after_the_lookup_and_is_checked_against_the_ceiling():
    ev = evaluator(INDEX, ceiling=0.19)
    assert ev.evaluate(50_000.0, 60, 9, 38.0, False, rate_addon_bps=50).nominal_annual_rate == 0.1875
    assert ev.evaluate(50_000.0, 60, 9, 38.0, False, rate_addon_bps=100).priced is False


def test_a_lookup_returns_the_cell_of_the_band_the_amount_falls_in():
    index = card((0, 19.75), (50_000.0, 18.25))
    assert index.lookup(49_999.0, 60, 9) == (19.75, "cell-0")
    assert index.lookup(50_000.0, 60, 9) == (18.25, "cell-1")


def test_the_credit_life_table_is_looked_up_by_age_term_and_cover_type():
    index = CreditLifeIndex()
    assert index.lookup(38.0, 60, False)[0] == 5.60
    assert index.lookup(25.0, 12, False)[0] == 3.20
    assert index.lookup(38.0, 60, True)[0] == 9.40
