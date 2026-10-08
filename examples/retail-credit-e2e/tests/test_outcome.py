"""Reasons, final validation and the outcome."""
from decider import flow, step

from retail_credit import validation


def test_the_ranked_reasons_put_the_most_severe_first_and_name_the_primary(decide):
    out = decide(is_sanctioned=True, applicant_age_years=17.0, declared_living_expenses=15_900.0)
    assert out["decline_reason_codes"] == [1006, 1001, 2140]
    assert (out["primary_reason_code"], out["reason_registry_version"]) == (1006, "reasons-2026.09")


def test_a_decline_without_an_offer_prices_nothing(decide):
    out = decide(is_sanctioned=True)
    assert (out["has_offer"], out["offer_amount"], out["instalment"], out["rate_cell_id"]) == (False, 0.0, 0.0, "")


def test_a_fraud_refer_leaves_an_indicative_offer_but_refers(decide):
    out = decide(identity_number_reused=True, synthetic_identity_score=0.9)
    assert out["has_offer"] is True and out["outcome_code"] == 3 and out["decline_reason_codes"] == [2002]


def test_a_watchlist_hit_declines_while_a_bureau_outage_outside_the_envelope_only_refers(decide):
    assert decide(internal_fraud_register_hit=True)["outcome_code"] == 4
    assert decide(bureau_available=False)["outcome_code"] == 3


def test_validation_lists_every_assertion_the_offer_fails():
    checks = flow(step(validation.revalidate_offer, outputs=("validation_passed", "validation_failed_assertions")),
                  name="validation").compile("interpreted")
    offer = {"has_offer": True, "offer_amount": 75_100.0, "offer_term_months": 60.0, "term_cap": 48.0,
             "amount_cap": 75_000.0, "nominal_annual_rate": 0.30, "instalment": 2_600.0, "instalment_ceiling": 2_500.0,
             "amount_grid": 250.0}
    out = checks.score(offer, {})
    assert (out["validation_passed"], out["validation_failed_assertions"]) == (False, [1701, 1702, 1703, 1704, 1705])
    sound = {**offer, "offer_amount": 75_000.0, "term_cap": 60.0, "nominal_annual_rate": 0.2, "instalment": 2_500.0}
    assert checks.score(sound, {})["validation_failed_assertions"] == []
    assert checks.score({**sound, "instalment": 2_500.02}, {})["validation_failed_assertions"] == [1705]  # a cent is tolerated
