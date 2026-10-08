"""The people component: weights, the log-odds blend, the caps and the worst-entity override."""
import math

import pytest

from business_nested import blend, vocab
from business_nested.records import entity, event

WEIGHT_POLICY = dict(inclusion_floor_pct=5.0, control_ownership_pct=50.0, control_weight_floor=0.6,
                     controller_points=10.0, controller_cap_points=30.0)


def person(role=vocab.SHAREHOLDER, pct=0.0, controlling=False, owner=None):
    """What the weights read of an assessed entity."""
    return {"relationship_type_code": role, "effective_ownership_pct": pct, "is_controlling": controlling,
            "is_owner": role in vocab.OWNING_ROLES if owner is None else owner}


def weights(*people):
    return blend.blend_weights(list(people), **WEIGHT_POLICY)


def test_owners_are_weighted_by_their_stake():
    assert weights(person(pct=60.0), person(pct=40.0))[:2] == pytest.approx([0.6, 0.4])


def test_weights_always_sum_to_one():
    assert sum(weights(person(pct=30.0), person(pct=20.0), person(vocab.DIRECTOR), person(vocab.TRUSTEE))) == pytest.approx(1.0)


def test_the_person_who_controls_carries_at_least_sixty_percent_and_the_rest_share_what_is_left():
    result = weights(person(pct=32.0), person(vocab.DIRECTOR, 55.0, controlling=True, owner=True), person(pct=13.0))
    assert result == pytest.approx([0.2844444, 0.6, 0.1155556])


def test_a_majority_owner_without_disclosed_control_is_treated_as_in_control():
    assert weights(person(pct=50.0), person(pct=30.0))[0] == pytest.approx(0.6)


def test_a_controlling_owner_of_ninety_percent_keeps_their_stake_as_weight():
    assert weights(person(pct=90.0, controlling=True), person(pct=10.0))[0] == pytest.approx(0.9)


def test_a_lone_controller_has_all_the_weight():
    assert weights(person(vocab.DIRECTOR, 0.0, controlling=True)) == [1.0]


def test_directors_and_trustees_carry_ten_notional_points_each_capped_at_thirty_in_all():
    result = weights(person(pct=70.0), *[person(vocab.DIRECTOR) for _ in range(4)])
    raw = [0.7, 0.1, 0.1, 0.1, 0.0]
    assert result == pytest.approx([w / sum(raw) for w in raw])


@pytest.mark.parametrize("role", sorted(vocab.OUTSIDE_THE_BLEND_ROLES))
def test_sureties_guarantors_group_companies_and_signatories_are_outside_the_blend(role):
    assert weights(person(pct=100.0), person(role, 0.0))[1] == 0.0


def test_a_small_holder_below_the_inclusion_floor_is_outside_the_blend():
    assert weights(person(pct=95.0), person(pct=4.9))[1] == 0.0
    assert weights(person(pct=95.0), person(pct=5.0))[1] > 0.0


def test_the_blend_is_on_the_log_odds_scale_not_an_average_of_probabilities():
    blended = blend.log_odds_blend([0.01, 0.5], [0.5, 0.5])
    assert blended == pytest.approx(1 / (1 + math.exp(-0.5 * (math.log(0.01 / 0.99) + 0))))
    assert blended != pytest.approx(0.255)


def test_with_nobody_in_the_blend_the_worst_probability_stands():
    assert blend.log_odds_blend([0.1, 0.4], [0.0, 0.0]) == 0.4


def test_the_people_pd_is_the_blend_of_the_entities_pds_at_their_weights(assess):
    out = assess()
    assert out["people_weights"] == pytest.approx([0.2844444, 0.6, 0.1155556])
    entities = out["assessed_entities"]
    expected = blend.log_odds_blend([e["entity_pd"] for e in entities], out["people_weights"])
    assert out["people_pd"] == pytest.approx(expected) and out["people_pd"] == pytest.approx(0.1112389, abs=1e-6)
    assert out["people_pd_unadjusted"] < out["people_pd"]
    assert out["people_pd_overlay_contribution"] == pytest.approx(out["people_pd"] - out["people_pd_unadjusted"])


def test_the_people_grade_is_graded_from_the_blended_probability(assess):
    out = assess()  # a probability of 11.1% is grade 6 on the people ladder (9% to 13%)
    assert (out["people_grade_blended"], out["people_grade"]) == (6, 6)


def test_a_weak_holder_of_a_fifth_of_the_business_keeps_the_people_grade_at_ten_or_worse(assess):
    weak = entity(1, 5, 20.0, bureau_score=400.0, months_on_record=6.0, worst_delinquency_months=9.0)
    strong = entity(2, 5, 80.0)
    out = assess([weak, strong])
    assert out["assessed_entities"][0]["entity_grade"] >= 11
    assert out["people_grade_blended"] < 10 and out["people_grade"] == 10
    assert out["people_cap_rule"] == blend.PP_06_WEAK_LARGE_HOLDER


def test_a_critical_entity_with_a_material_record_keeps_the_people_grade_at_five_or_worse(assess):
    owner = entity(1, 5, 100.0, [event(1, 12_000.0)], bureau_score=800.0, months_on_record=200.0)  # material at R10 000
    clean = assess([entity(1, 5, 100.0, bureau_score=800.0, months_on_record=200.0)])
    capped = assess([owner])
    assert clean["people_grade"] < 5 and clean["people_cap_rule"] == ""
    assert (capped["people_grade"], capped["people_cap_rule"]) == (5, blend.PP_06_CRITICAL_MATERIAL)


def test_an_entity_outside_the_blend_still_declines_the_business(assess):
    fraud = event(7, None, event_type_code=12)  # a fraud conviction disqualifies at any size
    small = entity(3, vocab.SHAREHOLDER, 2.0, [fraud])
    out = assess([entity(1, 5, 98.0), small])
    assert out["people_weights"][1] == 0.0
    assert (out["business_decline_from_entity"], out["business_decline_entity_id"]) == (True, 3)


def test_the_first_disqualifying_entity_by_id_is_the_one_named(assess):
    fraud = event(7, None, event_type_code=12)
    out = assess([entity(5, 5, 50.0, [fraud]), entity(2, 5, 50.0, [event(8, None, event_type_code=12)])])
    assert out["business_decline_entity_id"] == 2


def test_ownership_below_the_inclusion_floor_does_not_count_towards_coverage(assess, with_params):
    holders = [entity(1, vocab.SHAREHOLDER, 70.0)] + [entity(i, vocab.SHAREHOLDER, 3.0) for i in range(2, 12)]
    out = assess(holders)
    assert out["insufficient_people_coverage"] is True  # 30 of 100 percent belongs to holders under the 5% floor
    assert out["outcome_code"] == 3 and 5401 in out["decline_reason_codes"]
    loosened = with_params(("business_credit", "decision", "people", "people_blend"), inclusion_floor_pct=2.0)
    assert assess(holders, params_document=loosened)["insufficient_people_coverage"] is False


def test_a_business_with_no_natural_person_in_the_blend_is_a_referral(assess):
    company = entity(1, vocab.SHAREHOLDER, 100.0, is_natural_person=False)
    out = assess([company])
    assert out["no_scoreable_natural_person"] is True
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [5403])
    assert assess([entity(1, vocab.SHAREHOLDER, 100.0)])["no_scoreable_natural_person"] is False
