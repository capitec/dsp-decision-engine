"""The outcome: every reason that applies, the cut-off grades, and the entity and events a decline names."""
from datetime import date

import pytest

from business_nested.documents import shipped
from business_nested.records import entity, event

OUTCOME_PARAMS = ("business_credit", "decision", "outcome", "business_outcome")


@pytest.fixture
def clean(sample):
    """The sample's entities without the judgment that declines it."""
    sample["entities"][0]["adverse_events"] = [e for e in sample["entities"][0]["adverse_events"] if e["event_id"] != 41]
    return sample["entities"]


def test_a_clean_business_in_a_good_grade_is_approved_with_no_reasons(assess, clean):
    out = assess(clean)
    assert (out["outcome_code"], out["decline_reason_codes"], out["primary_reason_code"]) == (1, [], None)
    assert (out["attributing_entity_id"], out["attributing_event_ids"], out["attributing_rule_id"]) == (None, [], "")


def test_the_sample_is_declined_for_the_judgment_on_the_first_entity(assess):
    out = assess()
    assert (out["outcome_code"], out["decline_reason_codes"], out["primary_reason_code"]) == (4, [5402], 5402)
    assert (out["attributing_entity_id"], out["attributing_event_ids"], out["attributing_rule_id"]) == (1, [41], "AE-R-01")
    assert out["reason_registry_version"] == shipped("reason_codes").version


def test_a_decline_names_every_event_that_satisfied_the_binding_rule(assess):
    judgments = [event(i, 90_000.0, event_date=date(2024, i, 1)) for i in (3, 1, 2, 4, 5, 6)]
    # A significant holder: R90 000 is material, not disqualifying, but six of them are R540 000 unsatisfied.
    out = assess([entity(1, 5, 85.0), entity(2, 5, 15.0, judgments)], requested_amount=800_000.0)
    assert out["outcome_code"] == 4 and out["attributing_entity_id"] == 2
    assert (out["attributing_event_ids"], out["attributing_rule_id"]) == ([1, 2, 3, 4, 5, 6], "AE-R-07")


def test_a_grade_at_the_decline_cut_off_is_declined_for_the_grade(assess, clean, with_params):
    out = assess(clean, params_document=with_params(OUTCOME_PARAMS, decline_cutoff_grade=7))
    assert (out["risk_grade"], out["outcome_code"], out["decline_reason_codes"]) == (7, 4, [5602])
    assert out["attributing_entity_id"] is None  # no entity is to blame for a grade


def test_a_grade_in_the_referral_band_is_referred(assess, clean, with_params):
    out = assess(clean, params_document=with_params(OUTCOME_PARAMS, refer_cutoff_grade=7))
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [5603])


def test_every_reason_that_applies_is_reported_most_severe_first(assess, with_params):
    out = assess(params_document=with_params(OUTCOME_PARAMS, decline_cutoff_grade=7))
    assert out["decline_reason_codes"] == [5402, 5602] and out["primary_reason_code"] == 5402


def test_a_referral_reason_alongside_a_decline_is_reported_but_the_decline_stands(assess):
    company = entity(2, 5, 100.0, is_natural_person=False)  # the only natural person is outside the blend
    out = assess([entity(1, 5, 0.0, [event(7, None, event_type_code=12)]), company])
    assert out["outcome_code"] == 4 and out["decline_reason_codes"] == [5402, 5403]


def test_the_cut_off_grades_are_params(assess, clean, with_params):
    assert assess(clean)["outcome_code"] == 1
    assert assess(clean, params_document=with_params(OUTCOME_PARAMS, refer_cutoff_grade=7))["outcome_code"] == 3
    assert assess(clean, params_document=with_params(OUTCOME_PARAMS, decline_cutoff_grade=7))["outcome_code"] == 4
