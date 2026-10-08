"""A disputed event is never the sole cause of a decline."""
from datetime import date

from business_nested.records import entity, event


# A significant holder (15%): each R90 000 judgment is material, and six of them add up to R540 000 unsatisfied.
OWNER = entity(1, 5, 85.0)


def holder(disputed: int):
    return entity(2, 5, 15.0, [event(i, 90_000.0, event_date=date(2024, i, 1), is_disputed=i <= disputed) for i in range(1, 7)])


def test_a_decline_that_stands_without_the_disputed_events_is_still_a_decline(assess):
    out = assess([OWNER, holder(disputed=0)], requested_amount=800_000.0)
    assert (out["outcome_code"], out["decline_reason_codes"]) == (4, [5402])
    assert out["business_decline_from_disputed_event"] is False


def test_a_decline_resting_on_disputed_events_becomes_a_referral_that_names_the_dispute(assess):
    out = assess([OWNER, holder(disputed=2)], requested_amount=800_000.0)  # without them: R360 000, material not disqualifying
    assert out["business_decline_from_disputed_event"] is True
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [5404])
    assert (out["attributing_entity_id"], out["attributing_event_ids"]) == (2, [1, 2, 3, 4, 5, 6])
    assert out["assessed_entities"][1]["adverse_verdict_provisional"] is True


def test_a_disputed_event_does_not_save_a_business_that_is_declined_for_another_reason(assess, with_params):
    harsh = with_params(("business_credit", "decision", "outcome", "business_outcome"), decline_cutoff_grade=1)
    out = assess([OWNER, holder(disputed=2)], requested_amount=800_000.0, params_document=harsh)
    assert out["outcome_code"] == 4 and 5602 in out["decline_reason_codes"]


def test_a_single_disputed_disqualifying_event_is_not_disqualifying_in_the_first_place(assess):
    out = assess([entity(1, 5, 100.0, [event(1, 80_000.0, is_disputed=True)])])  # R80 000 on a critical entity
    entity_out = out["assessed_entities"][0]
    assert (entity_out["adverse_verdict_code"], entity_out["adverse_verdict_provisional"]) == (3, True)
    assert out["outcome_code"] != 4
