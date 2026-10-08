"""The decision store, what the Bank knew when, and the split of a probability movement into causes."""
from datetime import date

import pytest

from business_credit_e2e import history, vocab

PREVIOUS = {"risk_grade": 6, "master_scale_version": "v1", "probability_of_default": 0.0625,
            "probability_of_default_unadjusted": 0.05}


def director_resigned_in_march_recorded_in_november():
    return [
        history.entity_fact(7, "director", {"role": "director"}, date(2020, 1, 1), date(2028, 3, 1), date(2020, 1, 5)),
        history.entity_fact(7, "director", {"role": "resigned"}, date(2028, 3, 1), None, date(2028, 11, 1)),
    ]


def test_reproducing_the_june_review_sees_the_director_the_bank_knew_then():
    (fact,) = history.query_knowledge(director_resigned_in_march_recorded_in_november(), date(2028, 6, 30))
    assert fact["value"]["role"] == "director"


def test_a_correction_appears_in_knowledge_views_only_from_the_day_it_was_learned():
    facts = director_resigned_in_march_recorded_in_november()
    assert history.query_knowledge(facts, date(2028, 10, 31))[0]["value"]["role"] == "director"
    assert history.query_knowledge(facts, date(2028, 11, 1))[0]["value"]["role"] == "resigned"


def test_the_effective_view_sees_the_resignation_from_the_day_it_happened():
    facts = director_resigned_in_march_recorded_in_november()
    assert [f["value"]["role"] for f in history.query_effective(facts, date(2028, 2, 28))] == ["director"]
    assert [f["value"]["role"] for f in history.query_effective(facts, date(2028, 3, 31))] == ["resigned"]


def test_the_store_numbers_decisions_in_order_and_returns_the_latest_per_facility():
    store = history.DecisionStore()
    first = store.append({"facility_id": 1})
    store.append({"facility_id": 2})
    third = store.append({"facility_id": 1})
    assert (first["decision_of_record_id"], third["decision_of_record_id"]) == (1, 3)
    assert store.latest(1) == third and store.history(1) == [first, third]
    assert store.latest(99) is None


def current(**changes):
    return {**PREVIOUS, "risk_grade": 8, "probability_of_default": 0.12, "probability_of_default_unadjusted": 0.10, **changes}


def test_each_cause_is_the_step_it_was_measured_in():
    migration = history.grade_migration(PREVIOUS, current(), previous_rescored=0.06, business_only=0.07, with_structure=0.09)
    assert migration["causes"] == pytest.approx({
        "model": 0.01, "business_data": 0.01, "entity_structure": 0.02, "entity_data": 0.01, "overlay": 0.0075})
    assert sum(migration["causes"].values()) == pytest.approx(migration["observed_pd_movement"])


def test_a_change_of_master_scale_makes_two_grades_not_comparable():
    same = history.grade_migration(PREVIOUS, current(), 0.05, 0.05, 0.05)
    other = history.grade_migration(PREVIOUS, current(master_scale_version="v2"), 0.05, 0.05, 0.05)
    assert (same["comparison_basis_code"], other["comparison_basis_code"]) == (
        vocab.COMPARISON_AS_GRADED, vocab.COMPARISON_NOT_COMPARABLE)
