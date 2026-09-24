"""Statefulness across time: the decision-history store, the bi-temporal
entity-structure query pair (spec 11 §5.13.2's own worked example), and grade
migration's six-cause decomposition (§5.4.1, §5.10.3)."""
from __future__ import annotations

from datetime import date

from business_credit_e2e import history, vocab


def _resignation_facts():
    """Spec 11 §5.13.2's own worked example: a director appointed long ago,
    resigned March 2028, the Bank learning of it in November 2028."""
    appointment = history.new_entity_fact(
        entity_id=7, fact_kind="director", effective_from=date(2020, 1, 1), effective_to=date(2028, 3, 1),
        known_from=date(2020, 1, 5), source_code="registry", value={"role": "director"},
    )
    resignation_recorded = history.new_entity_fact(
        entity_id=7, fact_kind="director", effective_from=date(2028, 3, 1), effective_to=None,
        known_from=date(2028, 11, 1), source_code="registry", value={"role": "resigned"},
    )
    return [appointment, resignation_recorded]


def test_reproducing_the_june_2028_review_sees_the_director_still_appointed():
    facts = _resignation_facts()
    known = history.query_knowledge(facts, date(2028, 6, 30))
    assert len(known) == 1
    assert known[0]["value"]["role"] == "director"  # the resignation was not yet known


def test_asking_who_was_director_in_march_2028_sees_the_resignation():
    facts = _resignation_facts()
    effective = history.query_effective(facts, date(2028, 3, 31))
    # The appointment fact's effective window ends exactly at 2028-03-01 (exclusive upper
    # bound), so as at 2028-03-31 only the resignation fact (open-ended) covers the date.
    assert len(effective) == 1
    assert effective[0]["value"]["role"] == "resigned"


def test_a_late_arriving_fact_does_not_appear_in_an_earlier_knowledge_view():
    facts = _resignation_facts()
    known_before_disclosure = history.query_knowledge(facts, date(2028, 10, 31))
    assert known_before_disclosure[0]["value"]["role"] == "director"
    known_after_disclosure = history.query_knowledge(facts, date(2028, 11, 2))
    assert known_after_disclosure[0]["value"]["role"] == "resigned"


def test_decision_history_store_is_append_only_and_ordered():
    store = history.DecisionHistoryStore()
    r1 = history.new_decision_of_record(
        facility_id=1, assessment_kind_code=vocab.EP1_NEW_TO_BANK, decision_date=date(2026, 1, 1),
        knowledge_date=date(2026, 1, 1), predecessor_id=None, comparison_basis_code=vocab.COMPARISON_ORIGINATION,
        outcome_code=1, risk_grade=6, master_scale_version="v1", probability_of_default=0.05,
    )
    store.append(r1)
    r2 = history.new_decision_of_record(
        facility_id=1, assessment_kind_code=vocab.EP3_ANNUAL_REVIEW, decision_date=date(2027, 1, 1),
        knowledge_date=date(2027, 1, 1), predecessor_id=r1["decision_of_record_id"],
        comparison_basis_code=vocab.COMPARISON_AS_GRADED, outcome_code=1, risk_grade=7,
        master_scale_version="v1", probability_of_default=0.08,
    )
    store.append(r2)
    assert store.latest(1)["decision_of_record_id"] == r2["decision_of_record_id"]
    assert [r["decision_of_record_id"] for r in store.history(1)] == [r1["decision_of_record_id"], r2["decision_of_record_id"]]
    assert store.latest(999) is None


def test_grade_migration_causes_sum_exactly_to_the_observed_movement():
    previous = {"risk_grade": 6, "master_scale_version": "v1", "probability_of_default": 0.05,
                "probability_of_default_unadjusted": 0.04}
    current = {"risk_grade": 8, "master_scale_version": "v1", "probability_of_default": 0.12,
               "probability_of_default_unadjusted": 0.10}
    migration = history.grade_migration(previous, current,
                                         counterfactual_pd_business_only=0.07,
                                         counterfactual_pd_business_and_structure=0.09)
    assert migration["causes_sum_to_observed"] is True
    assert abs(sum(migration["causes"].values()) - migration["observed_pd_movement"]) < 1e-9
    assert set(migration["causes"]) == set(vocab.CAUSE_ORDER)


def test_scale_residual_is_always_zero_in_this_single_master_scale_slice():
    """Honest limitation, documented in history.grade_migration's own docstring:
    because `entity_data` is defined as "whatever remains to reach the actual
    current PD" (pd3 is pinned to pd_current, not independently re-derived),
    the five other causes telescope to the observed movement **by construction**
    for any counterfactual inputs, in a slice with only one live scorecard
    version and one live master scale -- so `scale`'s residual-defect check is
    exercised as "always passes" here, never as a real catch. A second live
    scorecard/master-scale version (SCOPE.md explicitly skips this, matching
    00/05's own precedent) is what would let a genuine residual appear."""
    previous = {"risk_grade": 6, "master_scale_version": "v1", "probability_of_default": 0.05,
                "probability_of_default_unadjusted": 0.05}
    current = {"risk_grade": 8, "master_scale_version": "v1", "probability_of_default": 0.20,
               "probability_of_default_unadjusted": 0.20}
    # Even with counterfactual inputs that explain almost none of the movement,
    # the residual lands at exactly zero, not flagged as a defect.
    migration = history.grade_migration(previous, current,
                                         counterfactual_pd_business_only=0.051,
                                         counterfactual_pd_business_and_structure=0.052)
    assert migration["causes"][vocab.CAUSE_SCALE] == 0.0
    assert migration["residual_is_defect"] is False
    # All of the unexplained movement instead lands on entity_data, honestly --
    # not hidden, just attributed to the last cause in the chain.
    assert migration["causes"][vocab.CAUSE_ENTITY_DATA] > 0.14


def test_grade_migration_is_not_comparable_when_the_master_scale_changed():
    previous = {"risk_grade": 6, "master_scale_version": "v1", "probability_of_default": 0.05,
                "probability_of_default_unadjusted": 0.05}
    current = {"risk_grade": 7, "master_scale_version": "v2", "probability_of_default": 0.06,
               "probability_of_default_unadjusted": 0.06}
    migration = history.grade_migration(previous, current, 0.05, 0.05)
    assert migration["comparison_basis_code"] == vocab.COMPARISON_NOT_COMPARABLE
    assert migration["residual_is_defect"] is False  # a scale change legitimately absorbs the residual
