from datetime import date

import pytest

from credit_core import adverse_events as ae


def severity(**event):
    return ae.event_severity_code(**event)


def test_the_catalogue_has_fourteen_distinct_event_types():
    assert len(set(ae.EVENT_TYPES)) == 14


@pytest.mark.parametrize("amount,expected", [
    (0.0, ae.IMMATERIAL), (1.0, ae.MINOR), (4999.99, ae.MINOR), (5000.0, ae.MATERIAL),
    (49999.99, ae.MATERIAL), (50000.0, ae.DISQUALIFYING)])
def test_an_amount_event_is_graded_by_the_thresholds(amount, expected):
    assert severity(event_type_code=ae.JUDGMENT, amount=amount) == expected


def test_the_caller_supplies_the_thresholds(score):
    record = {"event_type_code": ae.JUDGMENT, "amount": 10_000.0}
    lenient = {"event_severity_code": {"material_threshold": 50_000.0, "disqualifying_threshold": 100_000.0}}
    strict = {"event_severity_code": {"material_threshold": 1_000.0, "disqualifying_threshold": 5_000.0}}
    assert score(ae.event_severity_code_step, record, lenient)["event_severity_code"] == ae.MINOR
    assert score(ae.event_severity_code_step, record, strict)["event_severity_code"] == ae.DISQUALIFYING


@pytest.mark.parametrize("event_type", [ae.SEQUESTRATION, ae.FRAUD_CONVICTION, ae.ADMINISTRATION_ORDER])
def test_some_events_disqualify_while_active_whatever_the_amount(event_type):
    assert severity(event_type_code=event_type, amount=1.0) == ae.DISQUALIFYING
    assert severity(event_type_code=event_type, amount=1.0, status_is_active=False) == ae.MINOR


def test_status_only_events_are_material_while_active_and_minor_once_lifted():
    assert severity(event_type_code=ae.DEBT_REVIEW) == ae.MATERIAL
    assert severity(event_type_code=ae.DEBT_REVIEW, status_is_active=False) == ae.MINOR


def test_a_satisfied_event_is_immaterial_unless_it_always_disqualifies():
    assert severity(event_type_code=ae.JUDGMENT, amount=60_000.0, satisfied=True) == ae.IMMATERIAL
    assert severity(event_type_code=ae.SEQUESTRATION, satisfied=True) == ae.DISQUALIFYING


def test_a_disputed_inactive_event_is_minor():
    assert severity(event_type_code=ae.JUDGMENT, amount=60_000.0, disputed=True, status_is_active=False) == ae.MINOR


def test_age_is_whole_calendar_months_and_never_negative():
    assert ae.event_age_months(date(2025, 9, 30), date(2026, 9, 1)) == 12
    assert ae.event_age_months(date(2026, 10, 1), date(2026, 9, 1)) == 0
