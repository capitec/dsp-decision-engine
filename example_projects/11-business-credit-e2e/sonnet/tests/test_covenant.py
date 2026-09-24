"""The DSCR covenant: version binding that never moves once an instance
exists (spec 11 §5.5.1), and the three dates of §5.5.2 kept distinct."""
from __future__ import annotations

from datetime import date

from business_credit_e2e import covenant, vocab


def test_bind_covenant_instance_picks_the_version_in_force_at_binding():
    early = covenant.bind_covenant_instance(facility_id=1, decision_date=date(2025, 6, 1))
    later = covenant.bind_covenant_instance(facility_id=2, decision_date=date(2027, 1, 1))
    assert early["covenant_definition_version"] == "dscr-2024.01"
    assert later["covenant_definition_version"] == "dscr-2026.03"


def test_a_bound_instance_never_re_resolves_even_after_a_new_version_ships():
    """Spec 11 §5.5.1's own worked example: a facility bound in 2026 must not
    see the 2028 lease-liability version, even when tested in 2029."""
    instance = covenant.bind_covenant_instance(facility_id=3, decision_date=date(2026, 6, 1))
    assert instance["covenant_definition_version"] == "dscr-2026.03"

    result = covenant.test_covenant(
        instance, ebitda=1_200_000.0, existing_accounts=[],
        new_facility_instalment=10_000.0, test_date=date(2029, 6, 30),
        delivery_date=date(2029, 8, 29), determination_date=date(2029, 9, 4),
    )
    # Still the version bound in 2026, not the 2028.09 version in force by the 2029 test.
    assert result["covenant_definition_version"] == "dscr-2026.03"


def test_three_dates_are_recorded_distinctly_and_never_conflated():
    instance = covenant.bind_covenant_instance(facility_id=4, decision_date=date(2026, 1, 1))
    result = covenant.test_covenant(
        instance, ebitda=1_200_000.0, existing_accounts=[],
        new_facility_instalment=10_000.0, test_date=date(2029, 6, 30),
        delivery_date=date(2029, 8, 29), determination_date=date(2029, 9, 4),
    )
    assert result["test_date"] == date(2029, 6, 30)
    assert result["delivery_date"] == date(2029, 8, 29)
    assert result["determination_date"] == date(2029, 9, 4)
    assert len({result["test_date"], result["delivery_date"], result["determination_date"]}) == 3


def test_dscr_debt_service_uses_core_obligations_and_adds_the_new_facility():
    accounts = [{"account_type_code": 1, "instalment": 5000.0, "is_internal": False, "closed": False}]
    debt_service = covenant.dscr_debt_service(accounts, new_facility_instalment=2000.0)
    assert debt_service == 7000.0  # 5000 stated instalment (USE_STATED_INSTALMENT) + 2000 new


def test_breach_classification_bands():
    instance = {"covenant_instance_id": 1, "covenant_definition_version": "dscr-2026.03",
                "threshold": 1.25, "cure_days": 30}
    passing = covenant.test_covenant(instance, ebitda=1_500_000.0, existing_accounts=[],
                                      new_facility_instalment=10_000.0, test_date=date(2027, 6, 30),
                                      delivery_date=date(2027, 8, 29), determination_date=date(2027, 8, 29))
    assert passing["breach_class_code"] == vocab.BREACH_NONE

    severe = covenant.test_covenant(instance, ebitda=100_000.0, existing_accounts=[],
                                     new_facility_instalment=50_000.0, test_date=date(2027, 6, 30),
                                     delivery_date=date(2027, 8, 29), determination_date=date(2027, 8, 29))
    assert severe["breach_class_code"] == vocab.BREACH_SEVERE
    assert severe["cure_days_remaining"] is not None


def test_certificate_not_received_is_a_fourth_state_not_a_breach_or_a_pass():
    instance = covenant.bind_covenant_instance(facility_id=5, decision_date=date(2026, 1, 1))
    result = covenant.certificate_not_received(instance, test_date=date(2027, 6, 30),
                                                 determination_date=date(2027, 9, 1))
    assert result["breach_class_code"] == vocab.COVENANT_NOT_TESTED
    assert result["measured_dscr"] is None
    assert result["information_undertaking_breached"] is True
