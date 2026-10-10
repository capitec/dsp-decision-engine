"""Event classification: thresholds by criticality, the construction overlay, disputes and unknown amounts."""
from datetime import date

import pytest
from decikit import Engine

from business_nested import events, vocab

SECTOR_ONE, CONSTRUCTION = 100, 412
IMMATERIAL, MINOR, MATERIAL, DISQUALIFYING = 1, 2, 3, 4

classify = Engine().bind(events.event_unit(), mode="interpreted")


def classified(amount=1_000.0, event_type_code=1, criticality_class=vocab.CRITICAL, sector_code=SECTOR_ONE,
               decision_date=date(2026, 9, 24), params=None, **flags):
    item = {"event_id": 1, "event_type_code": event_type_code, "amount": amount, "event_date": date(2026, 8, 1),
            "status_is_active": True, "is_disputed": False, "is_satisfied": False, "decision_date": decision_date,
            "sector_code": sector_code, "criticality_class": criticality_class, **flags}
    return classify.run(item, params)


@pytest.mark.parametrize("criticality_class, severity", [
    (vocab.CRITICAL, DISQUALIFYING), (vocab.SIGNIFICANT, MATERIAL), (vocab.PERIPHERAL, MATERIAL)])
def test_the_same_unsatisfied_judgment_weighs_more_on_a_more_critical_entity(criticality_class, severity):
    out = classified(80_000.0, criticality_class=criticality_class)
    assert out["event_severity_code"] == severity


@pytest.mark.parametrize("amount, severity", [
    (9_999.0, MINOR), (10_000.0, MATERIAL), (49_999.0, MATERIAL), (50_000.0, DISQUALIFYING)])
def test_a_critical_judgment_is_material_from_ten_thousand_and_disqualifying_from_fifty_thousand(amount, severity):
    assert classified(amount)["event_severity_code"] == severity


def test_thresholds_are_read_for_the_event_type_and_the_row_is_recorded():
    out = classified(20_000.0, event_type_code=5, criticality_class=vocab.PERIPHERAL)  # tax notice, peripheral
    assert (out["material_threshold_base"], out["disqualifying_threshold_base"]) == (100_000.0, 500_000.0)
    assert out["threshold_cell_id"] == "event_thresholds@ev-thresh-2026.09#5|3"
    assert out["event_severity_code"] == MINOR


def test_an_event_type_without_its_own_row_uses_the_default_thresholds():
    out = classified(6_000.0, event_type_code=6)
    assert (out["material_threshold_base"], out["disqualifying_threshold_base"]) == (5_000.0, 50_000.0)
    assert out["event_severity_code"] == MATERIAL


def test_a_satisfied_judgment_is_immaterial():
    assert classified(80_000.0, is_satisfied=True)["event_severity_code"] == IMMATERIAL


def test_a_judgment_of_unknown_amount_is_material_and_flagged():
    out = classified(None)
    assert (out["event_severity_code"], out["amount_unknown"]) == (MATERIAL, True)
    assert classified(None, is_satisfied=True)["event_severity_code"] == IMMATERIAL
    assert classified(0.0)["amount_unknown"] is False


def test_a_fraud_conviction_is_disqualifying_whatever_the_amount():
    assert classified(None, event_type_code=12, criticality_class=vocab.PERIPHERAL)["event_severity_code"] == DISQUALIFYING


def test_construction_judgments_are_judged_against_a_halved_materiality_threshold():
    out = classified(7_000.0, sector_code=CONSTRUCTION)
    assert (out["material_threshold_base"], out["material_threshold"]) == (10_000.0, 5_000.0)
    assert out["threshold_overlay_ids"] == ["ADJ-05-014"]
    assert (out["event_severity_code"], out["event_severity_code_unadjusted"]) == (MATERIAL, MINOR)


def test_the_overlay_leaves_other_sectors_and_other_event_types_alone():
    other_sector = classified(7_000.0, sector_code=SECTOR_ONE)
    default_listing = classified(4_000.0, event_type_code=2, sector_code=CONSTRUCTION)  # material from 5 000, base
    for out in (other_sector, default_listing):
        assert out["threshold_overlay_ids"] == [] and out["material_threshold"] == out["material_threshold_base"]
    assert other_sector["event_severity_code"] == MINOR


def test_the_overlay_applies_only_while_it_is_in_force():
    assert classified(7_000.0, sector_code=CONSTRUCTION, decision_date=date(2026, 12, 30))["threshold_overlay_ids"] == ["ADJ-05-014"]
    after = classified(7_000.0, sector_code=CONSTRUCTION, decision_date=date(2027, 1, 1))
    assert after["threshold_overlay_ids"] == [] and after["event_severity_code"] == MINOR


def test_the_overlays_can_be_switched_off_by_a_param():
    off = {"event": {"overlaid_thresholds": {"event_overlays_enabled": False}}}
    out = classified(7_000.0, sector_code=CONSTRUCTION, params=off)
    assert (out["threshold_overlay_ids"], out["event_severity_code"]) == ([], MINOR)


@pytest.mark.parametrize("amount, undisputed, disputed", [
    (80_000.0, DISQUALIFYING, MATERIAL), (20_000.0, MATERIAL, MINOR), (5_000.0, MINOR, MINOR), (0.0, IMMATERIAL, IMMATERIAL)])
def test_a_disputed_event_is_classified_one_class_lower_but_never_below_minor(amount, undisputed, disputed):
    assert classified(amount)["event_severity_code"] == undisputed
    assert classified(amount, is_disputed=True)["event_severity_code"] == disputed


def test_the_age_of_an_event_is_the_calendar_months_between_it_and_the_decision():
    assert classified()["age_months"] == 1  # August to September
    assert classified(decision_date=date(2027, 9, 1))["age_months"] == 13
