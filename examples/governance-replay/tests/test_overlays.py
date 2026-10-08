from datetime import date

from governance import overlays

from conftest import BASE_IDS


def test_the_register_lists_every_flows_overlays_once():
    rows = overlays.register()
    assert [(r["flow"], r["adjustment_id"]) for r in rows] == [
        ("fraud", "ADJ-2026-001"), ("fraud", "ADJ-2026-002"), ("fraud", "ADJ-2026-003"),
        ("loan", "ADJ-03-001"), ("loan", "ADJ-03-002"), ("loan", "ADJ-03-003"), ("loan", "ADJ-03-004"),
        ("business", "ADJ-05-014"), ("business", "ADJ-05-ENT-001"), ("business", "ADJ-05-BIZ-001")]


def test_ageing_lists_overlays_past_review_and_not_yet_lapsed_oldest_first():
    on = date(2026, 1, 10)  # the festive overlay was due for review on the 8th and lapses on the 15th
    assert [(r["adjustment_id"], r["age_days"]) for r in overlays.ageing(on)] == [("ADJ-2026-002", 2)]
    assert overlays.ageing(date(2020, 1, 1)) == []
    assert [r["age_days"] for r in overlays.ageing(date(2026, 12, 31))] == sorted(
        (r["age_days"] for r in overlays.ageing(date(2026, 12, 31))), reverse=True)


def test_switching_off_flips_every_switch_wherever_it_sits_and_leaves_the_rest():
    params = {"a": {"adjustment_stack_enabled": True, "cut": 0.4}, "b": {"c": {"event_overlays_enabled": True}}}
    assert overlays.switched_off(params) == {"a": {"adjustment_stack_enabled": False, "cut": 0.4},
                                             "b": {"c": {"event_overlays_enabled": False}}}
    assert params["a"]["adjustment_stack_enabled"] is True


def test_unwinding_the_overlays_changes_only_the_decision_the_festive_overlay_tipped(evidence):
    result = overlays.unwind_estimate(evidence.all("fraud"))
    assert result == {"decisions": 9, "changed": 1, "moves": {f"{BASE_IDS['fraud']}-v2": (40, 30)}}


def test_unwinding_overlays_that_never_decided_anything_changes_nothing(evidence):
    assert overlays.unwind_estimate(evidence.all("loan"))["changed"] == 0
