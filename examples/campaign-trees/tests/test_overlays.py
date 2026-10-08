from dataclasses import replace
from datetime import date

import pytest

from campaign_trees.overlays import CAMPAIGN_ID, TREE_OVERLAYS, resolve

TREE_PARAMS = {"n6_discretionary_income_floor", "n3_behaviour_score_floor"}
VOLUME_DIAL, CUTOFF_SHIFT = TREE_OVERLAYS


def test_an_overlay_is_resolved_by_the_cycle_date_not_by_today():
    assert resolve(TREE_OVERLAYS, CAMPAIGN_ID, date(2026, 8, 31), TREE_PARAMS) == ({}, "OS-NONE")
    assert resolve(TREE_OVERLAYS, CAMPAIGN_ID, date(2026, 9, 24), TREE_PARAMS) == (
        {"n6_discretionary_income_floor": 2600.0}, "OS-ADJ-C23-VOL-001")
    assert resolve(TREE_OVERLAYS, CAMPAIGN_ID, date(2026, 12, 1), TREE_PARAMS) == ({}, "OS-NONE")


def test_overlays_stack_in_the_declared_order_and_the_stack_id_records_it():
    params, stack_id = resolve(TREE_OVERLAYS, CAMPAIGN_ID, date(2026, 10, 5), TREE_PARAMS)
    assert params == {"n6_discretionary_income_floor": 2600.0, "n3_behaviour_score_floor": 640.0}
    assert stack_id == "OS-ADJ-C23-VOL-001+ADJ-C23-CUT-001"


def test_another_campaign_s_overlays_do_not_apply():
    assert resolve(TREE_OVERLAYS, 7, date(2026, 10, 5), TREE_PARAMS) == ({}, "OS-NONE")


def test_an_overlay_past_its_review_date_fails_the_cycle():
    with pytest.raises(ValueError, match="ADJ-C23-VOL-001 is past its review date"):
        resolve(TREE_OVERLAYS, CAMPAIGN_ID, date(2026, 11, 20), TREE_PARAMS)


def test_an_overlay_for_a_param_the_tree_lacks_fails_the_cycle():
    with pytest.raises(ValueError, match="n6_discretionary_income_floor"):
        resolve(TREE_OVERLAYS, CAMPAIGN_ID, date(2026, 9, 24), {"n3_behaviour_score_floor"})


def test_an_overlay_that_loosens_a_floor_cannot_be_defined():
    with pytest.raises(ValueError, match="loosens a floor"):
        replace(VOLUME_DIAL, new_value=2000.0)
