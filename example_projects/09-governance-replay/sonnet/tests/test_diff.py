"""Semantic version diff (spec 09 §5.4): keyed by content, not position."""
import copy
import json
from pathlib import Path

import pytest

from governance import diff, flows


def _rate_card_doc() -> dict:
    path = flows.get("03").project_dir() / "configs" / "0.1.0" / "rate_card_flex_loan.json"
    return json.loads(Path(path).read_text())


def _live_rules_doc() -> dict:
    path = flows.get("01").project_dir() / "configs" / "0.1.0" / "live_rules.json"
    return json.loads(Path(path).read_text())


def test_reordering_a_table_unchanged_produces_an_empty_diff():
    doc = _rate_card_doc()
    reordered = list(reversed(doc["rows"]))
    changes = diff.diff_table_rows(doc["rows"], reordered, ("amt_lo", "term_lo", "grade"), "rate")
    assert changes == []


def test_one_repriced_cell_is_the_only_change_reported():
    doc = _rate_card_doc()
    new_rows = copy.deepcopy(doc["rows"])
    new_rows[100]["rate"] += 0.25
    changes = diff.diff_table_rows(doc["rows"], new_rows, ("amt_lo", "term_lo", "grade"), "rate")
    assert len(changes) == 1
    assert changes[0].change == "repriced"
    assert changes[0].after - changes[0].before == 0.25


def test_summary_aggregates_instead_of_listing_every_cell():
    doc = _rate_card_doc()
    new_rows = copy.deepcopy(doc["rows"])
    for row in new_rows:
        if row["grade"] in (7, 8, 9):
            row["rate"] += 0.31
    changes = diff.diff_table_rows(doc["rows"], new_rows, ("amt_lo", "term_lo", "grade"), "rate")
    summary = diff.summarize_table_diff(changes, len(doc["rows"]), group_field_index=2)
    assert summary["repriced"] == len(changes)
    assert summary["mean_move"] == pytest.approx(0.31)
    assert summary["any_moved_down"] is False
    assert set(summary["by_group"]) == {7, 8, 9}


def test_rule_set_diff_is_keyed_by_rule_id_not_position():
    """09 §5.15 item 2: inserting a rule mid-priority must not make every later rule
    look changed."""
    doc = _live_rules_doc()
    new_doc = copy.deepcopy(doc)
    inserted = copy.deepcopy(new_doc["tree"]["rules"][0])
    inserted["meta"]["name"] = "CF-9999"
    new_doc["tree"]["rules"].insert(1, inserted)  # shifts every later rule's list position
    changes = diff.diff_rule_set(doc, new_doc)
    assert len(changes) == 1
    assert changes[0].rule_id == "CF-9999"
    assert changes[0].change == "added"


def test_rule_set_diff_finds_a_reworded_condition():
    doc = _live_rules_doc()
    new_doc = copy.deepcopy(doc)
    new_doc["tree"]["rules"][3]["rule"]["conditions"][0]["threshold"] = 12345
    changes = diff.diff_rule_set(doc, new_doc)
    assert len(changes) == 1
    assert changes[0].change == "reworded"


def test_params_diff_reports_the_full_key_path():
    before = {"loan_granting": {"statutory_ceiling": {"repo_rate": 0.07}}}
    after = {"loan_granting": {"statutory_ceiling": {"repo_rate": 0.075}}}
    changes = diff.diff_params(before, after)
    assert changes == [{"path": "loan_granting/statutory_ceiling/repo_rate", "before": 0.07, "after": 0.075}]


def test_find_param_does_not_match_a_step_that_shares_its_param_s_name():
    """A step and its one param can share a name (`evaluation_ceiling_per_term`); only
    the leaf value must match, never the enclosing step dict."""
    params = {"loan_granting": {"evaluation_ceiling_per_term": {"evaluation_ceiling_per_term": 24}}}
    hits = diff.find_param(params, "evaluation_ceiling_per_term")
    assert hits == ["loan_granting/evaluation_ceiling_per_term/evaluation_ceiling_per_term"]
