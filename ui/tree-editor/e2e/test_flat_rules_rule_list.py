"""Managing the rules of a document: add, rename, describe, reorder, clone, delete, search and size."""

import time

import pytest
from playwright.sync_api import Page, expect

from flat_docs import APPROVE, OUTPUT3, decisions_for, document, open_doc, run_frame
from flat_rules_ui import rename_rule, rule_card, rule_names, save
from seed import create_project, export_document, get_project

SMALL = {"type": "unary", "condition": {"type": "unary", "op": "<=", "feature": "amount", "threshold": 5}, "then": APPROVE}


def numbered(count: int) -> list[dict]:
    return [{"type": "unary", "condition": {"type": "unary", "op": "<=", "feature": "amount", "threshold": i},
             "then": APPROVE} for i in range(count)]


def test_added_rule_is_saved_after_the_existing_ones(signed_in: Page):
    project = open_doc(signed_in, SMALL)
    signed_in.get_by_role("button", name="Add Rule").click()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(2)

    assert save(signed_in)["validation_errors"] == []
    names = [r["meta"]["name"] for r in export_document(signed_in, project)["rules"]]
    assert names == ["Rule 1", "Rule 2"]


@pytest.mark.smoke
def test_renamed_rule_keeps_its_new_name_after_save_and_reload(signed_in: Page):
    open_doc(signed_in, SMALL)
    rename_rule(signed_in, "Low amounts")

    assert save(signed_in)["project_data"]["rules"][0]["meta"]["name"] == "Low amounts"
    signed_in.reload()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(1, timeout=30_000)
    assert rule_names(signed_in) == ["Low amounts"]


@pytest.mark.parametrize("name", ["Zażółć gęślą 評価 🙂", "n" * 200])
def test_unicode_and_very_long_rule_names_survive_a_round_trip(signed_in: Page, name):
    project = open_doc(signed_in, SMALL)
    rename_rule(signed_in, name)

    assert save(signed_in)["validation_errors"] == []
    assert export_document(signed_in, project)["rules"][0]["meta"]["name"] == name
    signed_in.reload()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(1, timeout=30_000)
    assert rule_names(signed_in) == [name]


@pytest.mark.xfail(strict=True, reason="an empty rule name is saved as is")
def test_empty_rule_name_is_not_saved_as_an_empty_name(signed_in: Page):
    open_doc(signed_in, SMALL)
    rename_rule(signed_in, "")

    saved = save(signed_in)
    assert saved["project_data"]["rules"][0]["meta"]["name"]


@pytest.mark.xfail(strict=True, reason="two rules of one name in all mode save valid, and the second's answer replaces the first's")
def test_duplicate_rule_names_in_all_mode_are_reported_or_both_rules_answer(signed_in: Page):
    never = {**SMALL, "condition": {**SMALL["condition"], "threshold": 0}}
    project = open_doc(signed_in, SMALL, never, mode="all")
    rename_rule(signed_in, "Same", 0)
    rename_rule(signed_in, "Same", 1)

    reply = save(signed_in)
    if reply["validation_errors"]:
        return  # refused, with the reason on the rules
    (row,) = run_frame(export_document(signed_in, project), {"amount": [1.0]})
    assert sorted(row.values()) == ["approve", "decline"]


@pytest.mark.xfail(strict=True, reason="a rule description is not part of the saved document")
def test_rule_description_survives_save_and_reload(signed_in: Page):
    open_doc(signed_in, SMALL)
    card = rule_card(signed_in)
    card.get_by_text("Add description...").click()
    card.get_by_placeholder("Add description...").fill("Small amounts only")
    save(signed_in)
    signed_in.reload()

    expect(rule_card(signed_in).get_by_text("Small amounts only")).to_be_visible()


def test_moving_a_rule_up_and_down_changes_the_saved_order_and_the_first_match(signed_in: Page):
    below = {**SMALL, "then": {"type": "leaf", "result_idx": 0}}
    other = {**SMALL, "then": {"type": "leaf", "result_idx": 1}}
    project = open_doc(signed_in, other, below, output=OUTPUT3)
    assert decisions_for(export_document(signed_in, project), {"amount": [1.0]}) == ["review"]

    signed_in.get_by_test_id("rule-move-up").nth(1).click()
    assert save(signed_in)["validation_errors"] == []
    doc = export_document(signed_in, project)
    assert [r["meta"]["name"] for r in doc["rules"]] == ["Rule 2", "Rule 1"]
    assert decisions_for(doc, {"amount": [1.0]}) == ["approve"]

    signed_in.get_by_test_id("rule-move-down").first.click()
    save(signed_in)
    assert [r["meta"]["name"] for r in export_document(signed_in, project)["rules"]] == ["Rule 1", "Rule 2"]


def test_first_rule_cannot_move_up_and_last_cannot_move_down(signed_in: Page):
    open_doc(signed_in, SMALL, SMALL)

    expect(signed_in.get_by_test_id("rule-move-up").first).to_be_disabled()
    expect(signed_in.get_by_test_id("rule-move-down").last).to_be_disabled()


def test_cloned_rule_is_a_named_copy_that_runs_like_the_original(signed_in: Page):
    project = open_doc(signed_in, SMALL, mode="all")
    rule_card(signed_in).get_by_title("Clone Rule").click()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(2)
    assert rule_names(signed_in) == ["Rule 1", "Rule 1 (Copy 1)"]

    assert save(signed_in)["validation_errors"] == []
    rows = run_frame(export_document(signed_in, project), {"amount": [1.0, 9.0]})
    assert rows == [{"Rule 1.decision": "approve", "Rule 1 (Copy 1).decision": "approve"},
                    {"Rule 1.decision": "decline", "Rule 1 (Copy 1).decision": "decline"}]


def test_deleting_a_rule_asks_first_and_removes_it_from_the_export(signed_in: Page):
    project = open_doc(signed_in, SMALL, SMALL)
    signed_in.get_by_test_id("rule-delete").first.click()
    dialog = signed_in.get_by_role("alertdialog")
    expect(dialog).to_contain_text("Rule 1")
    dialog.get_by_role("button", name="Cancel").click()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(2)

    signed_in.get_by_test_id("rule-delete").first.click()
    signed_in.get_by_role("button", name="Delete Rule").click()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(1)
    save(signed_in)
    assert [r["meta"]["name"] for r in export_document(signed_in, project)["rules"]] == ["Rule 2"]


def test_the_only_rule_cannot_be_deleted(signed_in: Page):
    open_doc(signed_in, SMALL)

    expect(signed_in.get_by_test_id("rule-delete")).to_be_disabled()


def test_the_prioritisation_mode_of_a_document_survives_an_edit_and_save(signed_in: Page):
    project = open_doc(signed_in, SMALL, SMALL, mode="all")
    rename_rule(signed_in, "Renamed")
    save(signed_in)

    assert get_project(signed_in, project)["project_data"]["mode"] == "all"
    assert signed_in.get_by_text("Mode").count() == 0  # the editor has no control for it


def test_search_filters_rules_by_name_and_clears(signed_in: Page):
    open_doc(signed_in, SMALL, SMALL, SMALL)
    rename_rule(signed_in, "Alpha", 0)
    rename_rule(signed_in, "Beta", 1)
    rename_rule(signed_in, "Gamma", 2)
    search = signed_in.get_by_placeholder("Search rules by name, description, or priority...")

    search.fill("bet")
    assert rule_names(signed_in) == ["Beta"]
    search.fill("nothing like it")
    expect(signed_in.get_by_text('No rules found matching "nothing like it"')).to_be_visible()
    search.fill("")
    assert rule_names(signed_in) == ["Alpha", "Beta", "Gamma"]


@pytest.mark.parametrize("count", [
    1,
    pytest.param(64, marks=pytest.mark.slow),
    pytest.param(65, marks=pytest.mark.slow),
    pytest.param(200, marks=pytest.mark.slow),
])
def test_documents_of_many_rules_open_edit_and_save_within_a_bound(signed_in: Page, count):
    project = create_project(signed_in, "Many", document(*numbered(count)))["id"]

    started = time.perf_counter()
    signed_in.goto(f"/flat-rules/{project}")
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(count, timeout=60_000)
    opened = time.perf_counter() - started
    last = signed_in.get_by_test_id("rule-card").last
    last.scroll_into_view_if_needed()
    rename_rule(signed_in, "Edited", count - 1)

    started = time.perf_counter()
    reply = save(signed_in)
    saved = time.perf_counter() - started
    print(f"{count} rules: open {opened:.1f}s, save {saved:.1f}s")
    assert opened < 10 and saved < 10
    assert reply["validation_errors"] == []
    doc = export_document(signed_in, project)
    assert doc["rules"][-1]["meta"]["name"] == "Edited" and len(doc["rules"]) == count
    assert decisions_for(doc, {"amount": [0.0, 1000.0]}) == ["approve", "decline"]
