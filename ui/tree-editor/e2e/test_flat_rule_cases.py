"""Cases nodes in flat rules: `otherwise` is an index into `branches`, kept valid through edits, clones and pastes."""

from playwright.sync_api import Page, expect

from flat_rules_ui import export, open_flat_rules, rule_card, save
from seed import create_project
from test_flat_rule_params import OTHERWISE_DEFAULT, THEN_APPROVE, decisions, flat_document

REVIEW = {"type": "leaf", "result_idx": 1}
BANDS = {"type": "cases", "op": "ranges", "strict": True, "feature": "amount", "otherwise": 2,
         "conditions": [{"when": {"min": None, "max": 1000}, "then": 0}, {"when": {"min": None, "max": 5000}, "then": 1}],
         "branches": [THEN_APPROVE, OTHERWISE_DEFAULT, REVIEW]}
AMOUNTS = [500.0, 3000.0, 9000.0]
EXPECTED = ["approve", "decline", "review"]


def open_bands(page: Page, *rules: dict) -> None:
    document = flat_document(*(rules or [BANDS]))
    document["output"]["data"].append({"decision": "review"})
    open_flat_rules(page, create_project(page, "Cases", document)["id"])


def test_a_cases_node_round_trips_with_its_otherwise_branch(signed_in: Page, tmp_path):
    open_bands(signed_in)
    assert save(signed_in)["validation_errors"] == []
    document = export(signed_in, tmp_path)
    assert document["rules"][0]["rule"]["otherwise"] == 2
    assert decisions(document, AMOUNTS) == EXPECTED


def test_cloned_rule_with_a_cases_node_saves_valid_and_runs_the_same(signed_in: Page, tmp_path):
    open_bands(signed_in)
    rule_card(signed_in).get_by_title("Clone Rule").click()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(2)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert [r["rule"]["otherwise"] for r in reply["project_data"]["rules"]] == [2, 2]


def test_pasted_cases_node_keeps_its_otherwise_branch(signed_in: Page):
    open_bands(signed_in, BANDS, OTHERWISE_DEFAULT)
    rule_card(signed_in, 0).get_by_title("Copy node").first.click()
    rule_card(signed_in, 1).get_by_title("Paste here").first.click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert [r["rule"]["otherwise"] for r in reply["project_data"]["rules"]] == [2, 2]


def test_deleting_a_branch_before_the_otherwise_branch_keeps_pointing_at_it(signed_in: Page, tmp_path):
    open_bands(signed_in)
    rule_card(signed_in).get_by_title("Delete branch").nth(1).click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert reply["project_data"]["rules"][0]["rule"]["otherwise"] == 1  # still the review branch


def test_the_otherwise_branch_of_a_cases_node_can_be_chosen(signed_in: Page, tmp_path):
    open_bands(signed_in)
    rule_card(signed_in).get_by_test_id("cases-otherwise").click()
    signed_in.get_by_role("option", name="Branch 0").click()

    assert save(signed_in)["project_data"]["rules"][0]["rule"]["otherwise"] == 0
    assert decisions(export(signed_in, tmp_path), AMOUNTS) == ["approve", "decline", "approve"]
