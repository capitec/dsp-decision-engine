"""Composite nesting, cases nodes, branches and what the editor drops or keeps of them on save."""

import pytest
from playwright.sync_api import Page, expect

from flat_docs import APPROVE, DEFAULT, OUTPUT3, decisions_for, open_doc
from flat_rules_ui import (
    add_isin_values,
    add_otherwise,
    add_then,
    build_nested_composite,
    choose_feature,
    choose_operator,
    choose_output,
    fill_number,
    fill_pattern,
    rule_card,
    save,
    set_node_type,
    unary_rule,
)
from seed import export_document


def saved_rule(reply: dict, rule: int = 0) -> dict:
    return reply["project_data"]["rules"][rule]["rule"]


@pytest.mark.parametrize("ops", [
    pytest.param(["and"], marks=pytest.mark.smoke),
    ["or", "not"],
    ["and", "or", "and"],
    ["not", "and", "or", "not"],
    ["or", "and", "not", "or", "and"],
])
def test_nested_composite_of_each_depth_saves_and_runs(signed_in: Page, ops):
    project = open_doc(signed_in)
    card = rule_card(signed_in)
    build_nested_composite(signed_in, card, ops)
    add_then(signed_in, card)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    depth, node = 0, saved_rule(reply)
    while node["type"] == "composite":
        assert node["op"] == ops[depth]
        depth, node = depth + 1, node["conditions"][0]
    assert (depth, node["op"], node["threshold"]) == (len(ops), "<=", 100)

    negated = ops.count("not") % 2 == 1
    expected = [negated != holds for holds in (True, False)]
    got = [d == "approve" for d in decisions_for(export_document(signed_in, project), {"amount": [50.0, 150.0]})]
    assert got == expected


def test_composite_with_two_conditions_combines_them(signed_in: Page):
    project = open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Composite")
    choose_operator(signed_in, card, "or", testid="composite-op")
    for _ in range(2):
        card.get_by_role("button", name="Add Condition", exact=True).click()
    rows = card.get_by_test_id("condition-row")
    choose_feature(signed_in, rows.nth(0), "amount")
    fill_number(rows.nth(0), "10")
    choose_operator(signed_in, rows.nth(1), ">=", testid="condition-op")
    choose_feature(signed_in, rows.nth(1), "amount")
    fill_number(rows.nth(1), "100")
    add_then(signed_in, card)

    assert save(signed_in)["validation_errors"] == []
    decided = decisions_for(export_document(signed_in, project), {"amount": [5.0, 50.0, 150.0]})
    assert decided == ["approve", "decline", "approve"]


def test_not_is_not_offered_once_a_composite_has_two_conditions(signed_in: Page):
    open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Composite")
    for _ in range(2):
        card.get_by_role("button", name="Add Condition", exact=True).click()
    card.get_by_test_id("composite-op").click()

    expect(signed_in.get_by_role("option", name="NOT", exact=True)).to_be_disabled()


def test_changing_a_nested_composite_operator_keeps_its_conditions(signed_in: Page):
    open_doc(signed_in)
    card = rule_card(signed_in)
    build_nested_composite(signed_in, card, ["and", "and"])
    inner = card.get_by_test_id("condition-row").first
    choose_operator(signed_in, inner, "or", testid="condition-op")

    nested = saved_rule(save(signed_in))["conditions"][0]
    assert (nested["op"], len(nested["conditions"])) == ("or", 1)


def build_cases(page: Page, op: str, feature: str, branches: int = 3):
    """Rule 1 becomes a cases node of kind `op` over `feature` with `branches` default leaves."""
    card = rule_card(page)
    set_node_type(page, card, "Cases")
    if op != "ranges":
        card.get_by_test_id("cases-op").click()
        page.get_by_role("option", name={"isin": "IsIn", "string_match": "String Match"}[op], exact=True).click()
    choose_feature(page, card, feature)
    for _ in range(branches):
        card.get_by_role("button", name="Add Branch").click()
    return card


def set_case_branch(page: Page, card, row: int, branch: int) -> None:
    card.get_by_test_id("case-then").nth(row).click()
    page.get_by_role("option", name=f"Branch {branch}", exact=True).click()


def test_cases_isin_routes_values_to_branches_and_the_rest_to_otherwise(signed_in: Page):
    project = open_doc(signed_in, output=OUTPUT3)
    card = build_cases(signed_in, "isin", "count")
    choose_output(signed_in, card, "approve", 1)
    choose_output(signed_in, card, "review", 2)
    for row, (values, branch) in enumerate([([1, 2], 1), ([3], 2)]):
        card.get_by_role("button", name="Add Condition", exact=True).click()
        add_isin_values(card, values)
        set_case_branch(signed_in, card, row, branch)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    rule = saved_rule(reply)
    assert [(c["when"]["values"], c["then"]) for c in rule["conditions"]] == [([1, 2], 1), ([3], 2)]
    assert rule["otherwise"] == 0 and len(rule["branches"]) == 3
    decided = decisions_for(export_document(signed_in, project), {"count": [1, 2, 3, 4]})
    assert decided == ["approve", "approve", "review", "decline"]


def test_cases_string_match_routes_patterns_to_branches(signed_in: Page):
    project = open_doc(signed_in, output=OUTPUT3)
    card = build_cases(signed_in, "string_match", "band")
    choose_output(signed_in, card, "approve", 1)
    choose_output(signed_in, card, "review", 2)
    for row, (pattern, branch) in enumerate([("gold", 1), ("silver", 2)]):
        card.get_by_role("button", name="Add Condition", exact=True).click()
        fill_pattern(card, pattern, row)
        set_case_branch(signed_in, card, row, branch)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    decided = decisions_for(export_document(signed_in, project), {"band": ["gold", "silver", "bronze"]})
    assert decided == ["approve", "review", "decline"]


def test_cases_ranges_in_strict_mode_infer_the_lower_bounds(signed_in: Page):
    project = open_doc(signed_in, output=OUTPUT3)
    card = build_cases(signed_in, "ranges", "amount")
    choose_output(signed_in, card, "approve", 1)
    choose_output(signed_in, card, "review", 2)
    card.get_by_role("checkbox", name="Strict Mode").click()
    for row, (upper, branch) in enumerate([("10", 1), ("20", 2)]):
        card.get_by_role("button", name="Add Condition", exact=True).click()
        fill_number(card, upper, row)
        set_case_branch(signed_in, card, row, branch)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_rule(reply)["strict"] is True
    decided = decisions_for(export_document(signed_in, project), {"amount": [-50.0, 9.0, 15.0, 25.0]})
    assert decided == ["approve", "approve", "review", "decline"]  # past the last band: otherwise, branch 0


def manual_ranges(page: Page, bands: list[tuple[str, str, int]]):
    card = build_cases(page, "ranges", "amount", branches=2)
    choose_output(page, card, "approve", 1)
    for row, (lower, upper, branch) in enumerate(bands):
        card.get_by_role("button", name="Add Condition", exact=True).click()
        fill_number(card, lower, 2 * row)
        fill_number(card, upper, 2 * row + 1)
        set_case_branch(page, card, row, branch)
    return card


def test_manual_ranges_with_a_gap_save_valid(signed_in: Page):
    open_doc(signed_in)
    manual_ranges(signed_in, [("0", "10", 1), ("20", "30", 1)])

    assert save(signed_in)["validation_errors"] == []


def test_manual_ranges_run_the_way_the_editor_shows_them(signed_in: Page):
    project = open_doc(signed_in)
    manual_ranges(signed_in, [("0", "10", 1), ("10", "20", 1)])

    assert save(signed_in)["validation_errors"] == []
    decided = decisions_for(export_document(signed_in, project), {"amount": [-5.0, 5.0]})
    assert decided == ["decline", "approve"]


def test_deleting_the_then_branch_makes_a_match_use_the_default_row(signed_in: Page):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "<=", then="approve", otherwise=True)
    fill_number(card, "100")
    card.get_by_title("Delete THEN branch").click()
    expect(card.get_by_role("button", name="Add THEN")).to_be_visible()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert "then" not in saved_rule(reply)
    assert decisions_for(export_document(signed_in, project), {"amount": [50.0, 150.0]}) == ["decline", "decline"]


def test_otherwise_leaf_can_be_pointed_at_a_record_when_there_is_no_then_branch(signed_in: Page):
    project = open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Unary")
    choose_feature(signed_in, card, "amount")
    fill_number(card, "100")
    add_otherwise(card)
    choose_output(signed_in, card, "approve", 0)  # the only leaf: OTHERWISE

    reply = save(signed_in)
    assert saved_rule(reply)["otherwise"]["result_idx"] == 0
    assert decisions_for(export_document(signed_in, project), {"amount": [50.0, 150.0]}) == ["decline", "approve"]


def test_otherwise_branch_can_be_added_pointed_at_a_record_and_deleted(signed_in: Page):
    project = open_doc(signed_in, output=OUTPUT3)
    card = unary_rule(signed_in, "<=", then="approve", otherwise=True)
    fill_number(card, "100")
    choose_output(signed_in, card, "review", 1)

    reply = save(signed_in)
    assert saved_rule(reply)["otherwise"]["result_idx"] == 1
    decided = decisions_for(export_document(signed_in, project), {"amount": [50.0, 150.0]})
    assert decided == ["approve", "review"]

    card.get_by_title("Delete OTHERWISE branch").click()
    assert "otherwise" not in saved_rule(save(signed_in))


def test_a_unary_node_offers_no_branch_list_and_saves_none(signed_in: Page):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "<=")
    fill_number(card, "100")
    expect(card.get_by_role("button", name="Branch", exact=True)).to_have_count(0)  # only cases nodes have branch lists

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert "branches" not in saved_rule(reply)
    assert decisions_for(export_document(signed_in, project), {"amount": [50.0, 150.0]}) == ["approve", "decline"]


def two_branch_cases(otherwise: int) -> dict:
    return {"type": "cases", "op": "isin", "feature": "count", "otherwise": otherwise,
            "conditions": [{"when": {"values": [1]}, "then": 1}], "branches": [DEFAULT, APPROVE]}


def test_copying_a_leaf_and_pasting_it_over_another_leaf_saves_and_runs(signed_in: Page):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "<=", otherwise=True)
    fill_number(card, "100")
    nodes = card.get_by_test_id("rule-node")
    nodes.nth(1).get_by_title("Copy node").first.click()
    nodes.nth(2).get_by_title("Paste here").first.click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_rule(reply)["otherwise"]["result_idx"] == 0
    assert decisions_for(export_document(signed_in, project), {"amount": [50.0, 150.0]}) == ["approve", "approve"]


def test_deleting_a_cases_branch_keeps_the_otherwise_index_valid(signed_in: Page):
    open_doc(signed_in, two_branch_cases(1))
    card = rule_card(signed_in)
    card.get_by_title("Delete branch").nth(1).click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_rule(reply)["otherwise"] == 0  # the only branch left


def test_dragging_a_node_grip_does_not_break_the_rule(signed_in: Page):
    errors = []
    signed_in.on("pageerror", lambda e: errors.append(str(e)))
    open_doc(signed_in)
    card = unary_rule(signed_in, "<=", otherwise=True)
    nodes = card.get_by_test_id("rule-node")
    nodes.nth(1).locator("[draggable=true]").first.drag_to(nodes.nth(2))

    assert errors == []
    assert save(signed_in)["validation_errors"] == []
