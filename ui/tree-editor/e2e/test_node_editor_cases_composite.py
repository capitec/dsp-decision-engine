"""Node-editor cases and composite nodes: every case type and AND/OR/NOT nesting saves a document decikit runs."""

import re

import pytest
from playwright.sync_api import Page, expect

from node_ui import (
    SCHEMA,
    cases,
    choose_feature,
    composite,
    document,
    edge,
    leaf,
    open_tree,
    run,
    save,
    saved_data,
    select_node,
    set_node_type,
    sidebar,
    two_leaves,
    unary,
)
from seed import create_project, export_document


# ---- cases ----------------------------------------------------------------------------------------------------------

def three_leaves(root: dict) -> dict:
    leaves = [leaf("low", 0, 0), leaf("mid", 1, 250), leaf("high", -1, 500)]
    return document([root, *leaves], [edge(root["id"], t, i) for i, t in enumerate(("low", "mid", "high"))],
                    input_schema=SCHEMA)


def start_cases(page: Page, project_doc: dict, op: str) -> str:
    project = open_tree(page, project_doc)
    select_node(page, "root")
    set_node_type(page, "Cases")
    if op != "Ranges":
        sidebar(page).get_by_role("combobox").filter(has_text="Ranges").click()
        page.get_by_role("option", name=op, exact=True).click()
    return project


def add_condition(page: Page) -> None:
    sidebar(page).get_by_role("button", name="Add Condition").click()


def test_cases_ranges_strict_bands_save_and_route_each_value_to_its_branch(signed_in: Page):
    project = start_cases(signed_in, three_leaves(unary("root")), "Ranges")
    choose_feature(signed_in, "amount")
    for upper in ("100", "1000"):
        add_condition(signed_in)
        sidebar(signed_in).get_by_placeholder("∞", exact=True).last.fill(upper)
    sidebar(signed_in).get_by_label("Strict Mode").click()
    expect(sidebar(signed_in).get_by_label("Strict Mode")).to_be_checked()
    assert save(signed_in)["validation_errors"] == []

    data = saved_data(signed_in, project)
    assert data["strict"] is True
    assert [c["max"] for c in data["conditions"]] == [100, 1000]
    assert run(export_document(signed_in, project), amount=[50.0, 100.0, 999.0, 5000.0]) == [
        "approve", "review", "review", "decline"]


@pytest.mark.xfail(strict=True, reason="ticking Strict Mode on a ranges node with no bands yet turns strict off instead of on")
def test_strict_mode_ticked_before_any_band_is_added_stays_on(signed_in: Page):
    start_cases(signed_in, three_leaves(unary("root")), "Ranges")
    sidebar(signed_in).get_by_label("Strict Mode").click()

    expect(sidebar(signed_in).get_by_label("Strict Mode")).to_be_checked()


@pytest.mark.xfail(strict=True, reason="a ranges node made in the editor shows Strict Mode off but saves no `strict`, so decikit applies strict and rejects gaps")
def test_cases_ranges_with_the_strict_box_left_off_save_without_validation_errors(signed_in: Page):
    project = start_cases(signed_in, three_leaves(unary("root")), "Ranges")
    choose_feature(signed_in, "amount")
    expect(sidebar(signed_in).get_by_label("Strict Mode")).not_to_be_checked()
    for lower, upper in (("0", "10"), ("20", "30")):  # a gap between the bands is fine when strict is off
        add_condition(signed_in)
        sidebar(signed_in).get_by_placeholder("-∞", exact=True).last.fill(lower)
        sidebar(signed_in).get_by_placeholder("∞", exact=True).last.fill(upper)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_data(signed_in, project).get("strict") is False


@pytest.mark.xfail(strict=True, reason="a ranges node made in the editor shows Strict Mode off but saves no `strict`, so decikit applies strict and rejects gaps")
def test_cases_ranges_with_manual_bounds_route_values_into_the_gap_to_otherwise(signed_in: Page):
    project = start_cases(signed_in, three_leaves(unary("root")), "Ranges")
    choose_feature(signed_in, "amount")
    for lower, upper in (("0", "10"), ("20", "30")):
        add_condition(signed_in)
        sidebar(signed_in).get_by_placeholder("-∞", exact=True).last.fill(lower)
        sidebar(signed_in).get_by_placeholder("∞", exact=True).last.fill(upper)
    assert save(signed_in)["validation_errors"] == []

    assert run(export_document(signed_in, project), amount=[5.0, 15.0, 25.0]) == ["approve", "decline", "review"]


def test_turning_strict_on_sorts_the_bands_by_upper_bound_after_confirming(signed_in: Page):
    bands = cases("root", "ranges", [{"min": 500, "max": 1000}, {"min": 0, "max": 100}], strict=False)
    project = open_tree(signed_in, three_leaves(bands))
    select_node(signed_in, "root")
    sidebar(signed_in).get_by_label("Strict Mode").click()
    signed_in.get_by_role("alertdialog").get_by_role("button", name="Enable Strict Mode").click()
    assert save(signed_in)["validation_errors"] == []

    conditions = saved_data(signed_in, project)["conditions"]
    assert [c["max"] for c in conditions] == [100, 1000]
    assert all(c.get("min") is None for c in conditions)


def test_cases_isin_routes_each_value_set_to_its_branch(signed_in: Page):
    project = start_cases(signed_in, three_leaves(unary("root")), "IsIn")
    choose_feature(signed_in, "code")
    for value in ("1", "2"):
        add_condition(signed_in)
        values = sidebar(signed_in).get_by_placeholder("Type values and press Enter").last
        values.fill(value)
        values.press("Enter")
    assert save(signed_in)["validation_errors"] == []

    assert run(export_document(signed_in, project), code=[1, 2, 3]) == ["approve", "review", "decline"]


def test_cases_string_match_routes_each_pattern_to_its_branch(signed_in: Page):
    project = start_cases(signed_in, three_leaves(unary("root")), "String Match")
    choose_feature(signed_in, "band")
    for pattern in ("gold", "silver"):
        add_condition(signed_in)
        sidebar(signed_in).get_by_placeholder("Pattern 1").last.fill(pattern)
    assert save(signed_in)["validation_errors"] == []

    assert run(export_document(signed_in, project), band=["gold", "silver", "bronze"]) == ["approve", "review", "decline"]


def test_removing_a_case_drops_its_condition(signed_in: Page):
    bands = cases("root", "ranges", [{"max": 100}, {"max": 1000}], strict=True)
    project = open_tree(signed_in, three_leaves(bands))
    select_node(signed_in, "root")
    sidebar(signed_in).get_by_title("Remove condition").first.click()
    save(signed_in)

    assert [c["max"] for c in saved_data(signed_in, project)["conditions"]] == [1000]


# ---- composite ------------------------------------------------------------------------------------------------------

def condition_card(page: Page, n: int):
    return sidebar(page).get_by_test_id("composite-condition").filter(has_text=re.compile(rf"^Condition {n}\b")).first


def pick_in_card(page: Page, card, trigger_text: str, option: str) -> None:
    card.get_by_role("combobox").filter(has_text=trigger_text).first.click()
    page.get_by_role("option", name=option, exact=True).click()


def add_composite_condition(page: Page, feature: str, op: str, value: str, n: int) -> None:
    sidebar(page).get_by_role("button", name="Add Condition").click()
    card = condition_card(page, n)
    choose_feature(page, feature, scope=card)
    if op != "≤":
        pick_in_card(page, card, "≤", op)
    card.get_by_placeholder("0", exact=True).fill(value)


def test_composite_and_requires_both_conditions(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_node_type(signed_in, "Composite")
    add_composite_condition(signed_in, "amount", "≥", "500", 1)
    add_composite_condition(signed_in, "code", "==", "1", 2)
    assert save(signed_in)["validation_errors"] == []

    data = saved_data(signed_in, project)
    assert (data["op"], len(data["conditions"])) == ("and", 2)
    assert run(export_document(signed_in, project), amount=[100.0, 900.0, 900.0], code=[1, 1, 2]) == [
        "review", "approve", "review"]


def test_composite_or_needs_one_condition(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_node_type(signed_in, "Composite")
    sidebar(signed_in).get_by_role("combobox").filter(has_text=re.compile(r"^AND$")).click()
    signed_in.get_by_role("option", name="OR", exact=True).click()
    add_composite_condition(signed_in, "amount", "≥", "500", 1)
    add_composite_condition(signed_in, "code", "==", "1", 2)
    assert save(signed_in)["validation_errors"] == []

    data = saved_data(signed_in, project)
    assert (data["op"], len(data["conditions"])) == ("or", 2)
    assert run(export_document(signed_in, project), amount=[100.0, 100.0, 900.0], code=[2, 1, 2]) == [
        "review", "approve", "approve"]


def test_composite_not_negates_its_one_condition(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_node_type(signed_in, "Composite")
    add_composite_condition(signed_in, "amount", "≥", "500", 1)
    sidebar(signed_in).get_by_role("combobox").filter(has_text=re.compile(r"^AND$")).click()
    signed_in.get_by_role("option", name="NOT", exact=True).click()
    assert save(signed_in)["validation_errors"] == []

    expect(sidebar(signed_in).get_by_role("button", name="Add Condition")).to_have_count(0)
    assert run(export_document(signed_in, project), amount=[100.0, 900.0]) == ["approve", "review"]


def test_not_is_not_offered_once_the_composite_has_two_conditions(signed_in: Page):
    open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_node_type(signed_in, "Composite")
    add_composite_condition(signed_in, "amount", "≥", "500", 1)
    add_composite_condition(signed_in, "code", "==", "1", 2)
    sidebar(signed_in).get_by_role("combobox").filter(has_text=re.compile(r"^AND$")).click()

    expect(signed_in.get_by_role("option", name="NOT", exact=True)).to_have_attribute("aria-disabled", "true")


def test_nested_or_inside_and_is_saved_nested_and_decides(signed_in: Page):
    inner = {"type": "composite", "op": "or", "conditions": [
        {"type": "unary", "op": "==", "feature": "code", "threshold": 1},
        {"type": "unary", "op": "==", "feature": "code", "threshold": 2}]}
    outer = composite("comp", "and", [{"type": "unary", "op": ">=", "feature": "amount", "threshold": 500}, inner], y=150)
    doc = document([unary("top", "<", "amount", threshold=-1e9), outer, leaf("yes", 0, 100, 350), leaf("no", 1, 500, 350),
                    leaf("never", 1, 0, 150)],
                   [edge("top", "never", 0), edge("top", "comp", 1), edge("comp", "yes", 0), edge("comp", "no", 1)],
                   input_schema=SCHEMA)
    project = open_tree(signed_in, doc)
    select_node(signed_in, "comp")
    expect(sidebar(signed_in).get_by_text("Nested 1")).to_be_visible()
    sidebar(signed_in).get_by_placeholder("0", exact=True).first.fill("600")
    assert save(signed_in)["validation_errors"] == []

    data = saved_data(signed_in, project, "comp")
    assert data["conditions"][1]["conditions"][0]["threshold"] == 1
    assert run(export_document(signed_in, project), amount=[550.0, 650.0, 650.0, 650.0], code=[1, 1, 2, 3]) == [
        "review", "approve", "approve", "review"]


def test_nested_condition_can_be_added_inside_a_composite_condition(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_node_type(signed_in, "Composite")
    sidebar(signed_in).get_by_role("button", name="Add Condition").click()
    pick_in_card(signed_in, condition_card(signed_in, 1), "≤", "OR")
    sidebar(signed_in).get_by_role("button", name="Add Nested Condition").click()
    expect(sidebar(signed_in).get_by_text("Nested 1")).to_be_visible()
    nested = sidebar(signed_in).get_by_test_id("composite-condition").filter(has_text="Nested 1").last
    choose_feature(signed_in, "amount", scope=nested)
    save(signed_in)

    data = saved_data(signed_in, project)
    assert data["conditions"][0]["op"] == "or"
    assert data["conditions"][0]["conditions"][0]["feature"] == "amount"


def test_empty_composite_shows_what_is_missing_in_the_sidebar(signed_in: Page):
    open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_node_type(signed_in, "Composite")

    expect(sidebar(signed_in).get_by_text("A composite needs at least one condition")).to_be_visible()


@pytest.mark.xfail(strict=True, reason="a document whose root is a composite node crashes the editor on open")
def test_composite_node_as_the_root_opens_without_a_page_error(signed_in: Page):
    errors = []
    signed_in.on("pageerror", lambda e: errors.append(str(e)))
    cond = [{"type": "unary", "op": ">=", "feature": "amount", "threshold": 500}]
    project = create_project(signed_in, "Composite root", two_leaves(composite("root", "and", cond)), project_type="tree")
    signed_in.goto(f"/projects/{project['id']}")

    expect(signed_in.get_by_role("button", name="Add Node")).to_be_visible()
    assert errors == []
