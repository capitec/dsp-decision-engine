"""Custom output types and parameters: create, edit, delete, the usage checks, and what decikit then runs."""

import re

import pytest
from playwright.sync_api import Page, expect

from flat_docs import APPROVE, DEFAULT, open_doc, run_frame
from flat_rules_ui import (
    add_categorical_type,
    add_output_column,
    open_custom_types,
    open_tab,
    output_cell,
    rule_card,
    save,
)
from seed import export_document

LIMIT = {"limit": {"type": "Float64", "default_value": 1000}}


def type_defs(reply: dict) -> dict:
    return reply["project_data"]["output"]["type_defs"]


def params_dialog(page: Page):
    page.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    return page.get_by_role("dialog", name="Manage Parameters")


def test_categorical_type_is_added_and_saved_valid(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold", "silver"])
    expect(dialog.get_by_role("heading", name="Tier")).to_be_visible()
    dialog.get_by_role("button", name="Close").click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    (definition,) = type_defs(reply).values()
    assert (definition["name"], definition["type"]) == ("Tier", "categorical")
    assert definition["definition"]["categories"] == ["gold", "silver"]


def test_column_of_a_categorical_type_saves_valid_and_runs(signed_in: Page):
    project = open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold", "silver"])
    signed_in.get_by_role("dialog").get_by_role("button", name="Close").click()
    add_output_column(signed_in, "tier", "Tier")
    output_cell(signed_in, 0, 1).click()
    signed_in.get_by_role("option", name="silver", exact=True).click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert reply["project_data"]["output"]["data"][0]["tier"] == "silver"
    assert run_frame(export_document(signed_in, project), {"amount": [1.0]})[0]["tier"] == "silver"


def test_struct_type_is_added_and_saved_valid(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    dialog.get_by_role("button", name="New Struct").first.click()
    dialog.get_by_placeholder("e.g., ActionType, OutcomeCategories").fill("Offer")
    dialog.get_by_placeholder("Enter id...").fill("basic")
    dialog.get_by_role("button", name="Save Type").click()
    expect(dialog.get_by_role("heading", name="Offer")).to_be_visible()
    dialog.get_by_role("button", name="Close").click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    (definition,) = type_defs(reply).values()
    assert (definition["name"], definition["type"]) == ("Offer", "struct")


def test_edited_type_name_is_saved(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold"])
    dialog.get_by_role("heading", name="Tier").click()
    dialog.get_by_placeholder("e.g., ActionType, OutcomeCategories").fill("Level")
    dialog.get_by_role("button", name="Save Type").click()
    dialog.get_by_role("button", name="Close").click()

    (definition,) = type_defs(save(signed_in)).values()
    assert definition["name"] == "Level"


def test_a_type_with_a_duplicate_name_is_refused_with_a_message(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold"])
    add_categorical_type(signed_in, "Tier", ["silver"])

    expect(dialog.get_by_text('A type with the name "Tier" already exists')).to_be_visible()


def test_unused_type_can_be_deleted(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold"])
    dialog.locator(".grid .cursor-pointer").filter(has_text="Tier").get_by_role("button").last.click()
    dialog.get_by_role("button", name="Close").click()

    assert type_defs(save(signed_in)) == {}


@pytest.mark.xfail(strict=True, reason="a type that a column uses is deleted without any warning, leaving the column pointing at nothing")
def test_deleting_a_type_a_column_uses_is_blocked_or_warned(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold"])
    dialog.get_by_role("button", name="Close").click()
    add_output_column(signed_in, "tier", "Tier")
    dialog = open_custom_types(signed_in)
    messages: list[str] = []
    signed_in.on("dialog", lambda d: messages.append(d.message))
    dialog.locator(".grid .cursor-pointer").filter(has_text="Tier").get_by_role("button").last.click()

    assert any("used" in m or "column" in m for m in messages) or dialog.get_by_text(re.compile("used|column")).count()


def test_removing_a_category_a_record_uses_shows_the_delete_impact_and_cleans_up(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold", "silver"])
    dialog.get_by_role("button", name="Close").click()
    add_output_column(signed_in, "tier", "Tier")
    output_cell(signed_in, 0, 1).click()
    signed_in.get_by_role("option", name="silver", exact=True).click()
    dialog = open_custom_types(signed_in)
    dialog.get_by_role("heading", name="Tier").click()
    dialog.get_by_text("silver", exact=True).locator("xpath=following-sibling::*[1]").click()
    dialog.get_by_role("button", name="Save Type").click()

    impact = signed_in.get_by_role("dialog").filter(has_text="Delete Category?")
    expect(impact).to_contain_text("silver")
    expect(impact).to_contain_text("tier")
    impact.get_by_role("button", name="Delete and Clean Up").click()
    dialog.get_by_role("button", name="Close").click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert reply["project_data"]["output"]["data"][0]["tier"] == ""
    assert next(iter(type_defs(reply).values()))["definition"]["categories"] == ["gold"]


def test_cancelling_the_delete_impact_keeps_the_category(signed_in: Page):
    open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    dialog = open_custom_types(signed_in)
    add_categorical_type(signed_in, "Tier", ["gold", "silver"])
    dialog.get_by_role("button", name="Close").click()
    add_output_column(signed_in, "tier", "Tier")
    output_cell(signed_in, 0, 1).click()
    signed_in.get_by_role("option", name="silver", exact=True).click()
    dialog = open_custom_types(signed_in)
    dialog.get_by_role("heading", name="Tier").click()
    dialog.get_by_text("silver", exact=True).locator("xpath=following-sibling::*[1]").click()
    dialog.get_by_role("button", name="Save Type").click()
    signed_in.get_by_role("dialog").filter(has_text="Delete Category?").get_by_role("button", name="Cancel").click()
    dialog.get_by_role("button", name="Cancel").first.click()
    dialog.get_by_role("button", name="Close").click()

    reply = save(signed_in)
    assert reply["project_data"]["output"]["data"][0]["tier"] == "silver"


def test_parameter_dialog_offers_the_same_types_the_document_accepts(signed_in: Page):
    open_doc(signed_in, DEFAULT)
    dialog = params_dialog(signed_in)
    dialog.get_by_role("button", name="Add Your First Parameter").click()
    dialog.locator("tbody tr").get_by_role("combobox").click()
    offered = signed_in.get_by_role("option").all_inner_texts()

    assert {"String", "Int64", "Float64", "Boolean", "Date", "Datetime"} <= set(offered)


def uses_limit(op: str = "<=") -> dict:
    return {"type": "unary", "then": APPROVE,
            "condition": {"type": "unary", "op": op, "feature": "amount", "threshold": {"param": "limit"}}}


def first_param_row(dialog):
    return dialog.locator("tbody tr").first


def test_unused_parameter_can_be_renamed_and_the_new_name_is_saved(signed_in: Page):
    open_doc(signed_in, DEFAULT, parameters=LIMIT)
    dialog = params_dialog(signed_in)
    first_param_row(dialog).get_by_role("textbox").first.fill("cap")
    dialog.get_by_role("button", name="Save Changes").click()

    assert list(save(signed_in)["project_data"]["parameters"]) == ["cap"]


def test_unused_parameter_can_be_deleted_and_the_deletion_is_saved(signed_in: Page):
    open_doc(signed_in, DEFAULT, parameters=LIMIT)
    dialog = params_dialog(signed_in)
    first_param_row(dialog).get_by_title("Delete parameter").click()
    dialog.get_by_role("button", name="Save Changes").click()

    assert save(signed_in)["project_data"]["parameters"] == {}


def test_parameter_in_use_is_marked_and_cannot_be_renamed_or_deleted(signed_in: Page):
    open_doc(signed_in, uses_limit(), parameters=LIMIT)
    dialog = params_dialog(signed_in)
    row = first_param_row(dialog)

    expect(row.get_by_text("Yes")).to_be_visible()
    expect(row.get_by_role("textbox").first).to_be_disabled()
    expect(row.get_by_title("Cannot delete parameter that is in use")).to_be_disabled()
    expect(row.get_by_title("Cannot rename parameter that is in use")).to_be_visible()


def test_parameter_stays_in_use_until_every_use_is_gone(signed_in: Page):
    open_doc(signed_in, uses_limit(), uses_limit(">"), parameters=LIMIT)
    # switch rule 1's threshold back to a plain number
    rule_card(signed_in, 0).get_by_role("button", name="#").first.click()
    dialog = params_dialog(signed_in)
    expect(first_param_row(dialog).get_by_text("Yes")).to_be_visible()
    dialog.get_by_role("button", name="Cancel").click()

    rule_card(signed_in, 1).get_by_role("button", name="#").first.click()
    dialog = params_dialog(signed_in)
    expect(first_param_row(dialog).get_by_text("Yes")).to_have_count(0)
    expect(first_param_row(dialog).get_by_title("Delete parameter")).to_be_enabled()


@pytest.mark.parametrize("name, message", [
    ("", "Parameter name is required"),
    ("1limit", "not a valid name"),
    ("max limit", "not a valid name"),
    ("class", "Python keyword"),
])
def test_invalid_new_parameter_name_is_refused_in_the_dialog_with_a_reason(signed_in: Page, name, message):
    open_doc(signed_in, DEFAULT)
    dialog = params_dialog(signed_in)
    dialog.get_by_role("button", name="Add Your First Parameter").click()
    dialog.get_by_placeholder("parameter_name").fill(name)
    dialog.get_by_role("button", name="Add", exact=True).click()

    expect(dialog).to_contain_text(message)


def test_duplicate_parameter_name_is_refused_in_the_dialog(signed_in: Page):
    open_doc(signed_in, DEFAULT, parameters=LIMIT)
    dialog = params_dialog(signed_in)
    dialog.get_by_role("button", name="Add Parameter").click()
    dialog.get_by_placeholder("parameter_name").fill("limit")
    dialog.get_by_role("button", name="Add", exact=True).click()

    expect(dialog).to_contain_text("Parameter name must be unique")


OFFERED_TYPES = ["String", "Int64", "Float64", "Boolean", "Date", "Datetime", "Int8", "Int16", "Int32", "Int128", "UInt8",
                 "UInt16", "UInt32", "UInt64", "UInt128", "Float16", "Float32", "Binary", "Null", "Time", "Duration",
                 "Decimal", "Categorical", "List", "Set", "Array"]


def test_every_type_the_parameter_dialog_offers_saves_a_valid_document(signed_in: Page):
    open_doc(signed_in, DEFAULT)
    dialog = params_dialog(signed_in)
    dialog.get_by_role("button", name="Add Your First Parameter").click()
    for index, dtype in enumerate(OFFERED_TYPES):
        if index:
            dialog.get_by_role("button", name="Add Parameter").click()
        row = dialog.locator("tbody tr").filter(has=signed_in.get_by_placeholder("parameter_name"))
        row.get_by_placeholder("parameter_name").fill(f"p_{index}")
        if dtype != "String":
            row.get_by_role("combobox").first.click()
            signed_in.get_by_role("option", name=dtype, exact=True).click()
        row.get_by_role("button", name="Add", exact=True).click()
    dialog.get_by_role("button", name="Save Changes").click()

    reply = save(signed_in)
    offered = {name: p["type"] for name, p in reply["project_data"]["parameters"].items()}
    assert len(offered) == len(OFFERED_TYPES)
    bad = [e["msg"] for e in reply["validation_errors"] if "parameters" in e["loc"]]
    assert bad == [], f"types saved as {offered}"
