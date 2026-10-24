"""Node-editor outputs, input features and parameters: edits save a document decikit runs with the expected columns."""

import re

import polars as pl
import pytest
from decikit import TreeConfig
from playwright.sync_api import Locator, Page, expect

from flat_rules_ui import add_parameter
from node_ui import (
    deselect,
    document,
    edge,
    leaf,
    nudge,
    open_tree,
    output_table,
    run,
    save,
    select_node,
    sidebar,
    unary,
    use_parameter,
)
from seed import export_document, get_project

ROOT_ON_AMOUNT = {"id": "root", "position": {"x": 300, "y": 0}, "data": {
    "type": "cases", "op": "ranges", "feature": "amount", "strict": True, "conditions": [{"max": 100}, {"max": 1000}]}}


def banded(output: dict, indices=(0, 1, 2), **kwargs) -> dict:
    """amount < 100 -> leaf `low`, < 1000 -> `mid`, else `high`; the leaves point at `indices`."""
    low, mid, high = indices
    return document([ROOT_ON_AMOUNT, leaf("low", low, 0), leaf("mid", mid, 250), leaf("high", high, 500)],
                    [edge("root", "low", 0), edge("root", "mid", 1), edge("root", "high", 2)], output=output, **kwargs)


def open_output_editor(page: Page):
    page.get_by_role("button", name="Edit Output").click()
    return page.get_by_role("dialog")


def output_record(dialog: Locator, index: int) -> Locator:
    """The Edit Output row of record `index` (a leaf's result_idx); the default row comes first, on its own."""
    return dialog.locator("tbody tr:not([data-testid='output-default-row'])").nth(index)


def close_dialog(page: Page) -> None:
    page.get_by_role("button", name="Close").click()
    expect(page.get_by_role("dialog")).to_be_hidden()


def result_indices(page: Page, project: str) -> dict:
    nodes = get_project(page, project)["project_data"]["nodes"]
    return {n["id"]: n["data"]["result_idx"] for n in nodes if n["data"]["type"] == "leaf"}


def test_record_added_in_edit_output_is_saved_last_and_can_be_picked_by_a_leaf(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review"), indices=(-1, -1, -1)))
    dialog = open_output_editor(signed_in)
    dialog.get_by_role("button", name="Add Record").click()
    output_record(dialog, 2).locator("input").fill("refer")
    close_dialog(signed_in)
    select_node(signed_in, "low")
    signed_in.get_by_role("combobox").nth(1).click()
    signed_in.get_by_role("option", name=re.compile("refer")).click()
    assert save(signed_in)["validation_errors"] == []

    output = get_project(signed_in, project)["project_data"]["output"]
    assert [row["decision"] for row in output["data"]] == ["approve", "review", "refer"]
    assert result_indices(signed_in, project)["low"] == 2
    assert run(export_document(signed_in, project), amount=[50.0, 5000.0]) == ["refer", "decline"]


def test_editing_a_record_changes_what_its_leaves_return(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_output_editor(signed_in)
    output_record(dialog, 1).locator("input").fill("reviewed")
    close_dialog(signed_in)
    assert save(signed_in)["validation_errors"] == []

    assert run(export_document(signed_in, project), amount=[50.0, 500.0, 5000.0]) == ["approve", "reviewed", "refer"]


def test_saved_output_has_only_the_documented_keys(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_output_editor(signed_in)
    output_record(dialog, 1).locator("input").fill("reviewed")
    close_dialog(signed_in)
    save(signed_in)

    output = get_project(signed_in, project)["project_data"]["output"]
    assert set(output) == {"data", "default", "dtypes", "type_defs"}


def test_default_row_of_the_dialog_shows_the_output_default(signed_in: Page):
    open_tree(signed_in, banded(output_table("approve", "review", "refer"), indices=(0, 1, -1)))
    dialog = open_output_editor(signed_in)

    expect(dialog.get_by_test_id("output-default-row").locator("input")).to_have_value("decline")


def test_editing_the_first_record_leaves_the_output_default_alone(signed_in: Page):
    doc = document([unary("root"), leaf("a", 0, 100, 250)], [edge("root", "a", 0)], output=output_table("approve"))
    project = open_tree(signed_in, doc)
    dialog = open_output_editor(signed_in)
    output_record(dialog, 0).locator("input").fill("escalate")
    close_dialog(signed_in)
    save(signed_in)

    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["escalate", "decline"]


@pytest.mark.smoke
def test_deleting_a_record_above_a_leaf_keeps_the_leaf_on_the_same_record(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_output_editor(signed_in)
    output_record(dialog, 1).get_by_role("button").click()  # the "review" record
    close_dialog(signed_in)
    assert save(signed_in)["validation_errors"] == []

    assert result_indices(signed_in, project) == {"low": 0, "mid": -1, "high": 1}
    assert run(export_document(signed_in, project), amount=[50.0, 500.0, 5000.0]) == ["approve", "decline", "refer"]


def test_deleting_the_record_a_leaf_points_at_sends_it_to_the_default(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_output_editor(signed_in)
    output_record(dialog, 2).get_by_role("button").click()  # the last record
    close_dialog(signed_in)
    save(signed_in)

    assert result_indices(signed_in, project) == {"low": 0, "mid": 1, "high": -1}


def test_the_first_record_can_be_deleted_like_any_other(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_output_editor(signed_in)
    delete_first = output_record(dialog, 0).get_by_role("button")
    expect(delete_first).to_have_count(1)
    delete_first.click()
    close_dialog(signed_in)
    save(signed_in)

    assert result_indices(signed_in, project) == {"low": -1, "mid": 0, "high": 1}


def test_leaf_output_picker_lists_the_default_and_every_record(signed_in: Page):
    open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    select_node(signed_in, "mid")
    signed_in.get_by_role("combobox").nth(1).click()

    options = signed_in.get_by_role("option")
    expect(options).to_have_count(4)
    expect(options.nth(0)).to_contain_text("decline")


def test_editing_a_leaf_output_to_a_new_record_adds_it_and_repoints_the_leaf(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    select_node(signed_in, "low")
    signed_in.get_by_title("Edit output values").click()
    signed_in.locator("input").last.fill("fast-track")
    signed_in.get_by_role("button", name="Save as New").click()
    assert save(signed_in)["validation_errors"] == []

    assert result_indices(signed_in, project)["low"] == 3
    assert run(export_document(signed_in, project), amount=[50.0]) == ["fast-track"]


# ---- output column types --------------------------------------------------------------------------------------------

TYPES = [
    ("Int64", 3, 7), ("Float64", 2.5, 9.25), ("Boolean", True, False), ("String", "x", "y"),
]


@pytest.mark.parametrize("dtype, first, second", TYPES)
def test_output_column_of_a_basic_type_survives_an_edit_and_exports_typed(signed_in: Page, dtype, first, second):
    output = {"dtypes": [["decision", "String"], ["value", dtype]], "type_defs": {},
              "data": [{"decision": "approve", "value": first}, {"decision": "review", "value": second}],
              "default": {"decision": "decline", "value": second}}
    project = open_tree(signed_in, banded(output, indices=(0, 1, -1)))
    nudge(signed_in, "low")
    assert save(signed_in)["validation_errors"] == []

    tree = TreeConfig.load({"type": "tree", "name": "t", "tree": export_document(signed_in, project)})
    out = tree.run(pl.DataFrame({"amount": [50.0, 500.0, 5000.0]}))
    assert out["value"].to_list() == [first, second, second]


def test_new_output_column_is_added_with_a_type_and_its_values_are_exported(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_output_editor(signed_in)
    dialog.get_by_role("button", name="Add Field").click()
    dialog.get_by_placeholder("Column name").fill("limit")
    dialog.get_by_role("combobox").first.click()  # the new column's type; the table headers come after
    signed_in.get_by_role("option", name="Int64", exact=True).click()
    dialog.get_by_role("button", name="Add", exact=True).click()
    for row, value in enumerate(("10", "20", "30")):
        output_record(dialog, row).locator("input").nth(1).fill(value)
    close_dialog(signed_in)
    assert save(signed_in)["validation_errors"] == []

    tree = TreeConfig.load({"type": "tree", "name": "t", "tree": export_document(signed_in, project)})
    out = tree.run(pl.DataFrame({"amount": [50.0, 500.0]}))
    assert out["limit"].to_list() == [10, 20]


CATEGORICAL = {"tier": {"name": "Tier", "type": "categorical", "definition": {"categories": ["gold", "silver"]}}}
STRUCT = {"act": {"name": "Act", "type": "struct", "definition": {
    "fields": [["Outcome", "String"], ["Code", "Int64"]], "display_field": "Outcome",
    "data": [{"Outcome": "Approve", "Code": 1}, {"Outcome": "Decline", "Code": 2}]}}}


def test_custom_categorical_output_column_survives_an_edit_and_exports_as_text(signed_in: Page):
    output = {"dtypes": [["tier", {"type": "Custom", "type_id": "tier"}]], "type_defs": CATEGORICAL,
              "data": [{"tier": "gold"}, {"tier": "silver"}], "default": {"tier": "silver"}}
    project = open_tree(signed_in, banded(output, indices=(0, 1, -1)))
    nudge(signed_in, "low")
    assert save(signed_in)["validation_errors"] == []

    tree = TreeConfig.load({"type": "tree", "name": "t", "tree": export_document(signed_in, project)})
    assert tree.run(pl.DataFrame({"amount": [50.0, 500.0, 5000.0]}))["tier"].to_list() == ["gold", "silver", "silver"]


def test_custom_struct_output_column_survives_an_edit_and_exports_one_column_per_field(signed_in: Page):
    output = {"dtypes": [["Action", {"type": "Custom", "type_id": "act"}]], "type_defs": STRUCT,
              "data": [{"Action": {"$key": "Approve"}}, {"Action": {"$key": "Decline"}}], "default": {"Action": None}}
    project = open_tree(signed_in, banded(output, indices=(0, 1, -1)))
    nudge(signed_in, "low")
    assert save(signed_in)["validation_errors"] == []

    tree = TreeConfig.load({"type": "tree", "name": "t", "tree": export_document(signed_in, project)})
    out = tree.run(pl.DataFrame({"amount": [50.0, 500.0]}))
    assert (out["Action.Outcome"].to_list(), out["Action.Code"].to_list()) == (["Approve", "Decline"], [1, 2])


def test_list_output_column_survives_an_edit_and_exports_a_list(signed_in: Page):
    output = {"dtypes": [["tags", {"type": "List", "inner": "String"}]], "type_defs": {},
              "data": [{"tags": ["a", "b"]}, {"tags": ["c"]}], "default": {"tags": []}}
    project = open_tree(signed_in, banded(output, indices=(0, 1, -1)))
    nudge(signed_in, "low")
    assert save(signed_in)["validation_errors"] == []

    tree = TreeConfig.load({"type": "tree", "name": "t", "tree": export_document(signed_in, project)})
    assert tree.run(pl.DataFrame({"amount": [50.0, 500.0]}))["tags"].to_list() == [["a", "b"], ["c"]]


# ---- input features -------------------------------------------------------------------------------------------------

def open_features(page: Page):
    page.get_by_role("button", name="Edit Features").click()
    return page.get_by_role("dialog")


def test_feature_added_in_the_features_dialog_is_saved_in_the_input_schema(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_features(signed_in)
    dialog.get_by_role("button", name="Add Feature").first.click()
    dialog.get_by_placeholder("Field name").last.fill("income")
    signed_in.keyboard.press("Escape")
    save(signed_in)

    names = [name for name, _ in get_project(signed_in, project)["project_data"]["input_schema"]]
    assert names == ["amount", "band", "code", "flag", "income"]


def test_feature_picked_in_a_node_is_added_to_the_input_schema_when_it_is_new(signed_in: Page):
    doc = document([unary("root", feature="")], [], input_schema=[["amount", "Float64"]])
    project = open_tree(signed_in, doc)
    select_node(signed_in, "root")
    sidebar(signed_in).get_by_role("combobox").filter(has_text="Select or search feature").click()
    signed_in.get_by_placeholder("Search or create feature...").fill("income")
    signed_in.get_by_role("option", name='Create "income"').click()
    signed_in.get_by_role("button", name="Create Feature").click()
    save(signed_in)

    names = [name for name, _ in get_project(signed_in, project)["project_data"]["input_schema"]]
    assert "income" in names


def test_deleting_a_feature_a_cases_node_uses_is_refused_or_reported_when_saved(signed_in: Page):
    project = open_tree(signed_in, banded(output_table("approve", "review", "refer")))
    dialog = open_features(signed_in)
    dialog.locator("input[placeholder='Field name']").first.locator("xpath=ancestor::div[1]/..").get_by_role("button").last.click()
    signed_in.keyboard.press("Escape")
    reply = save(signed_in)

    data = next(n for n in get_project(signed_in, project)["project_data"]["nodes"] if n["id"] == "root")["data"]
    assert data["feature"] == "amount" or reply["validation_errors"]


# ---- parameters -----------------------------------------------------------------------------------------------------

def test_threshold_from_a_new_parameter_saves_valid_and_runs_with_its_default(signed_in: Page):
    doc = document([unary("root"), leaf("yes", 0, 100, 250), leaf("no", 1, 500, 250)],
                   [edge("root", "yes", 0), edge("root", "no", 1)])
    project = open_tree(signed_in, doc)
    add_parameter(signed_in, "limit", "Float64", "1000")
    select_node(signed_in, "root")
    use_parameter(signed_in, "limit")
    assert save(signed_in)["validation_errors"] == []

    exported = export_document(signed_in, project)
    assert exported["nodes"][0]["data"]["condition"]["threshold"] == {"param": "limit"}
    assert exported["parameters"]["limit"]["default_value"] == 1000
    assert run(exported, amount=[500.0, 2000.0]) == ["approve", "review"]


@pytest.mark.parametrize("dtype, value, expected", [("Int64", "1000", 1000), ("Float64", "999.5", 999.5)])
def test_parameter_default_of_a_numeric_type_is_saved_with_that_type(signed_in: Page, dtype, value, expected):
    project = open_tree(signed_in, document([unary("root")], []))
    add_parameter(signed_in, "limit", dtype, value)
    save_reply = save(signed_in)

    assert save_reply["validation_errors"] == []
    assert get_project(signed_in, project)["project_data"]["parameters"]["limit"] == {"type": dtype, "default_value": expected}


def test_one_parameter_can_drive_two_nodes_and_is_marked_in_use(signed_in: Page):
    doc = document(
        [unary("root", threshold=1), unary("second", x=600, y=150, threshold=1), leaf("a", 0, 0, 300), leaf("b", 1, 300, 300),
         leaf("c", 0, 600, 300)],
        [edge("root", "a", 0), edge("root", "second", 1), edge("second", "b", 0), edge("second", "c", 1)])
    project = open_tree(signed_in, doc)
    add_parameter(signed_in, "limit", "Float64", "1000")
    for node_id in ("root", "second"):
        select_node(signed_in, node_id)
        use_parameter(signed_in, "limit")
    assert save(signed_in)["validation_errors"] == []

    deselect(signed_in)
    signed_in.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    dialog = signed_in.get_by_role("dialog", name="Manage Variables")
    expect(dialog.get_by_text("Yes", exact=True)).to_be_visible()
    expect(dialog.get_by_title("Cannot delete parameter that is in use")).to_be_disabled()
    expect(dialog.get_by_title("Cannot rename parameter that is in use")).to_be_disabled()
    thresholds = [n["data"]["condition"]["threshold"] for n in get_project(signed_in, project)["project_data"]["nodes"]
                  if n["data"]["type"] == "unary"]
    assert thresholds == [{"param": "limit"}] * 2


def test_unused_parameter_can_be_renamed_and_deleted(signed_in: Page):
    project = open_tree(signed_in, document([unary("root")], [], parameters={
        "old": {"type": "Float64", "default_value": 1}, "spare": {"type": "Float64", "default_value": 2}}))
    signed_in.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    dialog = signed_in.get_by_role("dialog", name="Manage Variables")
    dialog.get_by_title("Parameter name").first.fill("renamed")
    dialog.get_by_title("Delete parameter").last.click()
    dialog.get_by_role("button", name="Save Changes").click()
    save(signed_in)

    assert list(get_project(signed_in, project)["project_data"]["parameters"]) == ["renamed"]


def test_parameter_name_with_a_space_is_refused_in_the_dialog(signed_in: Page):
    open_tree(signed_in, document([unary("root")], []))
    signed_in.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    dialog = signed_in.get_by_role("dialog", name="Manage Variables")
    dialog.get_by_role("button", name="Add Your First Parameter").click()
    dialog.get_by_placeholder("parameter_name").fill("my limit")
    dialog.get_by_role("button", name="Add", exact=True).click()

    expect(dialog.locator("tbody tr").filter(has=signed_in.get_by_placeholder("parameter_name"))).to_be_visible()
    expect(dialog.get_by_text('"my limit" is not a valid name')).to_be_visible()
