"""Node-editor validation: an invalid tree is flagged on its node after Save, named in a message, refused on Export, and fixed."""

import json
import re

import pytest
from playwright.sync_api import Page, expect

from node_ui import (
    composite,
    document,
    edge,
    leaf,
    loan_document,
    node,
    nudge,
    open_menu,
    open_tree,
    run,
    save,
    settle,
    select_node,
    set_node_type,
    set_operator,
    set_threshold,
    sidebar,
    unary,
)
from seed import create_project, get_project


def export_from_menu(page: Page, tmp_path) -> dict:
    open_menu(page)
    with page.expect_download() as download:
        page.get_by_role("menuitem", name="Export Project").click()
    path = tmp_path / "export.json"
    download.value.save_as(path)
    return json.loads(path.read_text())


def toast(page: Page, title: str):
    return page.locator("li[data-state=open]").filter(has_text=title)


def marker(page: Page, node_id: str):
    """The red "!" badge a node with a validation error carries."""
    return node(page, node_id).get_by_text("!", exact=True)


def refused_export_says(page: Page) -> str:
    """Click Export Project and return the text of the 'Export failed' toast."""
    open_menu(page)
    page.get_by_role("menuitem", name="Export Project").click()
    failed = toast(page, "Export failed")
    expect(failed).to_be_visible()
    return failed.inner_text()


def invalid_via_ui(page: Page, how: str) -> str:
    """Open the loan tree and make `root` invalid the way a user would; returns the node id."""
    open_tree(page, loan_document())
    select_node(page, "root")
    if how == "between_without_bounds":
        set_operator(page, "Between")
    elif how == "empty_composite":
        set_node_type(page, "Composite")
    elif how == "unfinished_parameter":
        sidebar(page).get_by_role("button", name="#").first.click()
    return "root"


CASES = {
    "between_without_bounds": r"min or max",
    "empty_composite": r"at least one condition",
    "unfinished_parameter": r"param",
}
SMOKE_CASE = pytest.param("between_without_bounds", marks=pytest.mark.smoke)


@pytest.mark.parametrize("how", [SMOKE_CASE, *list(CASES)[1:]])
def test_invalid_node_is_marked_named_in_a_message_and_blocks_export(signed_in: Page, how):
    node_id = invalid_via_ui(signed_in, how)
    reply = save(signed_in)

    assert reply["validation_errors"], "decikit accepted a node the editor leaves unfinished"
    expect(toast(signed_in, "Saved with validation issues")).to_be_visible()
    message = toast(signed_in, "Saved with validation issues").inner_text()
    assert f'"{node_id}"' in message, message
    assert re.search(CASES[how], message, re.I), message
    expect(marker(signed_in, node_id)).to_be_visible()
    assert re.search(CASES[how], refused_export_says(signed_in), re.I)


@pytest.mark.xfail(strict=True, reason="a unary node with no feature chosen saves and exports without any message")
def test_unary_node_without_a_feature_is_flagged(signed_in: Page):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    set_node_type(signed_in, "Leaf")
    set_node_type(signed_in, "Unary")
    reply = save(signed_in)

    assert reply["validation_errors"]
    expect(marker(signed_in, "root")).to_be_visible()


@pytest.mark.xfail(strict=True, reason="the message for an unfinished parameter ends 'rename it, e.g. \'\'' and never says a name is missing")
def test_unfinished_parameter_message_says_a_name_is_missing(signed_in: Page):
    invalid_via_ui(signed_in, "unfinished_parameter")
    save(signed_in)
    message = toast(signed_in, "Saved with validation issues").inner_text()

    assert "e.g. ''" not in message, message


def test_fixing_the_invalid_node_clears_the_marker_and_the_export_works(signed_in: Page, tmp_path):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    set_operator(signed_in, "Between")
    save(signed_in)
    expect(marker(signed_in, "root")).to_be_visible()

    sidebar(signed_in).get_by_placeholder("null").nth(1).fill("1000")
    assert save(signed_in)["validation_errors"] == []
    expect(marker(signed_in, "root")).to_be_hidden()
    document_ = export_from_menu(signed_in, tmp_path)
    assert run(document_, amount=[500.0, 2000.0]) == ["approve", "review"]


def test_marker_stays_on_the_node_after_reload_until_it_is_fixed(signed_in: Page):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    set_operator(signed_in, "Between")
    save(signed_in)
    signed_in.reload()

    expect(signed_in.locator(".react-flow__node")).to_have_count(3)
    expect(marker(signed_in, "root")).to_be_visible()
    expect(marker(signed_in, "yes")).to_be_hidden()


def test_only_the_invalid_node_is_marked_when_two_nodes_differ(signed_in: Page):
    doc = document(
        [unary("root"), composite("bad", "and", [], x=600, y=150), leaf("a", 0, 100, 300), leaf("b", 1, 500, 300),
         leaf("c", 0, 800, 300)],
        [edge("root", "a", 0), edge("root", "bad", 1), edge("bad", "b", 0), edge("bad", "c", 1)])
    open_tree(signed_in, doc)
    nudge(signed_in, "a")
    save(signed_in)

    expect(marker(signed_in, "bad")).to_be_visible()
    expect(marker(signed_in, "root")).to_be_hidden()


def test_invalid_computed_feature_is_named_with_the_expression(signed_in: Page):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    sidebar(signed_in).get_by_title("Switch to computed feature").click()
    sidebar(signed_in).get_by_placeholder("feature or expression (e.g., feature1 + feature2)").fill("amount +")
    reply = save(signed_in)

    assert reply["validation_errors"]
    message = toast(signed_in, "Saved with validation issues").inner_text()
    assert '"root"' in message and "expression" in message, message


@pytest.mark.xfail(strict=True, reason="a threshold naming a parameter that is not defined saves with no message")
def test_parameter_that_is_not_defined_is_named_in_the_message(signed_in: Page):
    doc = loan_document()
    doc["nodes"][0]["data"]["condition"]["threshold"] = {"param": "ghost"}
    project = create_project(signed_in, "Ghost", doc, project_type="tree")

    assert any("ghost" in e["msg"] for e in project["validation_errors"]), project["validation_errors"]


def test_a_cycle_is_reported_when_saving(signed_in: Page):
    doc = document([unary("a"), unary("b", y=150)], [edge("a", "b", 0), edge("b", "a", 0)])
    project = create_project(signed_in, "Cycle", doc, project_type="tree")

    assert any("cycle" in e["msg"].lower() for e in project["validation_errors"]), project["validation_errors"]


def test_a_cycle_below_the_root_is_reported_and_export_is_refused(signed_in: Page):
    doc = document([unary("root"), unary("a", y=150), unary("b", y=300), leaf("l", 0, 0, 450)],
                   [edge("root", "a", 0), edge("a", "b", 0), edge("b", "a", 0)])
    project = create_project(signed_in, "Cycle", doc, project_type="tree")

    assert project["validation_errors"], "a cycle below the root saved without a message"
    assert signed_in.request.get(f"/api/v1/projects/{project['id']}/export").status >= 400


def test_leaf_pointing_past_the_output_records_is_reported_with_its_node(signed_in: Page):
    doc = loan_document()
    doc["nodes"][1]["data"]["result_idx"] = 7
    project = create_project(signed_in, "Past the end", doc, project_type="tree")
    signed_in.goto(f"/projects/{project['id']}")
    expect(signed_in.locator(".react-flow__node")).to_have_count(3)

    assert project["validation_errors"], "a leaf pointing at record 7 of 2 saved without a message"
    expect(node(signed_in, "yes")).to_contain_text("Invalid: 7")


# ---- editor behaviour: unsaved changes, leaving, export ---------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="a tree with no saved subtree entries opens marked 'Unsaved changes' before any edit")
def test_clean_tree_has_no_unsaved_indicator_and_save_is_disabled(signed_in: Page):
    open_tree(signed_in, loan_document())
    signed_in.wait_for_load_state("networkidle")

    expect(signed_in.get_by_text("Unsaved changes")).to_be_hidden()
    expect(signed_in.get_by_role("button", name="Save", exact=True)).to_be_disabled()


def test_an_edit_shows_unsaved_changes_and_saving_clears_it(signed_in: Page):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    set_threshold(signed_in, "2000")
    expect(signed_in.get_by_text("Unsaved changes")).to_be_visible()
    expect(signed_in.get_by_role("button", name="Save", exact=True)).to_be_enabled()
    save(signed_in)

    expect(signed_in.get_by_role("button", name="Save", exact=True)).to_be_disabled()


def test_back_with_unsaved_edits_asks_first_and_staying_keeps_the_edits(signed_in: Page):
    page = signed_in.context.new_page()
    open_tree(page, loan_document())
    select_node(page, "root")
    set_threshold(page, "2000")
    messages = []
    page.once("dialog", lambda d: (messages.append(d.message), d.dismiss()))
    page.get_by_role("button", name="Back").click()

    assert messages and "unsaved" in messages[0].lower()
    expect(page.get_by_text("Unsaved changes")).to_be_visible()
    expect(page).to_have_url(re.compile(r"/projects/[0-9a-f-]{36}$"))


def test_back_with_unsaved_edits_leaves_without_saving_when_confirmed(signed_in: Page):
    page = signed_in.context.new_page()
    project = open_tree(page, loan_document())
    select_node(page, "root")
    set_threshold(page, "2000")
    page.once("dialog", lambda d: d.accept())
    page.get_by_role("button", name="Back").click()

    expect(page).to_have_url(re.compile(r"/projects$"))
    condition = next(n for n in get_project(page, project)["project_data"]["nodes"] if n["id"] == "root")["data"]["condition"]
    assert condition["threshold"] == 1000


@pytest.mark.xfail(strict=True, reason="browser Back from the editor with unsaved edits leaves the editor even when its prompt is dismissed")
def test_browser_back_with_unsaved_edits_asks_and_staying_keeps_the_editor(signed_in: Page):
    page = signed_in.context.new_page()
    create_project(page, "Back button", loan_document(), project_type="tree")
    page.goto("/projects")
    page.get_by_role("button", name="Open Project").click()  # a client-side navigation, as a user makes it
    expect(page.get_by_role("button", name="Add Node")).to_be_visible()
    select_node(page, "root")
    set_threshold(page, "2000")
    messages = []
    page.once("dialog", lambda d: (messages.append(d.message), d.dismiss()))
    page.evaluate("() => history.back()")

    expect(page.get_by_text("Unsaved changes")).to_be_visible()
    assert messages and "unsaved" in messages[0].lower()


def test_unsaved_edits_install_a_leave_page_prompt(signed_in: Page):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    set_threshold(signed_in, "2000")

    prompted = signed_in.evaluate("() => { const e = new Event('beforeunload', {cancelable: true}); "
                                  "window.dispatchEvent(e); return e.defaultPrevented; }")
    assert prompted


def test_a_saved_tree_installs_no_leave_page_prompt(signed_in: Page):
    open_tree(signed_in, loan_document())
    settle(signed_in)

    prompted = signed_in.evaluate("() => { const e = new Event('beforeunload', {cancelable: true}); "
                                  "window.dispatchEvent(e); return e.defaultPrevented; }")
    assert not prompted


def test_export_from_the_menu_includes_unsaved_edits(signed_in: Page, tmp_path):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    set_threshold(signed_in, "2000")
    exported = export_from_menu(signed_in, tmp_path)

    assert exported["nodes"][0]["data"]["condition"]["threshold"] == 2000
    assert run(exported, amount=[1500.0, 2500.0]) == ["approve", "review"]


def test_importing_the_same_file_twice_makes_two_independent_projects(signed_in: Page, tmp_path):
    exported = tmp_path / "tree.json"
    exported.write_text(json.dumps(loan_document()))
    open_tree(signed_in, loan_document())
    settle(signed_in)
    signed_in.get_by_role("button", name="Back").click()
    for name in ("Copy one", "Copy two"):
        signed_in.get_by_role("button", name="Import Project").click()
        signed_in.locator("input[type=file]").set_input_files(exported)
        signed_in.get_by_placeholder("Enter project name").fill(name)
        signed_in.get_by_role("dialog").locator("button[type=submit]").click()
        expect(signed_in.get_by_role("dialog")).to_be_hidden()
        signed_in.goto("/projects")  # the import opens the new project

    expect(signed_in.get_by_text("Copy one", exact=True)).to_be_visible()
    expect(signed_in.get_by_text("Copy two", exact=True)).to_be_visible()
