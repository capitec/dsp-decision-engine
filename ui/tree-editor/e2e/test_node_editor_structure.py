"""Node-editor structure: adding, connecting, deleting nodes and edges, shared nodes, orphans, cycles, subtrees."""

import pytest
from playwright.sync_api import Page, expect

from node_ui import (
    cases,
    click_edge,
    connect,
    delete_selected,
    document,
    drag_node,
    edge,
    edge_between,
    edge_labels,
    leaf,
    loan_document,
    node,
    nudge,
    open_menu,
    open_tree,
    run,
    save,
    select_node,
    unary,
)
from seed import export_document, get_project
from tree_ui import create_tree_project

NODES = ".react-flow__node"
EDGES = ".react-flow__edge"


def saved_edges(page: Page, project_id: str) -> dict:
    """The saved edges as {source:target: sourceIndex}."""
    edges = get_project(page, project_id)["project_data"]["edges"]
    return {f"{e['source']}:{e['target']}": e["data"]["sourceIndex"] for e in edges}


def unconnected(*extra: dict) -> dict:
    """A unary root and two leaves (approve, review) with no edges."""
    return document([unary("root"), leaf("a", 0, 100, 250), leaf("b", 1, 500, 250), *extra], [])


def test_added_node_is_a_leaf_on_the_default_output(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    signed_in.get_by_role("button", name="Add Node").click()
    expect(signed_in.locator(NODES)).to_have_count(4)
    save(signed_in)

    nodes = get_project(signed_in, project)["project_data"]["nodes"]
    added = [n for n in nodes if n["id"] not in ("root", "yes", "no")]
    assert [n["data"] for n in added] == [{"type": "leaf", "result_idx": -1}]


@pytest.mark.smoke
def test_first_edge_is_the_branch_where_the_condition_holds_and_second_is_otherwise(signed_in: Page):
    project = open_tree(signed_in, unconnected())
    connect(signed_in, "root", "a")
    connect(signed_in, "root", "b")
    expect(signed_in.locator(EDGES)).to_have_count(2)
    expect(edge_labels(signed_in)).to_contain_text("≤ 1000")
    expect(edge_labels(signed_in)).to_contain_text("otherwise")
    assert save(signed_in)["validation_errors"] == []

    assert saved_edges(signed_in, project) == {"root:a": [0], "root:b": [1]}
    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["approve", "review"]


def test_connecting_a_cases_node_gives_one_branch_per_band_then_otherwise(signed_in: Page):
    bands = cases("bands", "ranges", [{"max": 100}, {"max": 1000}], strict=True)
    doc = document([bands, leaf("low", 0, 0), leaf("mid", 1, 250), leaf("high", -1, 500)], [])
    project = open_tree(signed_in, doc)
    for target in ("low", "mid", "high"):
        connect(signed_in, "bands", target)
    expect(signed_in.locator(EDGES)).to_have_count(3)
    assert save(signed_in)["validation_errors"] == []

    assert saved_edges(signed_in, project) == {"bands:low": [0], "bands:mid": [1], "bands:high": [2]}
    assert run(export_document(signed_in, project), amount=[50.0, 500.0, 5000.0]) == ["approve", "review", "decline"]


def test_a_node_with_a_new_parent_leaves_its_old_parent(signed_in: Page):
    doc = document([unary("root"), unary("other", x=700), leaf("a", 0, 100, 250), leaf("b", 1, 500, 250)],
                   [edge("root", "a", 0), edge("root", "b", 1)])
    project = open_tree(signed_in, doc)
    connect(signed_in, "other", "a")
    expect(edge_between(signed_in, "other", "a")).to_be_visible()
    expect(edge_between(signed_in, "root", "a")).to_have_count(0)
    save(signed_in)

    assert saved_edges(signed_in, project) == {"root:b": [1], "other:a": [0]}


def chain_with_refused_cycle(page: Page) -> str:
    doc = document([unary("root"), unary("mid", y=150), unary("low", y=300), leaf("a", 0, 0, 450)],
                   [edge("root", "mid", 0), edge("mid", "low", 0)])
    project = open_tree(page, doc)
    connect(page, "low", "root")
    connect(page, "low", "mid")
    return project


def test_a_connection_that_would_make_a_cycle_is_not_saved(signed_in: Page):
    project = chain_with_refused_cycle(signed_in)
    nudge(signed_in, "a")
    assert save(signed_in)["validation_errors"] == []

    assert saved_edges(signed_in, project) == {"root:mid": [0], "mid:low": [0]}


@pytest.mark.xfail(strict=True, reason="a connection the editor refuses (it would make a cycle) is still drawn on the canvas until reload")
def test_a_connection_that_would_make_a_cycle_is_not_drawn(signed_in: Page):
    chain_with_refused_cycle(signed_in)

    expect(signed_in.locator(EDGES)).to_have_count(2)


@pytest.mark.xfail(strict=True, reason="the Delete key does nothing on a selected node: only Backspace is bound")
def test_delete_key_also_removes_a_selected_node(signed_in: Page):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "yes")
    delete_selected(signed_in, "Delete")
    expect(signed_in.locator(NODES)).to_have_count(2)


def test_backspace_removes_a_node_and_its_edges(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    select_node(signed_in, "yes")
    delete_selected(signed_in)
    expect(signed_in.locator(NODES)).to_have_count(2)
    expect(signed_in.locator(EDGES)).to_have_count(1)
    assert save(signed_in)["validation_errors"] == []

    assert saved_edges(signed_in, project) == {"root:no": [1]}
    # the branch that lost its leaf takes the default output row
    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["decline", "review"]


def test_deleting_the_root_leaves_its_children_as_separate_trees(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    delete_selected(signed_in)
    expect(signed_in.locator(NODES)).to_have_count(2)
    expect(signed_in.locator(EDGES)).to_have_count(0)
    save(signed_in)

    data = get_project(signed_in, project)["project_data"]
    assert sorted(n["id"] for n in data["nodes"]) == ["no", "yes"]
    assert data["edges"] == []


def test_delete_key_removes_a_selected_edge_and_that_branch_takes_the_default(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    click_edge(signed_in, "root", "no")
    expect(signed_in.get_by_text("Edge Properties")).to_be_visible()
    delete_selected(signed_in)
    expect(signed_in.locator(EDGES)).to_have_count(1)
    save(signed_in)

    assert saved_edges(signed_in, project) == {"root:yes": [0]}
    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["approve", "decline"]


def test_a_node_with_one_missing_branch_sends_that_side_to_the_default_row(signed_in: Page):
    doc = document([unary("root"), leaf("a", 0, 100, 250)], [edge("root", "a", 0)])
    project = open_tree(signed_in, doc)
    nudge(signed_in, "a")
    assert save(signed_in)["validation_errors"] == []

    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["approve", "decline"]


def test_one_edge_can_serve_both_branches_of_a_node_from_the_edge_panel(signed_in: Page):
    doc = document([unary("root"), leaf("a", 0, 100, 250)], [edge("root", "a", 0)])
    project = open_tree(signed_in, doc)
    click_edge(signed_in, "root", "a")
    signed_in.get_by_role("combobox").click()
    signed_in.get_by_role("option").filter(has_text="otherwise").click()
    expect(signed_in.get_by_text("This edge will trigger for 2 different branches")).to_be_visible()
    save(signed_in)

    assert saved_edges(signed_in, project) == {"root:a": [0, 1]}
    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["approve", "approve"]


def test_removing_the_last_branch_of_an_edge_keeps_one_branch_assigned(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    click_edge(signed_in, "root", "yes")
    signed_in.get_by_test_id("remove-branch").last.click()
    save(signed_in)

    assert saved_edges(signed_in, project)["root:yes"] == [0]


def test_shared_node_with_two_parents_saves_and_runs(signed_in: Page):
    doc = document(
        [unary("root", threshold=1000), unary("low", "==", "code", x=100, y=150, threshold=1),
         unary("high", "==", "code", x=500, y=150, threshold=2), leaf("shared", 0, 300, 350)],
        [edge("root", "low", 0), edge("root", "high", 1), edge("low", "shared", 0), edge("high", "shared", 0)],
    )
    project = open_tree(signed_in, doc)
    expect(signed_in.locator(EDGES)).to_have_count(4)
    nudge(signed_in, "shared")
    assert save(signed_in)["validation_errors"] == []

    assert len(saved_edges(signed_in, project)) == 4
    out = run(export_document(signed_in, project), amount=[500.0, 500.0, 2000.0, 2000.0], code=[1, 2, 2, 1])
    assert out == ["approve", "decline", "approve", "decline"]


def test_stray_added_node_does_not_change_what_the_tree_decides(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    signed_in.get_by_role("button", name="Add Node").click()
    expect(signed_in.locator(NODES)).to_have_count(4)
    save(signed_in)

    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["approve", "review"]


def test_the_subtree_listed_first_is_the_tree_that_runs(signed_in: Page):
    # decikit runs only the first listed subtree's root; the others are ignored
    base = loan_document()
    doc = {**base, "nodes": [leaf("stray", -1, 900, 0), *base["nodes"]]}
    project = open_tree(signed_in, doc)
    expect(signed_in.get_by_role("button", name="Navigate to root node").first.locator("xpath=..")).to_contain_text("Leaf -1")
    nudge(signed_in, "stray")
    save(signed_in)

    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["decline", "decline"]


@pytest.mark.xfail(strict=True, reason="saved subtrees are rebuilt in node order on open, so the next save changes which subtree runs")
def test_saving_after_an_unrelated_edit_keeps_the_subtree_that_runs(signed_in: Page):
    base = loan_document()
    doc = {**base, "nodes": [*base["nodes"], leaf("stray", -1, 900, 0)],
           "subtrees": [{"id": "stray", "name": "Leaf -1"}, {"id": "root", "name": "amount ≤ 1000"}]}
    project = open_tree(signed_in, doc)
    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["decline", "decline"]
    nudge(signed_in, "yes")
    save(signed_in)

    assert run(export_document(signed_in, project), amount=[500.0, 2000.0]) == ["decline", "decline"]


@pytest.mark.xfail(strict=True, reason="a node nothing connects to is silently ignored: no message names the stray node")
def test_unreachable_node_is_named_in_the_validation_messages(signed_in: Page):
    open_tree(signed_in, loan_document())
    signed_in.get_by_role("button", name="Add Node").click()
    reply = save(signed_in)

    assert any("stray" in e["msg"].lower() or "unreachable" in e["msg"].lower() or "orphan" in e["msg"].lower()
               for e in reply["validation_errors"]), reply["validation_errors"]


def test_subtree_list_shows_one_entry_per_root_and_survives_reload(signed_in: Page):
    open_tree(signed_in, loan_document())
    signed_in.get_by_role("button", name="Add Node").click()
    expect(signed_in.get_by_text("Subtrees", exact=True)).to_be_visible()
    expect(signed_in.get_by_text("Drag to reorder priority")).to_be_visible()
    save(signed_in)
    signed_in.reload()
    expect(signed_in.locator(NODES)).to_have_count(4)
    expect(signed_in.get_by_text("Drag to reorder priority")).to_be_visible()


def test_subtree_can_be_renamed_and_the_name_is_saved(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    entry = signed_in.get_by_title("amount ≤ 1000")
    entry.hover()
    entry.locator("xpath=following-sibling::button").click()
    name = signed_in.get_by_placeholder("Subtree name")
    name.fill("Loan decision")
    name.press("Enter")
    expect(signed_in.get_by_title("Loan decision")).to_be_visible()
    save(signed_in)

    subtrees = get_project(signed_in, project)["project_data"]["subtrees"]
    assert [s["name"] for s in subtrees] == ["Loan decision"]


@pytest.mark.xfail(strict=True, reason="the Add Subtree button only logs to the console and adds nothing")
def test_add_subtree_button_adds_a_node(signed_in: Page):
    create_tree_project(signed_in, "Empty tree")
    signed_in.get_by_role("button", name="Add Subtree").click()

    expect(signed_in.locator(NODES)).to_have_count(1)


@pytest.mark.xfail(strict=True, reason="a renamed subtree is saved but comes back with its default name after a reload")
def test_renamed_subtree_keeps_its_name_after_a_reload(signed_in: Page):
    open_tree(signed_in, loan_document())
    entry = signed_in.get_by_title("amount ≤ 1000")
    entry.hover()
    entry.locator("xpath=following-sibling::button").click()
    name = signed_in.get_by_placeholder("Subtree name")
    name.fill("Loan decision")
    name.press("Enter")
    save(signed_in)
    signed_in.reload()

    expect(signed_in.get_by_title("Loan decision")).to_be_visible()


@pytest.mark.xfail(strict=True, reason="the Clear Workspace dialog promises Ctrl+Z but the editor has no undo")
def test_clear_workspace_can_be_undone_with_ctrl_z(signed_in: Page):
    open_tree(signed_in, loan_document())
    open_menu(signed_in)
    signed_in.get_by_role("menuitem", name="Clear Workspace").click()
    signed_in.get_by_role("alertdialog").get_by_role("button", name="Clear Workspace").click()
    expect(signed_in.locator(NODES)).to_have_count(0)

    signed_in.mouse.click(300, 600)
    signed_in.keyboard.press("Control+z")
    expect(signed_in.locator(NODES)).to_have_count(3)


def test_dragging_a_node_marks_the_project_unsaved_and_the_position_is_saved(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    before = node(signed_in, "yes").bounding_box()
    drag_node(signed_in, "yes", 40, 30)
    expect(signed_in.get_by_text("Unsaved changes")).to_be_visible()
    save(signed_in)

    after = node(signed_in, "yes").bounding_box()
    assert after["x"] != before["x"]
    pos = next(n for n in get_project(signed_in, project)["project_data"]["nodes"] if n["id"] == "yes")["position"]
    assert (pos["x"], pos["y"]) != (100.0, 250.0)
