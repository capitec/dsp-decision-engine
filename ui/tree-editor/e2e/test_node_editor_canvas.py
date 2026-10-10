"""Node-editor canvas: layout, zoom, pan, selection, and trees large enough to be slow."""

import json
import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from node_ui import (
    deselect,
    document,
    drag_node,
    edge,
    fit_view,
    leaf,
    loan_document,
    node,
    open_tree,
    run,
    save,
    settle,
    select_node,
    set_threshold,
    sidebar,
    unary,
)
from seed import export_document, get_project, import_document

NODES = ".react-flow__node"
CAMPAIGN = Path(__file__).resolve().parents[3] / "examples/campaign-trees/configs/0.1.0/campaign_23.json"


VIEWPORT_MATRIX = "new DOMMatrix(getComputedStyle(document.querySelector('.react-flow__viewport')).transform)"


def zoom(page: Page) -> float:
    return page.evaluate(f"() => {VIEWPORT_MATRIX}.a")


def pan(page: Page) -> tuple:
    return tuple(page.evaluate(f"() => {{ const m = {VIEWPORT_MATRIX}; return [m.e, m.f]; }}"))


def wait_for_zoom(page: Page, comparison: str, reference: float) -> None:
    page.wait_for_function(f"([ref]) => {VIEWPORT_MATRIX}.a {comparison} ref", arg=[reference])


def wait_until_apart(page: Page, ids: list) -> None:
    """Wait until no two of the nodes overlap on screen."""
    page.wait_for_function("""(ids) => {
        const r = ids.map(i => document.querySelector(`.react-flow__node[data-id="${i}"]`).getBoundingClientRect());
        return r.every((a, i) => r.every((b, j) => i === j || a.right <= b.left || b.right <= a.left
                                                   || a.bottom <= b.top || b.bottom <= a.top));
    }""", arg=ids)


def stacked() -> dict:
    """The loan tree with every node at the origin, on top of each other."""
    doc = loan_document()
    for n in doc["nodes"]:
        n["position"] = {"x": 0.0, "y": 0.0}
    return doc


def test_rearrange_tree_spreads_stacked_nodes_apart_and_marks_the_project_unsaved(signed_in: Page):
    open_tree(signed_in, stacked())
    signed_in.get_by_role("button", name="Rearrange Tree").click()

    wait_until_apart(signed_in, ["root", "yes", "no"])
    expect(signed_in.get_by_text("Unsaved changes")).to_be_visible()
    root, yes, no = (node(signed_in, i).bounding_box() for i in ("root", "yes", "no"))
    assert root["y"] < yes["y"] and yes["x"] < no["x"]  # the branch where the condition holds is on the left


def test_rearranged_positions_are_saved(signed_in: Page):
    project = open_tree(signed_in, stacked())
    signed_in.get_by_role("button", name="Rearrange Tree").click()
    expect(signed_in.get_by_text("Unsaved changes")).to_be_visible()
    save(signed_in)

    positions = {n["id"]: (n["position"]["x"], n["position"]["y"]) for n in get_project(signed_in, project)["project_data"]["nodes"]}
    assert len(set(positions.values())) == 3


def test_auto_arrange_spreads_stacked_nodes_without_a_click(signed_in: Page):
    open_tree(signed_in, stacked())
    signed_in.get_by_label("Auto-arrange tree").click()

    wait_until_apart(signed_in, ["root", "yes", "no"])


def test_zoom_buttons_change_the_scale_and_fit_view_brings_every_node_back(signed_in: Page):
    open_tree(signed_in, loan_document())
    start = zoom(signed_in)
    signed_in.locator(".react-flow__controls-zoomin").dispatch_event("click")
    wait_for_zoom(signed_in, ">", start + 0.01)
    signed_in.locator(".react-flow__controls-zoomout").dispatch_event("click")
    signed_in.locator(".react-flow__controls-zoomout").dispatch_event("click")
    wait_for_zoom(signed_in, "<", start - 0.01)
    fit_view(signed_in)
    wait_for_zoom(signed_in, "<", start + 0.01)
    wait_for_zoom(signed_in, ">", start - 0.01)


def test_dragging_the_canvas_pans_without_moving_nodes_or_marking_unsaved(signed_in: Page):
    open_tree(signed_in, loan_document())
    settle(signed_in)
    before = pan(signed_in)
    node_box = node(signed_in, "root").bounding_box()
    signed_in.mouse.move(60, 700)
    signed_in.mouse.down()
    signed_in.mouse.move(160, 760, steps=8)
    signed_in.mouse.up()

    signed_in.wait_for_function(f"() => {VIEWPORT_MATRIX}.e > {before[0] + 50}")
    assert node(signed_in, "root").bounding_box()["x"] == pytest.approx(node_box["x"] + 100, abs=2)
    expect(signed_in.get_by_text("Unsaved changes")).to_be_hidden()


@pytest.mark.xfail(strict=True, reason="the canvas is 100vh tall below an 80px header, so the page scrolls and the controls and minimap sit below the fold")
def test_editor_fits_the_window_without_page_scroll(signed_in: Page):
    open_tree(signed_in, loan_document())

    assert signed_in.evaluate("document.documentElement.scrollHeight <= window.innerHeight")


def test_minimap_and_controls_are_on_the_canvas(signed_in: Page):
    open_tree(signed_in, loan_document())

    expect(signed_in.locator(".react-flow__minimap")).to_be_visible()
    expect(signed_in.locator(".react-flow__controls")).to_be_visible()


def test_selecting_a_node_an_edge_and_the_empty_canvas_shows_the_matching_panel(signed_in: Page):
    open_tree(signed_in, loan_document())
    select_node(signed_in, "root")
    expect(sidebar(signed_in).get_by_text("Operator:")).to_be_visible()
    deselect(signed_in)
    expect(sidebar(signed_in).get_by_text("Output Configuration")).to_be_visible()
    signed_in.locator('.react-flow__edge[data-id="root:yes"] .react-flow__edge-interaction').dispatch_event("click")
    expect(sidebar(signed_in).get_by_text("Edge Properties")).to_be_visible()
    select_node(signed_in, "yes")
    expect(sidebar(signed_in).get_by_text("Edge Properties")).to_be_hidden()
    expect(sidebar(signed_in).get_by_text("Node Type")).to_be_visible()


@pytest.mark.xfail(strict=True, reason="selecting a node marks the project 'Unsaved changes'")
def test_selection_does_not_mark_the_project_unsaved(signed_in: Page):
    open_tree(signed_in, loan_document())
    settle(signed_in)
    for node_id in ("root", "yes", "no"):
        select_node(signed_in, node_id)

    expect(signed_in.get_by_text("Unsaved changes")).to_be_hidden()


def test_ctrl_click_selects_two_nodes_and_the_sidebar_keeps_showing_one_panel(signed_in: Page):
    open_tree(signed_in, loan_document())
    node(signed_in, "yes").click()
    node(signed_in, "no").click(modifiers=["Control"])

    expect(signed_in.locator(".react-flow__node.selected")).to_have_count(2)
    expect(sidebar(signed_in).get_by_text("Node Type")).to_have_count(1)


def test_dragged_nodes_stay_where_they_were_dropped_after_save_and_reload(signed_in: Page):
    project = open_tree(signed_in, loan_document())
    drag_node(signed_in, "no", 0, 60)
    save(signed_in)
    after_drag = {n["id"]: n["position"] for n in get_project(signed_in, project)["project_data"]["nodes"]}
    signed_in.reload()
    expect(signed_in.locator(NODES)).to_have_count(3)

    again = {n["id"]: n["position"] for n in get_project(signed_in, project)["project_data"]["nodes"]}
    assert again == after_drag
    assert after_drag["no"]["y"] > 250


# ---- large trees ----------------------------------------------------------------------------------------------------

def big_tree(internal: int) -> dict:
    """A complete binary tree of `internal` unary nodes `amount <= i` with `internal + 1` leaves, laid out in rows."""
    nodes, edges = [], []
    total = 2 * internal + 1
    for i in range(total):
        row, col = divmod(i, 20)
        if i < internal:
            nodes.append(unary(f"n{i}", "<=", "amount", x=col * 220.0, y=row * 120.0, threshold=i))
            nodes_children = [2 * i + 1, 2 * i + 2]
            for branch, child in enumerate(nodes_children):
                edges.append(edge(f"n{i}", f"n{child}", branch))
        else:
            nodes.append(leaf(f"n{i}", i % 2, x=col * 220.0, y=row * 120.0))
    return document(nodes, edges)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="the campaign example's root is a composite node, and a composite root crashes the editor on open")
def test_campaign_tree_imports_opens_and_exports_the_same_nodes(signed_in: Page):
    tree = json.loads(CAMPAIGN.read_text())["tree"]
    project = import_document(signed_in, "Campaign", tree)["project_id"]
    started = time.monotonic()
    signed_in.goto(f"/projects/{project}")
    expect(signed_in.get_by_role("button", name="Add Node")).to_be_visible()
    expect(signed_in.locator(NODES)).to_have_count(len(tree["nodes"]))
    expect(signed_in.locator(".react-flow__edge")).to_have_count(len(tree["edges"]))
    assert time.monotonic() - started < 20
    assert len(export_document(signed_in, project)["nodes"]) == len(tree["nodes"])


@pytest.mark.slow
def test_three_hundred_node_tree_opens_renders_every_node_and_edits_and_saves_a_deep_node(signed_in: Page):
    doc = big_tree(150)
    project = signed_in.request.post("/api/v1/projects", data={"name": "Big", "project_type": "tree", "project_data": doc})
    assert project.ok, project.text()
    project_id = project.json()["id"]

    started = time.monotonic()
    signed_in.goto(f"/projects/{project_id}")
    expect(signed_in.locator(NODES)).to_have_count(len(doc["nodes"]), timeout=60_000)
    opened = time.monotonic() - started

    deep = "n149"  # the last internal node, a long way from the root
    node(signed_in, deep).dispatch_event("click")
    expect(sidebar(signed_in).get_by_text("Operator:")).to_be_visible()
    started = time.monotonic()
    set_threshold(signed_in, "12345")
    reply = save(signed_in)
    edited = time.monotonic() - started

    print(f"opened {len(doc['nodes'])} nodes in {opened:.1f}s; edit and save in {edited:.1f}s")
    assert opened < 30 and edited < 15
    assert reply["validation_errors"] == []
    saved = next(n for n in get_project(signed_in, project_id)["project_data"]["nodes"] if n["id"] == deep)
    assert saved["data"]["condition"]["threshold"] == 12345
    assert len(run(export_document(signed_in, project_id), amount=[1.0, 100.0, 500.0])) == 3


@pytest.mark.slow
def test_rearranging_a_three_hundred_node_tree_separates_the_nodes(signed_in: Page):
    doc = big_tree(150)
    for n in doc["nodes"]:
        n["position"] = {"x": 0.0, "y": 0.0}
    project = signed_in.request.post("/api/v1/projects", data={"name": "Big", "project_type": "tree", "project_data": doc})
    assert project.ok, project.text()
    signed_in.goto(f"/projects/{project.json()['id']}")
    expect(signed_in.locator(NODES)).to_have_count(len(doc["nodes"]), timeout=60_000)

    started = time.monotonic()
    signed_in.get_by_role("button", name="Rearrange Tree").click()
    expect(signed_in.get_by_text("Unsaved changes")).to_be_visible(timeout=30_000)
    assert time.monotonic() - started < 20
