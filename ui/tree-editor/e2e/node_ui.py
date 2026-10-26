"""Node-editor (v3 tree) actions and document builders: each action does what a user does, nothing more."""

import re

import polars as pl
from decikit import TreeConfig
from playwright.sync_api import Locator, Page, expect

import flat_rules_ui
import tree_ui
from seed import create_project, get_project

SCHEMA = [["amount", "Float64"], ["band", "String"], ["code", "Int64"], ["flag", "Boolean"]]


def output_table(*labels: str, default: str = "decline") -> dict:
    """A one-column `decision` output with one record per label."""
    return {"dtypes": [["decision", "String"]], "type_defs": {}, "data": [{"decision": x} for x in labels],
            "default": {"decision": default}}


def leaf(node_id: str, result_idx: int = -1, x: float = 0.0, y: float = 300.0) -> dict:
    """A leaf node on output record `result_idx` (-1: the default row)."""
    return {"id": node_id, "position": {"x": x, "y": y}, "data": {"type": "leaf", "result_idx": result_idx}}


def unary(node_id: str, op: str = "<=", feature="amount", x: float = 300.0, y: float = 0.0, **condition) -> dict:
    """A unary node `feature op threshold` (`threshold=1000` unless other condition fields are given)."""
    if not condition:
        condition = {"threshold": 1000}
    data = {"type": "unary", "condition": {"op": op, "feature": feature, **condition}}
    return {"id": node_id, "position": {"x": x, "y": y}, "data": data}


def cases(node_id: str, op: str, conditions: list, x: float = 300.0, y: float = 0.0, **fields) -> dict:
    """A cases node (`ranges`, `isin` or `string_match`) on `amount` unless `feature=` says otherwise."""
    fields.setdefault("feature", "amount")
    return {"id": node_id, "position": {"x": x, "y": y},
            "data": {"type": "cases", "op": op, "conditions": conditions, **fields}}


def composite(node_id: str, op: str, conditions: list, x: float = 300.0, y: float = 0.0) -> dict:
    """An AND/OR/NOT node over `conditions`."""
    return {"id": node_id, "position": {"x": x, "y": y}, "data": {"type": "composite", "op": op, "conditions": conditions}}


def edge(source: str, target: str, *indices: int) -> dict:
    """An edge taking branches `indices` (default 0) of `source` to `target`."""
    return {"id": f"{source}:{target}", "source": source, "target": target, "data": {"sourceIndex": list(indices or [0])}}


def document(nodes: list, edges: list, output: dict | None = None, parameters: dict | None = None,
             input_schema: list | None = None) -> dict:
    """A v3 tree document as the editor saves it."""
    return {"type": "ui-tree", "formatVersion": 3, "nodes": nodes, "edges": edges, "subtrees": [],
            "input_schema": input_schema or SCHEMA, "parameters": parameters or {},
            "output": output or output_table("approve", "review")}


def loan_document(limit: float = 1000, op: str = "<=") -> dict:
    """`amount <op> limit` -> approve (record 0), otherwise review (record 1); the default row is decline."""
    return document(
        [unary("root", op, threshold=limit), leaf("yes", 0, 100, 250), leaf("no", 1, 500, 250)],
        [edge("root", "yes", 0), edge("root", "no", 1)],
    )


def run(doc: dict, **columns) -> list:
    """The decisions of the exported `doc` for the given input columns."""
    tree = TreeConfig.load({"type": "tree", "name": "t", "tree": doc})
    return tree.run(pl.DataFrame(columns))["decision"].to_list()


def open_tree(page: Page, doc: dict, name: str = "Tree") -> str:
    """Seed a tree project through the API and open it in the editor; returns the project id."""
    project = create_project(page, name, doc, project_type="tree")
    page.goto(f"/projects/{project['id']}")
    expect(page.get_by_role("button", name="Add Node")).to_be_visible(timeout=20_000)  # a cold backend can be slow to load
    expect(page.locator(".react-flow__node")).to_have_count(len(doc["nodes"]), timeout=20_000)
    fit_view(page)
    return project["id"]


def fit_view(page: Page) -> None:
    """Zoom the canvas so every node is on screen (seeded trees can sit far from the origin)."""
    page.locator(".react-flow__controls-fitview").dispatch_event("click")  # a real click scrolls the page: the controls sit below the fold


def node(page: Page, node_id: str) -> Locator:
    """The canvas node with id `node_id`."""
    return page.locator(f'.react-flow__node[data-id="{node_id}"]')


def select_node(page: Page, node_id: str) -> None:
    node(page, node_id).click()
    expect(page.get_by_text("Node Type", exact=True)).to_be_visible()


def deselect(page: Page) -> None:
    """Click the empty canvas: the sidebar goes back to the general panel."""
    page.locator(".react-flow__pane").click(position={"x": 5, "y": 5})
    expect(page.get_by_role("button", name="Add Node")).to_be_visible()


def edge_between(page: Page, source: str, target: str) -> Locator:
    """The canvas edge from `source` to `target`."""
    return page.locator(f'.react-flow__edge[data-id="{source}:{target}"]')


def edge_labels(page: Page) -> Locator:
    """The layer holding every edge's branch label."""
    return page.locator(".react-flow__edgelabel-renderer")


def connect(page: Page, source: str, target: str) -> None:
    """Drag from the source node's bottom handle to the target node's top handle."""
    tree_ui.connect(page, node(page, source), node(page, target))


def delete_selected(page: Page, key: str = "Backspace") -> None:
    """Press the delete key with the canvas focused (a selected node or edge goes away)."""
    page.keyboard.press(key)


def sidebar(page: Page) -> Locator:
    """The editor's right-hand panel (node, edge or general settings)."""
    return page.get_by_test_id("editor-sidebar")


def set_node_type(page: Page, kind: str) -> None:
    """Sidebar Node Type: Leaf Node, Unary Condition, Composite (AND/OR/NOT) or Cases."""
    sidebar(page).get_by_role("combobox").first.click()
    page.get_by_role("option", name=re.compile(f"^{re.escape(kind)}")).click()


def set_operator(page: Page, symbol: str) -> None:
    """Sidebar Operator of a unary node, by the option's text (≤, <, ==, >, ≥, !=, Between, IsIn, ...)."""
    sidebar(page).get_by_role("combobox").nth(1).click()
    page.get_by_role("option", name=symbol, exact=True).click()


def choose_feature(page: Page, feature: str, current: str = "Select or search feature...", scope: Locator | None = None) -> None:
    """Open the feature selector (showing `current`) and pick `feature`."""
    scope = scope or sidebar(page)
    scope.get_by_role("combobox").filter(has_text=current).first.click()
    page.get_by_role("option", name=feature, exact=True).click()


def set_threshold(page: Page, value: str) -> None:
    """Type `value` into the selected unary node's threshold."""
    sidebar(page).get_by_placeholder("0", exact=True).fill(value)


def use_parameter(page: Page, name: str, index: int = 0) -> None:
    """Click the `index`-th # in the sidebar (next to a value), then type the parameter's name."""
    flat_rules_ui.use_parameter(sidebar(page), name, index)


def save(page: Page) -> dict:
    """Click Save and wait for the indicator to clear; returns the server's reply (with `validation_errors`)."""
    reply = flat_rules_ui.save(page)
    expect(page.get_by_text("Unsaved changes")).to_be_hidden()
    return reply


def open_menu(page: Page) -> None:
    """Open the editor header's menu (Export Project, Clear Workspace, ...)."""
    page.get_by_test_id("project-menu").click()


def drag_node(page: Page, node_id: str, dx: int, dy: int) -> None:
    """Drag the canvas node `node_id` by (dx, dy) pixels."""
    tree_ui.drag(page, node(page, node_id), dx, dy)


def nudge(page: Page, node_id: str) -> None:
    """Drag a node a little: a change that makes the project dirty without touching the tree's meaning."""
    drag_node(page, node_id, 20, 20)


def click_edge(page: Page, source: str, target: str) -> None:
    """Select an edge (its wide invisible path takes the click)."""
    edge_between(page, source, target).locator(".react-flow__edge-interaction").dispatch_event("click")


def settle(page: Page) -> None:
    """Wait for the editor to finish loading, saving once if opening the project alone marked it unsaved."""
    page.wait_for_load_state("networkidle")
    if page.get_by_text("Unsaved changes").is_visible():
        save(page)


def two_leaves(root: dict, *more: dict, edges: tuple = ((0, "yes"), (1, "no"))) -> dict:
    """`root` with an approve leaf on branch 0 and a review leaf on branch 1."""
    return document([root, leaf("yes", 0, 100, 250), leaf("no", 1, 500, 250), *more],
                    [edge(root["id"], target, index) for index, target in edges], input_schema=SCHEMA)


def saved_data(page: Page, project: str, node_id: str = "root") -> dict:
    """The stored `data` of node `node_id`."""
    return next(n for n in get_project(page, project)["project_data"]["nodes"] if n["id"] == node_id)["data"]


def outcome(truth: list[bool]) -> list[str]:
    """approve where the condition holds, review otherwise (the `two_leaves` wiring)."""
    return ["approve" if t else "review" for t in truth]
