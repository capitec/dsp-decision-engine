"""Live editing of one node (v3) tree from two browsers on two backend nodes.

Runs only against a stack started with E2E_NODES=2 DSP_LIVE_COLLAB=true, like `test_live_collab.py`.
"""

import os
import re

import pytest
from playwright.sync_api import Page, expect

from node_ui import document, drag_node, edge, fit_view, leaf, node, select_node, set_threshold, unary
from platform_ui import press_save, toast
from seed import create_project, get_project
from test_live_collab import ENDED, OFFLINE, SECOND_NODE, Held, session, stored, two_nodes  # noqa: F401
from tree_ui import sign_in, sign_out
from test_live_offline import Line

pytestmark = [pytest.mark.skipif(os.environ.get("E2E_LIVE_COLLAB") != "true",
                                 reason="needs a stack started with DSP_LIVE_COLLAB=true"), two_nodes]


def chain() -> dict:
    """root (amount <= 1000) -> mid (amount <= 100) -> leaves; already as the editor holds it, so opening sends nothing."""
    doc = document(
        [unary("root", threshold=1000), unary("mid", x=100, y=250, threshold=100),
         leaf("yes", 0, 0, 500), leaf("maybe", 1, 200, 500), leaf("no", 1, 500, 250)],
        [edge("root", "mid", 0), edge("root", "no", 1), edge("mid", "yes", 0), edge("mid", "maybe", 1)],
    )
    doc["subtrees"] = [{"id": "root", "name": "Main"}]
    return doc


def open_live(page: Page, project: str, hold: bool = False) -> Held | None:
    held = Held(page) if hold else None
    page.goto(f"/projects/{project}")
    expect(page.locator(".react-flow__node")).to_have_count(5, timeout=20_000)
    fit_view(page)
    expect(page.get_by_test_id("live-status")).to_have_text("Live")
    return held


def threshold(page: Page, node_id: str, value: str) -> None:
    select_node(page, node_id)
    set_threshold(page, value)


def thresholds(data: dict) -> dict:
    return {n["id"]: n["data"]["condition"]["threshold"] for n in data["nodes"] if n["data"]["type"] == "unary"}


@pytest.fixture
def tree(signed_in: Page) -> str:
    return create_project(signed_in, "Live tree", chain(), project_type="tree")["id"]


def test_edits_to_different_nodes_in_two_browsers_are_both_kept(signed_in: Page, session, tree: str):
    second = session(node=2)
    held = [open_live(page, tree, hold=True) for page in (signed_in, second)]
    for h in held:
        h.holding = True

    threshold(signed_in, "root", "250")
    threshold(second, "mid", "50")
    held[0].wait_held(signed_in)
    held[1].wait_held(second)
    for h in held:
        h.release()

    for page in (signed_in, second):
        expect(node(page, "root")).to_contain_text("250")
        expect(node(page, "mid")).to_contain_text("50")
    stored(signed_in, tree, lambda d: thresholds(d) == {"root": 250, "mid": 50})


def test_the_same_field_edited_in_two_browsers_ends_with_the_last_writer_everywhere(signed_in: Page, session, tree: str):
    second = session(node=2)
    held = [open_live(page, tree, hold=True) for page in (signed_in, second)]
    for h in held:
        h.holding = True

    threshold(signed_in, "root", "250")
    threshold(second, "root", "300")
    held[0].wait_held(signed_in)
    held[1].wait_held(second)
    held[0].release()
    stored(signed_in, tree, lambda d: thresholds(d)["root"] == 250)
    held[1].release()

    stored(signed_in, tree, lambda d: thresholds(d)["root"] == 300)
    for page in (signed_in, second):
        expect(node(page, "root")).to_contain_text("300")


def test_an_edit_to_a_node_another_browser_deleted_is_discarded_and_its_author_told(signed_in: Page, session, tree: str):
    second = session(node=2)
    open_live(signed_in, tree)
    held = open_live(second, tree, hold=True)
    held.holding = True
    threshold(second, "mid", "50")
    held.wait_held(second)

    select_node(signed_in, "mid")
    signed_in.keyboard.press("Backspace")
    expect(node(second, "mid")).to_have_count(0)
    held.release()

    expect(toast(second, "was discarded")).to_be_visible()
    expect(toast(second, "amount")).to_be_visible()
    stored(signed_in, tree, lambda d: "mid" not in thresholds(d))
    expect(second.locator(".react-flow__node")).to_have_count(4)


def test_an_offline_edit_merges_on_reconnect_with_an_edit_made_meanwhile(signed_in: Page, session, tree: str):
    second = session(node=2)
    open_live(signed_in, tree)
    line = Line(second)
    open_live(second, tree)

    line.cut()
    expect(second.get_by_test_id("live-status")).to_have_text(OFFLINE)
    threshold(second, "mid", "50")
    threshold(signed_in, "root", "250")
    stored(signed_in, tree, lambda d: thresholds(d)["root"] == 250)
    line.restore()

    expect(toast(second, "Your offline edits were merged")).to_be_visible()
    for page in (signed_in, second):
        expect(node(page, "root")).to_contain_text("250")
        expect(node(page, "mid")).to_contain_text("50")
    stored(signed_in, tree, lambda d: thresholds(d) == {"root": 250, "mid": 50})


def test_an_offline_edit_to_a_node_someone_else_changed_is_refused_with_a_banner(signed_in: Page, session, tree: str):
    second = session(node=2)
    open_live(signed_in, tree)
    line = Line(second)
    open_live(second, tree)

    line.cut()
    expect(second.get_by_test_id("live-status")).to_have_text(OFFLINE)
    threshold(second, "root", "50")
    threshold(signed_in, "root", "250")
    stored(signed_in, tree, lambda d: thresholds(d)["root"] == 250)
    line.restore()

    banner = second.get_by_test_id("offline-conflict")
    expect(banner).to_contain_text("amount")
    expect(node(second, "root")).to_contain_text("250")
    with second.expect_download() as download:
        banner.get_by_role("button", name="Download my version").click()
    assert '"threshold": 50' in download.value.path().read_text()


def test_save_waits_for_the_server_and_the_edit_is_stored(signed_in: Page, tree: str):
    open_live(signed_in, tree)
    threshold(signed_in, "root", "250")

    press_save(signed_in)

    expect(toast(signed_in, "Project Saved")).to_be_visible()
    expect(signed_in.get_by_text("Unsaved changes")).to_be_hidden()
    assert thresholds(stored(signed_in, tree, lambda d: True))["root"] == 250


def test_signing_out_on_node_two_ends_live_editing_of_a_tree_on_node_one_until_i_sign_in_again(
        signed_in: Page, email: str, tree: str):
    held = open_live(signed_in, tree, hold=True)
    held.holding = True
    threshold(signed_in, "root", "250")
    held.wait_held(signed_in)

    elsewhere = signed_in.context.new_page()  # the same browser, so the same session
    elsewhere.goto(f"{SECOND_NODE}/projects")
    sign_out(elsewhere)

    expect(signed_in.get_by_test_id("live-status")).to_have_text(ENDED, timeout=30_000)
    expect(toast(signed_in, "sign in again")).to_be_visible()
    expect(node(signed_in, "root")).to_contain_text("250")
    expect(signed_in.get_by_text("Unsaved changes")).to_be_visible()
    opened = held.opened
    signed_in.wait_for_timeout(3000)
    assert held.opened == opened  # it stopped instead of reconnecting

    held.holding = False
    expect(signed_in.get_by_role("dialog").get_by_text("Session Expired")).to_be_visible(timeout=30_000)
    with signed_in.expect_popup() as popup:
        signed_in.get_by_role("button", name="Refresh Session").click()
    sign_in(popup.value, email)
    expect(toast(signed_in, "Your offline edits were merged")).to_be_visible()
    expect(signed_in.get_by_test_id("live-status")).to_have_text("Live")
    stored(signed_in, tree, lambda d: thresholds(d)["root"] == 250)


def positions(data: dict) -> dict:
    return {n["id"]: n["position"] for n in data["nodes"]}


def test_opening_a_tree_the_editor_writes_differently_stores_nothing_until_an_edit(signed_in: Page):
    raw = chain()
    raw["subtrees"] = []  # the editor holds one subtree per root, so its copy differs from the stored one
    project = create_project(signed_in, "Raw tree", raw, project_type="tree")["id"]
    opened = get_project(signed_in, project)["version"]
    open_live(signed_in, project)

    threshold(signed_in, "root", "250")

    stored(signed_in, project, lambda d: thresholds(d)["root"] == 250)
    assert get_project(signed_in, project)["version"] == opened + 1


def test_a_node_moved_in_another_browser_stays_put_when_this_one_auto_arranges(signed_in: Page, session, tree: str):
    second = session(node=2)
    open_live(signed_in, tree)
    open_live(second, tree)
    auto = signed_in.get_by_label("Auto-arrange tree")
    auto.click()
    laid_out = positions(stored(signed_in, tree, lambda d: positions(d) != positions(chain())))
    at = laid_out["no"]
    expect(node(second, "no")).to_have_attribute("style", re.compile(rf"translate\({at['x']:g}px, {at['y']:g}px\)"))

    drag_node(second, "no", 150, 120)

    moved = positions(stored(signed_in, tree, lambda d: positions(d)["no"] != laid_out["no"]))["no"]
    expect(auto).not_to_be_checked()
    assert positions(get_project(signed_in, tree)["project_data"])["no"] == moved
