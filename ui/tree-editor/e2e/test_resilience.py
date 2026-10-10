"""Saving when the server or the network misbehaves, slow saves, two tabs, and leaving an editor with unsaved edits."""

import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

import node_ui
from flat_rules_ui import add_parameter, open_flat_rules
from platform_ui import edit_is_stored, open_projects, open_with_edit, press_save, project_card, toast
from seed import create_project, get_project
from test_flat_rule_params import flat_document

SAVE_URL = re.compile(r"/api/v1/projects/[0-9a-f-]{36}$")
NO_CONFLICT_DETECTION = "no conflict detection: the second tab's save replaces the first tab's saved edit without a warning"
NO_FLAT_GUARD = "the flat-rule editor has no unsaved-changes guard: reloading drops the edit without asking"
DIRTY_FLAG_CLEARED = "an edit made while a save is in flight is neither stored nor flagged: the save's reply clears 'Unsaved changes'"


def on_saves(page: Page, handler) -> None:
    """Run `handler(route)` for every save (PUT) of a project; other requests go through."""
    page.route(SAVE_URL, lambda route: handler(route) if route.request.method == "PUT" else route.fallback())


def count_saves(page: Page) -> list:
    """The list that grows by one for every save request the page sends."""
    seen = []
    page.on("request", lambda request: seen.append(request) if request.method == "PUT" and SAVE_URL.search(request.url) else None)
    return seen


def held_save(page: Page, held: list) -> Route:
    """The first save request the page sent to a `held` handler (each round trip lets pending routes arrive)."""
    for _ in range(200):
        if held:
            return held[0]
        page.evaluate("0")
    raise AssertionError("no save request was sent")


def failure_shown(page: Page, kind: str) -> None:
    expect(toast(page, "Save Failed")).to_be_visible()
    if kind == "flat":
        expect(page.get_by_test_id("rule-card")).to_have_count(2)
    else:
        expect(page.get_by_text("Unsaved changes")).to_be_visible()


# ---- a save that fails -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["flat", "tree"])
@pytest.mark.parametrize("status,detail", [(500, "the database is down"), (400, "that document is not valid")])
def test_a_refused_save_shows_the_reason_keeps_the_edit_and_succeeds_when_retried(signed_in: Page, kind: str, status: int, detail: str):
    project = open_with_edit(signed_in, kind)
    on_saves(signed_in, lambda route: route.fulfill(status=status, json={"detail": detail}))

    press_save(signed_in)

    failure_shown(signed_in, kind)
    expect(toast(signed_in, detail)).to_be_visible()
    assert not edit_is_stored(signed_in, project, kind)
    signed_in.unroute(SAVE_URL)
    press_save(signed_in)
    expect(toast(signed_in, "Project Saved")).to_be_visible()
    assert edit_is_stored(signed_in, project, kind)


@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_save_that_never_reaches_the_server_shows_an_error_keeps_the_edit_and_succeeds_when_retried(signed_in: Page, kind: str):
    project = open_with_edit(signed_in, kind)
    on_saves(signed_in, lambda route: route.abort())

    press_save(signed_in)

    failure_shown(signed_in, kind)
    assert not edit_is_stored(signed_in, project, kind)
    signed_in.unroute(SAVE_URL)
    press_save(signed_in)
    expect(toast(signed_in, "Project Saved")).to_be_visible()
    assert edit_is_stored(signed_in, project, kind)


@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_failed_save_does_not_leave_the_editor_stuck_saving(signed_in: Page, kind: str):
    open_with_edit(signed_in, kind)
    on_saves(signed_in, lambda route: route.fulfill(status=500, json={"detail": "boom"}))

    press_save(signed_in)

    failure_shown(signed_in, kind)
    expect(signed_in.get_by_role("button", name="Save", exact=True)).to_be_enabled()


# ---- a slow save --------------------------------------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_slow_save_shows_progress_and_repeated_clicks_send_one_request(signed_in: Page, kind: str):
    project = open_with_edit(signed_in, kind)
    held = []
    on_saves(signed_in, held.append)
    seen = count_saves(signed_in)

    signed_in.get_by_role("button", name="Save", exact=True).dblclick()
    expect(signed_in.get_by_role("button", name="Saving...")).to_be_disabled()
    held_save(signed_in, held).continue_()

    expect(toast(signed_in, "Project Saved")).to_be_visible()
    assert len(seen) == 1
    assert edit_is_stored(signed_in, project, kind)


@pytest.mark.xfail(strict=True, reason=DIRTY_FLAG_CLEARED)
def test_an_edit_made_while_a_save_is_in_flight_is_not_reported_as_saved(signed_in: Page):
    project = node_ui.open_tree(signed_in, node_ui.loan_document(), "In flight")
    node_ui.nudge(signed_in, "root")
    held = []
    on_saves(signed_in, held.append)

    press_save(signed_in)
    held_save(signed_in, held)
    node_ui.nudge(signed_in, "yes")
    held[0].continue_()
    expect(toast(signed_in, "Project Saved")).to_be_visible()

    moved = {n["id"]: n["position"] for n in get_project(signed_in, project)["project_data"]["nodes"]}
    assert moved["yes"] != node_ui.loan_document()["nodes"][1]["position"] or signed_in.get_by_text("Unsaved changes").is_visible()


def test_double_clicking_export_downloads_the_same_file_once_or_twice(signed_in: Page, tmp_path):
    project = create_project(signed_in, "Exported twice", flat_document())["id"]
    open_flat_rules(signed_in, project)
    downloads = []
    signed_in.on("download", lambda download: downloads.append(download))

    signed_in.get_by_role("button", name="Export").dblclick()
    signed_in.get_by_role("button", name="Save", exact=True).click()  # a later action: the page is still responsive
    expect(toast(signed_in, "Project Saved").last).to_be_visible()

    contents = []
    for i, download in enumerate(downloads):
        path = tmp_path / f"export-{i}.json"
        download.save_as(path)
        contents.append(json.loads(path.read_text()))
    assert 1 <= len(contents) <= 2
    assert all(c == contents[0] for c in contents)


# ---- two tabs ---------------------------------------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=NO_CONFLICT_DETECTION)
def test_two_tabs_saving_different_edits_keep_both_edits(signed_in: Page):
    project = create_project(signed_in, "Two tabs", flat_document())["id"]
    first = signed_in
    open_flat_rules(first, project)
    second = signed_in.context.new_page()
    open_flat_rules(second, project)

    add_parameter(first, "from_first", "Float64", "1")
    press_save(first)
    expect(toast(first, "Project Saved")).to_be_visible()
    second.get_by_role("button", name="Add Rule").click()
    expect(second.get_by_test_id("rule-card")).to_have_count(2)
    press_save(second)
    expect(toast(second, "Project Saved")).to_be_visible()

    stored = get_project(signed_in, project)["project_data"]
    assert "from_first" in stored["parameters"] and len(stored["rules"]) == 2


def test_two_tabs_saving_the_same_project_end_with_the_last_save_stored(signed_in: Page):
    project = create_project(signed_in, "Two tabs", flat_document())["id"]
    first = signed_in
    open_flat_rules(first, project)
    second = signed_in.context.new_page()
    open_flat_rules(second, project)
    first.get_by_role("button", name="Add Rule").click()
    press_save(first)
    expect(toast(first, "Project Saved")).to_be_visible()

    second.get_by_role("button", name="Add Rule").click()
    second.get_by_role("button", name="Add Rule").click()
    press_save(second)
    expect(toast(second, "Project Saved")).to_be_visible()

    assert len(get_project(signed_in, project)["project_data"]["rules"]) == 3


# ---- leaving the editor ---------------------------------------------------------------------------------------------------------

def test_back_and_forward_between_the_project_list_and_an_editor(signed_in: Page):
    project = create_project(signed_in, "Back and forth", flat_document())["id"]
    open_projects(signed_in)
    project_card(signed_in, "Back and forth").get_by_role("button", name="Open Project").click()
    expect(signed_in.get_by_role("button", name="Add Rule")).to_be_visible()

    signed_in.go_back()
    expect(project_card(signed_in, "Back and forth")).to_be_visible()
    signed_in.go_forward()

    expect(signed_in).to_have_url(re.compile(project))
    expect(signed_in.get_by_role("button", name="Add Rule")).to_be_visible()
    expect(signed_in.get_by_test_id("rule-card")).to_have_count(1)


def test_reloading_the_node_editor_with_unsaved_edits_asks_first(signed_in: Page):
    open_with_edit(signed_in, "tree")
    asked = []
    signed_in.on("dialog", lambda dialog: asked.append(dialog.type))

    signed_in.reload()

    assert "beforeunload" in asked


def test_leaving_the_node_editor_with_unsaved_edits_asks_first(signed_in: Page):
    open_with_edit(signed_in, "tree")
    asked = []
    signed_in.on("dialog", lambda dialog: asked.append(dialog.message))

    signed_in.get_by_role("button", name="Back").click()

    expect(signed_in).to_have_url(re.compile(r"/projects$"))
    assert any("unsaved changes" in message for message in asked)


def test_reloading_the_node_editor_after_saving_does_not_ask(signed_in: Page):
    project = open_with_edit(signed_in, "tree")
    press_save(signed_in)
    expect(toast(signed_in, "Project Saved")).to_be_visible()
    asked = []
    signed_in.on("dialog", lambda dialog: asked.append(dialog.type))

    signed_in.reload()

    assert asked == []
    assert edit_is_stored(signed_in, project, "tree")


@pytest.mark.xfail(strict=True, reason=NO_FLAT_GUARD)
def test_reloading_the_flat_rule_editor_with_unsaved_edits_asks_first(signed_in: Page):
    open_with_edit(signed_in, "flat")
    asked = []
    signed_in.on("dialog", lambda dialog: asked.append(dialog.type))

    signed_in.reload()

    assert "beforeunload" in asked
