"""Saving when the server or the network misbehaves, slow saves, two tabs (a stale save is refused), and leaving an editor with unsaved edits."""

import json
import re

import pytest
from playwright.sync_api import Page, expect

import node_ui
from flat_rules_ui import LIVE, add_parameter, open_flat_rules, save
from platform_ui import edit_is_stored, first_held, open_projects, open_with_edit, press_save, project_card, toast
from seed import create_folder, create_project, get_project
from test_flat_rule_params import flat_document

SAVE_URL = re.compile(r"/api/v1/projects/[0-9a-f-]{36}$")


def on_saves(page: Page, handler) -> None:
    """Run `handler(route)` for every save (PUT) of a project; other requests go through."""
    page.route(SAVE_URL, lambda route: handler(route) if route.request.method == "PUT" else route.fallback())


def count_saves(page: Page) -> list:
    """The list that grows by one for every save request the page sends."""
    seen = []
    page.on("request", lambda request: seen.append(request) if request.method == "PUT" and SAVE_URL.search(request.url) else None)
    return seen


def failure_shown(page: Page, kind: str) -> None:
    expect(toast(page, "Save Failed")).to_be_visible()
    if kind == "flat":
        expect(page.get_by_test_id("rule-card")).to_have_count(2)
    else:
        expect(page.get_by_text("Unsaved changes")).to_be_visible()


# ---- a save that fails -------------------------------------------------------------------------------------------------

@pytest.mark.skipif(LIVE, reason="a live Save sends no request to refuse; test_live_offline's "
                                 "test_edits_the_server_fails_on_every_time_are_handed_back_after_three_tries covers a "
                                 "server failing on edits (the flat-rule editor; none for the node editor)")
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


@pytest.mark.skipif(LIVE, reason="a live Save sends no request to lose; edits made while the server is unreachable "
                                 "merge on reconnect: test_live_offline's "
                                 "test_offline_edits_to_another_rule_merge_on_reconnect_and_everyone_sees_them and "
                                 "test_live_node_editor's test_an_offline_edit_merges_on_reconnect_with_an_edit_made_meanwhile")
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


@pytest.mark.skipif(LIVE, reason="a live Save sends no request to fail; no live test covers a Save that fails")
@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_failed_save_does_not_leave_the_editor_stuck_saving(signed_in: Page, kind: str):
    open_with_edit(signed_in, kind)
    on_saves(signed_in, lambda route: route.fulfill(status=500, json={"detail": "boom"}))

    press_save(signed_in)

    failure_shown(signed_in, kind)
    expect(signed_in.get_by_role("button", name="Save", exact=True)).to_be_enabled()


# ---- a slow save --------------------------------------------------------------------------------------------------------

@pytest.mark.smoke
@pytest.mark.skipif(LIVE, reason="a live Save sends no request to hold; test_live_collab's and test_live_node_editor's "
                                 "test_save_waits_for_the_server_* cover Save while live, none a slow one")
@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_slow_save_shows_progress_and_repeated_clicks_send_one_request(signed_in: Page, kind: str):
    project = open_with_edit(signed_in, kind)
    held = []
    on_saves(signed_in, held.append)
    seen = count_saves(signed_in)

    signed_in.get_by_role("button", name="Save", exact=True).dblclick()
    expect(signed_in.get_by_role("button", name="Saving...")).to_be_disabled()
    first_held(signed_in, held).continue_()

    expect(toast(signed_in, "Project Saved")).to_be_visible()
    assert len(seen) == 1
    assert edit_is_stored(signed_in, project, kind)


@pytest.mark.skipif(LIVE, reason="a live Save sends no request to hold; no live test covers an edit made while Save "
                                 "waits")
def test_an_edit_made_while_a_save_is_in_flight_is_not_reported_as_saved(signed_in: Page):
    project = node_ui.open_tree(signed_in, node_ui.loan_document(), "In flight")
    node_ui.nudge(signed_in, "root")
    held = []
    on_saves(signed_in, held.append)

    press_save(signed_in)
    first_held(signed_in, held)
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

    with signed_in.expect_download():
        signed_in.get_by_role("button", name="Export").dblclick()
    save(signed_in)  # a later save still goes through: the page is responsive and the export left no save running

    contents = []
    for i, download in enumerate(downloads):
        path = tmp_path / f"export-{i}.json"
        download.save_as(path)
        content = json.loads(path.read_text())
        del content["metadata"]["exported_at"]  # two downloads are two exports, a moment apart
        contents.append(content)
    assert 1 <= len(contents) <= 2
    assert all(c == contents[0] for c in contents)


# ---- two tabs ---------------------------------------------------------------------------------------------------------------

def node_position(page: Page, node_id: str) -> dict:
    """Where the canvas shows node `node_id`, from its transform."""
    transform = node_ui.node(page, node_id).evaluate("el => el.style.transform")
    x, y = (float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", transform)[:2])
    return {"x": x, "y": y}


def stale_tab(signed_in: Page, kind: str) -> tuple[str, Page]:
    """Two tabs on one project; the first tab saves an edit, then the second tab makes a different one. (project, second)"""
    first = signed_in
    if kind == "flat":
        project = create_project(first, "Two tabs", flat_document())["id"]
        open_flat_rules(first, project)
        second = first.context.new_page()
        open_flat_rules(second, project)
        for _ in range(2):
            first.get_by_role("button", name="Add Rule").click()
        press_save(first)
        expect(toast(first, "Project Saved")).to_be_visible()
        second.get_by_role("button", name="Add Rule").click()
        expect(second.get_by_test_id("rule-card")).to_have_count(2)
    else:
        project = node_ui.open_tree(first, node_ui.loan_document(), "Two tabs")
        second = first.context.new_page()
        second.goto(f"/projects/{project}")
        expect(second.locator(".react-flow__node")).to_have_count(3, timeout=20_000)
        node_ui.fit_view(second)
        node_ui.nudge(first, "root")
        node_ui.save(first)
        node_ui.nudge(second, "yes")
        expect(second.get_by_text("Unsaved changes")).to_be_visible()
    return project, second


@pytest.mark.smoke
@pytest.mark.skipif(LIVE, reason="live, two tabs' edits merge instead of the later save being refused: "
                                 "test_live_collab's test_simultaneous_edits_to_different_rules_are_all_kept and "
                                 "test_live_node_editor's test_edits_to_different_nodes_in_two_browsers_are_both_kept")
@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_stale_save_from_a_second_tab_is_refused_and_its_edit_stays(signed_in: Page, kind: str):
    project, second = stale_tab(signed_in, kind)

    press_save(second)

    expect(toast(second, "Someone else saved this project")).to_be_visible()
    expect(toast(second, "Project Saved")).to_have_count(0)
    expect(second.get_by_text("Unsaved changes")).to_be_visible()
    stored = get_project(signed_in, project)["project_data"]
    if kind == "flat":
        expect(second.get_by_test_id("rule-card")).to_have_count(2)
        assert len(stored["rules"]) == 3
    else:
        positions = {n["id"]: n["position"] for n in stored["nodes"]}
        original = {n["id"]: n["position"] for n in node_ui.loan_document()["nodes"]}
        assert positions["root"] != original["root"] and positions["yes"] == original["yes"]


@pytest.mark.skipif(LIVE, reason="live, the second tab's save is not refused, so there is nothing to reload for: "
                                 "test_live_collab's test_simultaneous_edits_to_different_rules_are_all_kept and "
                                 "test_live_node_editor's test_edits_to_different_nodes_in_two_browsers_are_both_kept")
@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_after_a_reload_the_second_tab_sees_the_first_tabs_edit_and_can_save(signed_in: Page, kind: str):
    project, second = stale_tab(signed_in, kind)
    press_save(second)
    expect(toast(second, "Someone else saved this project")).to_be_visible()

    second.once("dialog", lambda dialog: dialog.accept())
    second.reload()

    if kind == "flat":
        expect(second.get_by_test_id("rule-card")).to_have_count(3)
    else:
        expect(second.locator(".react-flow__node")).to_have_count(3, timeout=20_000)
        stored = next(n for n in get_project(signed_in, project)["project_data"]["nodes"] if n["id"] == "root")
        assert node_position(second, "root") == pytest.approx(stored["position"], abs=0.01)  # the transform is rounded
    expect(second.get_by_text("Unsaved changes")).to_be_hidden()
    if kind == "flat":
        second.get_by_role("button", name="Add Rule").click()
    else:
        node_ui.nudge(second, "yes")
    press_save(second)
    expect(toast(second, "Project Saved")).to_be_visible()
    stored = get_project(signed_in, project)["project_data"]
    if kind == "flat":
        assert len(stored["rules"]) == 4
    else:
        original = {n["id"]: n["position"] for n in node_ui.loan_document()["nodes"]}
        assert all(n["position"] != original[n["id"]] for n in stored["nodes"] if n["id"] in ("root", "yes"))


def put_project(page: Page, project: str, **body):
    return page.request.put(f"/api/v1/projects/{project}", data=body)


def test_a_save_with_a_stale_base_version_is_refused_with_the_current_version(signed_in: Page):
    project = create_project(signed_in, "Versioned", flat_document())
    assert project["version"] == 1

    saved = put_project(signed_in, project["id"], project_data=flat_document(), name="Saved", base_version=1)
    stale = put_project(signed_in, project["id"], project_data=flat_document(), name="Stale", base_version=1)

    assert saved.ok and saved.json()["version"] == 2
    assert stale.status == 409
    assert stale.json()["detail"]["current_version"] == 2
    stored = get_project(signed_in, project["id"])
    assert (stored["name"], stored["version"]) == ("Saved", 2)


def test_concurrent_saves_of_one_base_version_let_exactly_one_through(signed_in: Page):
    project = create_project(signed_in, "Raced", flat_document())["id"]

    statuses = signed_in.evaluate(
        """([project, doc]) => Promise.all([...Array(6).keys()].map(i => fetch(`/api/v1/projects/${project}`, {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: `Racer ${i}`, project_data: doc, base_version: 1}),
        }).then(r => r.status)))""",
        [project, flat_document()],
    )

    assert sorted(statuses) == [200] + [409] * 5
    assert get_project(signed_in, project)["version"] == 2


def test_a_document_save_without_a_base_version_is_refused_with_422_and_changes_nothing(signed_in: Page):
    project = create_project(signed_in, "Unversioned", flat_document())["id"]

    refused = put_project(signed_in, project, project_data=flat_document(), name="Overwritten")

    assert refused.status == 422
    assert "base_version" in refused.json()["validation_errors"][0]["msg"]
    stored = get_project(signed_in, project)
    assert (stored["name"], stored["version"]) == ("Unversioned", 1)


def test_renaming_or_moving_a_project_leaves_its_version(signed_in: Page):
    project = create_project(signed_in, "Moved", flat_document())["id"]
    folder = create_folder(signed_in, "Target")["id"]

    renamed = put_project(signed_in, project, name="Renamed", base_version=99)
    moved = signed_in.request.post(f"/api/v1/projects/{project}/move", data={"new_folder_id": folder})

    assert renamed.ok and renamed.json()["version"] == 1
    assert moved.ok and moved.json()["version"] == 1
    assert get_project(signed_in, project)["version"] == 1


@pytest.mark.parametrize("change", ["rename", "move"])
@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_rename_or_move_elsewhere_does_not_make_an_open_editors_save_stale(signed_in: Page, kind: str, change: str):
    project = open_with_edit(signed_in, kind)

    # the project list's rename and move dialogs send these requests
    if change == "rename":
        reply = put_project(signed_in, project, name="Renamed in the list")
    else:
        folder = create_folder(signed_in, "Elsewhere")["id"]
        reply = signed_in.request.post(f"/api/v1/projects/{project}/move", data={"new_folder_id": folder})
    assert reply.ok
    press_save(signed_in)

    expect(toast(signed_in, "Project Saved")).to_be_visible()
    assert edit_is_stored(signed_in, project, kind)


@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_renaming_in_an_editor_does_not_make_its_next_save_stale(signed_in: Page, kind: str):
    if kind == "flat":
        project = create_project(signed_in, "Rename me", flat_document())["id"]
        open_flat_rules(signed_in, project)
        signed_in.locator("h1").locator("xpath=following-sibling::button").click()
    else:
        project = node_ui.open_tree(signed_in, node_ui.loan_document(), "Rename me")
        node_ui.open_menu(signed_in)
        signed_in.get_by_role("menuitem", name="Edit Project Details").click()
    signed_in.get_by_placeholder("Enter project name").fill("Renamed here")
    signed_in.get_by_role("button", name="Save Changes").click()
    expect(signed_in.get_by_role("dialog")).to_be_hidden()
    expect(toast(signed_in, "Project Updated" if kind == "flat" else "Project updated")).to_be_visible()

    if kind == "flat":
        signed_in.get_by_role("button", name="Add Rule").click()
    else:
        node_ui.nudge(signed_in, "root")
    press_save(signed_in)

    expect(toast(signed_in, "Project Saved")).to_be_visible()
    assert edit_is_stored(signed_in, project, kind)


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


def test_reloading_the_flat_rule_editor_with_unsaved_edits_asks_first(signed_in: Page):
    open_with_edit(signed_in, "flat")
    asked = []
    signed_in.on("dialog", lambda dialog: asked.append(dialog.type))

    signed_in.reload()

    assert "beforeunload" in asked
