"""The project list: rename, move, delete, search, pagination, the overview counts and unusual names."""

import json
import re

import pytest
from playwright.sync_api import Locator, Page, expect

from flat_rules_ui import open_flat_rules
from platform_ui import (
    create_folder as create_folder_ui,
    delete_project,
    first_held,
    folder_row,
    open_from_list,
    open_projects,
    project_action,
    project_card,
    rename_project,
    select_all_projects,
    select_folder,
)
from seed import create_folder, create_project, get_project
from test_flat_rule_params import flat_document

UNUSUAL_NAMES = {
    "quotes": "He said \"hi\" & 'bye'",
    "slashes": r"a/b\c..\d",
    "script_tag": "<script>alert(1)</script><b>bold</b>",
    "emoji": "Loan \U0001F680 café 日本語",
    "long": "L" * 200,
}


def total_projects_on_overview(page: Page) -> Locator:
    page.goto("/")
    return page.get_by_text("Total Projects").locator("xpath=following-sibling::*")


def test_renaming_a_project_changes_name_and_description(signed_in: Page):
    project = create_project(signed_in, "Before", flat_document())["id"]
    open_projects(signed_in)

    rename_project(signed_in, "Before", "After", "What it does")

    expect(project_card(signed_in, "After")).to_contain_text("What it does")
    expect(project_card(signed_in, "Before")).to_have_count(0)
    signed_in.reload()
    expect(project_card(signed_in, "After")).to_contain_text("What it does")
    stored = get_project(signed_in, project)
    assert (stored["name"], stored["description"]) == ("After", "What it does")


def test_renaming_a_project_keeps_its_document(signed_in: Page):
    document = flat_document()
    project = create_project(signed_in, "Keep me", document)["id"]
    open_projects(signed_in)

    rename_project(signed_in, "Keep me", "Kept")

    assert get_project(signed_in, project)["project_data"] == document


def test_rename_dialog_needs_a_name(signed_in: Page):
    create_project(signed_in, "Named", flat_document())
    open_projects(signed_in)

    project_action(signed_in, "Named", "Rename")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_placeholder("Enter project name").fill("   ")

    expect(dialog.get_by_role("button", name="Save Changes")).to_be_disabled()


@pytest.mark.smoke
def test_deleting_a_project_from_the_list_removes_it_for_good(signed_in: Page):
    doomed = create_project(signed_in, "Doomed", flat_document())["id"]
    create_project(signed_in, "Survivor", flat_document())
    open_projects(signed_in)

    delete_project(signed_in, "Doomed")

    expect(project_card(signed_in, "Doomed")).to_have_count(0)
    expect(project_card(signed_in, "Survivor")).to_be_visible()
    assert signed_in.request.get(f"/api/v1/projects/{doomed}").status == 404


def test_cancelling_the_delete_dialog_keeps_the_project(signed_in: Page):
    create_project(signed_in, "Kept", flat_document())
    open_projects(signed_in)

    project_action(signed_in, "Kept", "Delete")
    signed_in.get_by_role("alertdialog").get_by_role("button", name="Cancel").click()

    expect(project_card(signed_in, "Kept")).to_be_visible()


def test_opening_a_flat_rule_project_from_the_list_shows_its_editor(signed_in: Page):
    create_project(signed_in, "Opened", flat_document())
    open_projects(signed_in)

    open_from_list(signed_in, "Opened")

    expect(signed_in).to_have_url(re.compile(r"/flat-rules/"))
    expect(signed_in.get_by_role("button", name="Add Rule")).to_be_visible()


def test_list_view_shows_the_same_projects_with_actions(signed_in: Page):
    create_project(signed_in, "Row one", flat_document())
    open_projects(signed_in)

    signed_in.locator("button:has(svg.lucide-list)").click()
    row = signed_in.get_by_test_id("project-row").filter(has_text="Row one")
    row.get_by_test_id("project-list-menu").click()
    signed_in.get_by_role("menuitem", name="Rename", exact=True).click()

    expect(signed_in.get_by_role("dialog").get_by_placeholder("Enter project name")).to_have_value("Row one")


# ---- moving between folders -------------------------------------------------------------------------------------

def test_a_project_moved_into_a_folder_shows_in_that_folder_and_in_all_projects(signed_in: Page):
    create_folder(signed_in, "Inbox")
    create_project(signed_in, "Mover", flat_document())
    create_project(signed_in, "Stayer", flat_document())
    open_projects(signed_in)

    project_action(signed_in, "Mover", "Move to Folder")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_role("button", name="Inbox").click()
    dialog.get_by_role("button", name="Move Project").click()
    expect(dialog).to_be_hidden()

    select_folder(signed_in, "Inbox")
    expect(project_card(signed_in, "Mover")).to_be_visible()
    expect(project_card(signed_in, "Stayer")).to_have_count(0)
    select_all_projects(signed_in)
    expect(project_card(signed_in, "Mover")).to_be_visible()
    expect(project_card(signed_in, "Stayer")).to_be_visible()


def test_a_project_can_be_moved_back_to_the_root(signed_in: Page):
    folder = create_folder(signed_in, "Box")["id"]
    project = create_project(signed_in, "Boxed", flat_document(), folder_id=folder)["id"]
    open_projects(signed_in)
    select_folder(signed_in, "Box")

    project_action(signed_in, "Boxed", "Move to Folder")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_role("button", name="Root (No folder)").click()
    dialog.get_by_role("button", name="Move Project").click()

    expect(project_card(signed_in, "Boxed")).to_have_count(0)
    assert get_project(signed_in, project)["folder_id"] is None


@pytest.mark.xfail(strict=True, reason="the move dialog keeps the folder that was current when the page loaded selected, so Move Project is enabled with the root preselected")
def test_the_move_dialog_marks_the_current_folder_and_offers_no_no_op(signed_in: Page):
    folder = create_folder(signed_in, "Here")["id"]
    create_project(signed_in, "Already", flat_document(), folder_id=folder)
    open_projects(signed_in)
    select_folder(signed_in, "Here")

    project_action(signed_in, "Already", "Move to Folder")
    dialog = signed_in.get_by_role("dialog")

    expect(dialog.get_by_text("Here (current)")).to_be_visible()
    expect(dialog.get_by_role("button", name="Move Project")).to_be_disabled()


# ---- overview counts ----------------------------------------------------------------------------------------------

def test_overview_counts_follow_creating_and_deleting_projects(signed_in: Page):
    ids = [create_project(signed_in, f"Counted {i}", flat_document())["id"] for i in range(3)]
    expect(total_projects_on_overview(signed_in)).to_have_text("3")
    signed_in.request.delete(f"/api/v1/projects/{ids[0]}")

    expect(total_projects_on_overview(signed_in)).to_have_text("2")


def test_overview_lists_the_recent_projects_and_opens_one(signed_in: Page):
    create_project(signed_in, "Recent one", flat_document())
    signed_in.goto("/")

    signed_in.get_by_role("heading", name="Recent one").click()

    expect(signed_in).to_have_url(re.compile(r"/flat-rules/"))


def test_overview_of_a_new_user_invites_creating_a_project(signed_in: Page):
    signed_in.goto("/")

    expect(signed_in.get_by_text("No projects yet")).to_be_visible()


# ---- search and pagination ----------------------------------------------------------------------------------------

def seed_many(page: Page, count: int, prefix: str = "Bulk") -> None:
    document = flat_document()
    for i in range(count):
        create_project(page, f"{prefix} {i:02d}", document)


def test_a_second_page_of_projects_is_reachable(signed_in: Page):
    seed_many(signed_in, 14)
    open_projects(signed_in)

    expect(signed_in.get_by_test_id("project-card")).to_have_count(12)
    expect(signed_in.get_by_text("Showing 1-12 of 14 projects")).to_be_visible()
    signed_in.get_by_role("button", name="Next").click()

    expect(signed_in.get_by_text("Showing 13-14 of 14 projects")).to_be_visible()
    expect(signed_in.get_by_test_id("project-card")).to_have_count(2)
    expect(signed_in.get_by_role("button", name="Next")).to_be_disabled()


def test_every_project_appears_on_exactly_one_page(signed_in: Page):
    seed_many(signed_in, 14)
    open_projects(signed_in)
    names = []
    names += signed_in.get_by_test_id("project-card").locator("h3, [class*=text-lg]").all_inner_texts()
    signed_in.get_by_role("button", name="Next").click()
    expect(signed_in.get_by_text("Showing 13-14 of 14 projects")).to_be_visible()
    names += signed_in.get_by_test_id("project-card").locator("h3, [class*=text-lg]").all_inner_texts()

    assert sorted(names) == [f"Bulk {i:02d}" for i in range(14)]


def test_the_project_list_pages_most_recently_changed_first(signed_in: Page):
    ids = [create_project(signed_in, name, flat_document())["id"] for name in ("A", "B", "C")]
    signed_in.request.put(f"/api/v1/projects/{ids[0]}", data={"description": "changed"})

    pages = [signed_in.request.get(f"/api/v1/projects?page={n}&size=2").json()["items"] for n in (1, 2)]

    assert [[p["name"] for p in items] for items in pages] == [["A", "C"], ["B"]]


def test_page_size_can_be_changed(signed_in: Page):
    seed_many(signed_in, 8)
    open_projects(signed_in)

    signed_in.get_by_role("combobox").click()
    signed_in.get_by_role("option", name="6", exact=True).click()

    expect(signed_in.get_by_test_id("project-card")).to_have_count(6)
    expect(signed_in.get_by_text("Showing 1-6 of 8 projects")).to_be_visible()


def test_search_narrows_the_list_by_name_and_by_description(signed_in: Page):
    create_project(signed_in, "Mortgage rules", flat_document())
    create_project(signed_in, "Card limits", flat_document())
    signed_in.request.post("/api/v1/projects", data={
        "name": "Plain name", "project_type": "flat_rule", "description": "about mortgage pricing"})
    open_projects(signed_in)

    signed_in.get_by_placeholder("Search projects...").fill("mortgage")

    expect(signed_in.get_by_test_id("project-card")).to_have_count(2)
    expect(project_card(signed_in, "Card limits")).to_have_count(0)


def test_search_with_no_match_says_so(signed_in: Page):
    create_project(signed_in, "Something", flat_document())
    open_projects(signed_in)

    signed_in.get_by_placeholder("Search projects...").fill("zzzzqq")

    expect(signed_in.get_by_text("No projects match your search")).to_be_visible()


@pytest.mark.xfail(strict=True, reason="an older list response that arrives last replaces the newer one: the search result is lost")
def test_a_slow_older_list_response_does_not_replace_the_newer_search_result(signed_in: Page):
    seed_many(signed_in, 14)
    create_project(signed_in, "Needle", flat_document())
    open_projects(signed_in)
    signed_in.get_by_role("button", name="Next").click()
    expect(signed_in.get_by_text("Showing 13-15 of 15 projects")).to_be_visible()
    held = []
    late = re.compile(r"/api/v1/projects\?page=2&size=12&search=Needle")
    signed_in.route(late, lambda route: held.append(route))

    signed_in.get_by_placeholder("Search projects...").fill("Needle")
    expect(project_card(signed_in, "Needle")).to_be_visible()
    with signed_in.expect_response(late):
        first_held(signed_in, held).continue_()
        signed_in.unroute(late)
    for _ in range(2):  # let the page render what the late response carried
        signed_in.evaluate("new Promise(resolve => requestAnimationFrame(resolve))")

    expect(project_card(signed_in, "Needle")).to_be_visible()


def test_search_for_a_percent_sign_matches_only_names_containing_one(signed_in: Page):
    create_project(signed_in, "Discount 50% off", flat_document())
    create_project(signed_in, "No symbols here", flat_document())
    open_projects(signed_in)

    signed_in.get_by_placeholder("Search projects...").fill("%")

    expect(signed_in.get_by_test_id("project-card")).to_have_count(1)


def test_search_inside_a_folder_only_looks_in_that_folder(signed_in: Page):
    folder = create_folder(signed_in, "Only here")["id"]
    create_project(signed_in, "Report inside", flat_document(), folder_id=folder)
    create_project(signed_in, "Report outside", flat_document())
    open_projects(signed_in)
    select_folder(signed_in, "Only here")

    signed_in.get_by_placeholder("Search projects...").fill("Report")

    expect(project_card(signed_in, "Report inside")).to_be_visible()
    expect(project_card(signed_in, "Report outside")).to_have_count(0)


# ---- unusual names ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", UNUSUAL_NAMES.values(), ids=UNUSUAL_NAMES.keys())
def test_unusual_project_names_render_as_text_and_survive_reload(signed_in: Page, name: str):
    dialogs = []
    signed_in.on("dialog", lambda dialog: dialogs.append(dialog.message))
    create_project(signed_in, name, flat_document())
    open_projects(signed_in)

    expect(project_card(signed_in, name)).to_be_visible()
    signed_in.reload()

    expect(project_card(signed_in, name)).to_be_visible()
    assert dialogs == []
    assert signed_in.locator("script", has_text="alert(1)").count() == 0


@pytest.mark.parametrize("name", UNUSUAL_NAMES.values(), ids=UNUSUAL_NAMES.keys())
def test_renaming_a_project_to_an_unusual_name_stores_it_unchanged(signed_in: Page, name: str):
    project = create_project(signed_in, "Plain", flat_document())["id"]
    open_projects(signed_in)

    rename_project(signed_in, "Plain", name, description=name)

    expect(project_card(signed_in, name)).to_be_visible()
    stored = get_project(signed_in, project)
    assert (stored["name"], stored["description"]) == (name, name)


@pytest.mark.parametrize("name", UNUSUAL_NAMES.values(), ids=UNUSUAL_NAMES.keys())
def test_unusual_folder_names_render_as_text_and_survive_reload(signed_in: Page, name: str):
    dialogs = []
    signed_in.on("dialog", lambda dialog: dialogs.append(dialog.message))
    open_projects(signed_in)

    create_folder_ui(signed_in, name, description=name)
    select_folder(signed_in, name)
    signed_in.reload()

    expect(folder_row(signed_in, name)).to_be_visible()
    assert dialogs == []
    assert [f["name"] for f in signed_in.request.get("/api/v1/folders").json()] == [name]


@pytest.mark.parametrize("name", list(UNUSUAL_NAMES.values()), ids=list(UNUSUAL_NAMES))
def test_exporting_a_project_with_an_unusual_name_from_the_list_downloads_a_safe_file_name(signed_in: Page, name: str, tmp_path):
    document = flat_document()
    create_project(signed_in, name, document)
    open_projects(signed_in)

    with signed_in.expect_download() as download:
        project_action(signed_in, name, "Export")

    filename = download.value.suggested_filename
    assert filename.endswith("_export.json")
    assert not re.search(r'[/\\"<>]', filename)
    path = tmp_path / "export.json"
    download.value.save_as(path)
    assert json.loads(path.read_text())["type"] == document["type"]


@pytest.mark.parametrize("name", list(UNUSUAL_NAMES.values()), ids=list(UNUSUAL_NAMES))
def test_exporting_from_the_editor_a_project_with_an_unusual_name_downloads_a_safe_file_name(signed_in: Page, name: str, tmp_path):
    project = create_project(signed_in, name, flat_document())["id"]
    open_flat_rules(signed_in, project)

    with signed_in.expect_download() as download:
        signed_in.get_by_role("button", name="Export").click()

    filename = download.value.suggested_filename
    assert filename.endswith("_export.json")
    assert not re.search(r'[/\\"<>]', filename)
