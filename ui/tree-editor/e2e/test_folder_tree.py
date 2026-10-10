"""Folders: create, rename, move, delete (empty and with contents), nesting and the breadcrumbs."""

import pytest
from playwright.sync_api import Page, expect

from platform_ui import (
    confirm_folder_delete,
    create_folder as create_folder_ui,
    expand_folder,
    folder_action,
    folder_row,
    open_projects,
    project_card,
    select_all_projects,
    select_folder,
    toast,
)
from seed import create_folder, create_project, get_project, list_folders
from test_flat_rule_params import flat_document


def folder_named(page: Page, name: str) -> dict:
    return next(f for f in list_folders(page) if f["name"] == name)


def breadcrumb(page: Page):
    return page.get_by_label("Breadcrumb")


def three_levels(page: Page) -> tuple[dict, dict, dict]:
    top = create_folder(page, "Level 1")
    middle = create_folder(page, "Level 2", parent_id=top["id"])
    bottom = create_folder(page, "Level 3", parent_id=middle["id"])
    return top, middle, bottom


# ---- create ---------------------------------------------------------------------------------------------------------

@pytest.mark.smoke
def test_a_new_folder_appears_in_the_tree_and_is_stored(signed_in: Page):
    open_projects(signed_in)

    create_folder_ui(signed_in, "Mortgages", description="Home loans")

    expect(folder_row(signed_in, "Mortgages")).to_be_visible()
    stored = folder_named(signed_in, "Mortgages")
    assert (stored["description"], stored["parent_id"]) == ("Home loans", None)


def test_a_folder_needs_a_name(signed_in: Page):
    open_projects(signed_in)

    signed_in.get_by_role("button", name="New Folder").click()

    expect(signed_in.get_by_role("dialog").get_by_role("button", name="Create Folder")).to_be_disabled()


def test_a_duplicate_folder_name_in_the_same_place_is_not_stored_twice(signed_in: Page):
    create_folder(signed_in, "Twice")
    open_projects(signed_in)

    create_folder_ui(signed_in, "Twice", wait=False)
    expect(toast(signed_in, "Creation failed")).to_be_visible()

    assert [f["name"] for f in list_folders(signed_in)] == ["Twice"]


@pytest.mark.xfail(strict=True, reason="the server's reason (already exists) is dropped: the toast only says 'Failed to create folder' and the dialog's own duplicate check never matches root folders")
def test_a_duplicate_folder_name_says_that_it_already_exists(signed_in: Page):
    create_folder(signed_in, "Twice")
    open_projects(signed_in)

    create_folder_ui(signed_in, "Twice", wait=False)

    expect(signed_in.get_by_text("already exists").first).to_be_visible()


@pytest.mark.xfail(strict=True, reason="a refused folder create puts the sidebar into its 'Failed to load folders' state")
def test_a_refused_folder_create_leaves_the_folder_tree_alone(signed_in: Page):
    create_folder(signed_in, "Twice")
    open_projects(signed_in)

    create_folder_ui(signed_in, "Twice", wait=False)
    expect(toast(signed_in, "Creation failed")).to_be_visible()

    expect(signed_in.get_by_text("Failed to load folders")).to_have_count(0)
    expect(folder_row(signed_in, "Twice")).to_be_visible()


def test_the_same_folder_name_is_allowed_under_different_parents(signed_in: Page):
    a = create_folder(signed_in, "Parent A")
    create_folder(signed_in, "Parent B")
    create_folder(signed_in, "Archive", parent_id=a["id"])
    open_projects(signed_in)
    expand_folder(signed_in, "Parent A")
    select_folder(signed_in, "Parent A")

    create_folder_ui(signed_in, "Archive", parent="Parent B")

    assert sorted(f["name"] for f in list_folders(signed_in)) == ["Archive", "Archive", "Parent A", "Parent B"]


def test_a_folder_created_while_inside_another_defaults_to_being_its_subfolder(signed_in: Page):
    parent = create_folder(signed_in, "Outer")
    open_projects(signed_in)
    select_folder(signed_in, "Outer")

    create_folder_ui(signed_in, "Inner")

    assert folder_named(signed_in, "Inner")["parent_id"] == parent["id"]


@pytest.mark.xfail(strict=True, reason="Create Subfolder in a folder's menu ignores which folder it was opened on: the new folder lands in the open folder or the root")
def test_create_subfolder_from_the_menu_creates_it_inside_that_folder(signed_in: Page):
    target = create_folder(signed_in, "Target")
    create_folder(signed_in, "Bystander")
    open_projects(signed_in)

    folder_action(signed_in, "Target", "Create Subfolder")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_placeholder("Enter folder name").fill("Child")
    dialog.get_by_role("button", name="Create Folder").click()
    expect(toast(signed_in, "Folder created")).to_be_visible()

    assert folder_named(signed_in, "Child")["parent_id"] == target["id"]


# ---- rename ---------------------------------------------------------------------------------------------------------

def test_renaming_a_folder_changes_name_and_description(signed_in: Page):
    create_folder(signed_in, "Old name", description="old text")
    open_projects(signed_in)

    folder_action(signed_in, "Old name", "Rename")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_placeholder("Enter folder name").fill("New name")
    dialog.get_by_placeholder("Enter folder description (optional)").fill("new text")
    dialog.get_by_role("button").filter(has_text="Save").click()

    expect(folder_row(signed_in, "New name")).to_be_visible()
    expect(folder_row(signed_in, "Old name")).to_have_count(0)
    signed_in.reload()
    expect(folder_row(signed_in, "New name")).to_be_visible()
    assert folder_named(signed_in, "New name")["description"] == "new text"


def test_renaming_a_folder_keeps_its_projects(signed_in: Page):
    folder = create_folder(signed_in, "Before")["id"]
    create_project(signed_in, "Inside", flat_document(), folder_id=folder)
    open_projects(signed_in)

    folder_action(signed_in, "Before", "Rename")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_placeholder("Enter folder name").fill("After")
    dialog.get_by_role("button").filter(has_text="Save").click()
    select_folder(signed_in, "After")

    expect(project_card(signed_in, "Inside")).to_be_visible()


# ---- delete ---------------------------------------------------------------------------------------------------------

def test_deleting_an_empty_folder_asks_once_and_removes_it(signed_in: Page):
    create_folder(signed_in, "Empty one")
    open_projects(signed_in)

    folder_action(signed_in, "Empty one", "Delete")
    dialog = signed_in.get_by_role("alertdialog")
    expect(dialog.get_by_text("This folder is empty")).to_be_visible()
    confirm_folder_delete(signed_in)

    expect(folder_row(signed_in, "Empty one")).to_have_count(0)
    assert list_folders(signed_in) == []


def test_cancelling_the_folder_delete_dialog_keeps_the_folder(signed_in: Page):
    create_folder(signed_in, "Stay")
    open_projects(signed_in)

    folder_action(signed_in, "Stay", "Delete")
    signed_in.get_by_role("alertdialog").get_by_role("button", name="Cancel").click()

    expect(folder_row(signed_in, "Stay")).to_be_visible()


@pytest.mark.smoke
def test_deleting_a_folder_with_contents_moves_them_to_the_root_by_default(signed_in: Page):
    folder = create_folder(signed_in, "Full")
    child = create_folder(signed_in, "Child folder", parent_id=folder["id"])
    project = create_project(signed_in, "Inside project", flat_document(), folder_id=folder["id"])["id"]
    open_projects(signed_in)

    folder_action(signed_in, "Full", "Delete")
    dialog = signed_in.get_by_role("alertdialog")
    expect(dialog.get_by_text("1 project and 1 subfolder")).to_be_visible()
    confirm_folder_delete(signed_in)

    expect(folder_row(signed_in, "Full")).to_have_count(0)
    expect(folder_row(signed_in, "Child folder")).to_be_visible()
    select_all_projects(signed_in)
    expect(project_card(signed_in, "Inside project")).to_be_visible()
    assert get_project(signed_in, project)["folder_id"] is None
    assert folder_named(signed_in, "Child folder")["parent_id"] is None
    assert child["id"] == folder_named(signed_in, "Child folder")["id"]


def test_deleting_a_folder_with_delete_all_removes_its_projects(signed_in: Page):
    folder = create_folder(signed_in, "Purge")["id"]
    doomed = create_project(signed_in, "Goes too", flat_document(), folder_id=folder)["id"]
    bystander = create_project(signed_in, "Stays", flat_document())["id"]
    open_projects(signed_in)

    folder_action(signed_in, "Purge", "Delete")
    confirm_folder_delete(signed_in, "Delete All", "Delete all contents")

    expect(folder_row(signed_in, "Purge")).to_have_count(0)
    assert signed_in.request.get(f"/api/v1/projects/{doomed}").status == 404
    assert signed_in.request.get(f"/api/v1/projects/{bystander}").ok


@pytest.mark.xfail(strict=True, reason="Delete all contents promises to delete subfolders permanently; the server moves them to the root instead")
def test_delete_all_contents_removes_the_subfolders_it_promised_to(signed_in: Page):
    folder = create_folder(signed_in, "Purge parent")
    create_folder(signed_in, "Promised gone", parent_id=folder["id"])
    open_projects(signed_in)

    folder_action(signed_in, "Purge parent", "Delete")
    confirm_folder_delete(signed_in, "Delete All", "Delete all contents")
    expect(folder_row(signed_in, "Purge parent")).to_have_count(0)

    assert [f["name"] for f in list_folders(signed_in)] == []


@pytest.mark.xfail(strict=True, reason="after deleting the open folder the heading says All Projects but the list shows none (it is still filtered by the deleted folder)")
def test_deleting_the_open_folder_returns_to_all_projects(signed_in: Page):
    create_folder(signed_in, "Open one")
    create_project(signed_in, "Root project", flat_document())
    open_projects(signed_in)
    select_folder(signed_in, "Open one")

    folder_action(signed_in, "Open one", "Delete")
    confirm_folder_delete(signed_in)

    expect(signed_in.get_by_role("heading", level=1, name="All Projects")).to_be_visible()
    expect(project_card(signed_in, "Root project")).to_be_visible()


# ---- nesting and breadcrumbs ------------------------------------------------------------------------------------------

@pytest.mark.smoke
def test_three_levels_of_folders_show_a_full_breadcrumb_and_each_level_navigates(signed_in: Page):
    top, middle, bottom = three_levels(signed_in)
    create_project(signed_in, "Deep project", flat_document(), folder_id=bottom["id"])
    open_projects(signed_in)

    expand_folder(signed_in, "Level 1")
    expand_folder(signed_in, "Level 2")
    select_folder(signed_in, "Level 3")

    expect(breadcrumb(signed_in)).to_contain_text("All Projects")
    expect(breadcrumb(signed_in).get_by_title("Level 1")).to_be_visible()
    expect(breadcrumb(signed_in).get_by_title("Level 2")).to_be_visible()
    expect(breadcrumb(signed_in).get_by_text("Level 3")).to_have_attribute("aria-current", "page")
    expect(project_card(signed_in, "Deep project")).to_be_visible()

    breadcrumb(signed_in).get_by_title("Level 2").click()
    expect(signed_in.get_by_role("heading", level=1, name="Level 2")).to_be_visible()
    expect(project_card(signed_in, "Deep project")).to_have_count(0)
    breadcrumb(signed_in).get_by_label("Navigate to root").click()
    expect(signed_in.get_by_role("heading", level=1, name="All Projects")).to_be_visible()
    expect(project_card(signed_in, "Deep project")).to_be_visible()


def test_a_project_can_be_created_in_the_third_level_through_the_api_and_found_by_search(signed_in: Page):
    _, _, bottom = three_levels(signed_in)
    create_project(signed_in, "Deep project", flat_document(), folder_id=bottom["id"])
    open_projects(signed_in)
    expand_folder(signed_in, "Level 1")
    expand_folder(signed_in, "Level 2")

    expect(folder_row(signed_in, "Level 3")).to_contain_text("1")


def test_collapsing_a_folder_hides_its_children(signed_in: Page):
    three_levels(signed_in)
    open_projects(signed_in)
    expand_folder(signed_in, "Level 1")
    expect(folder_row(signed_in, "Level 2")).to_be_visible()

    folder_row(signed_in, "Level 1").get_by_label("Collapse folder").click()

    expect(folder_row(signed_in, "Level 2")).to_have_count(0)


# ---- move -----------------------------------------------------------------------------------------------------------

MOVE_DIALOG_EMPTY = "the Move Folder dialog offers only 'Root (No parent)': no destination folder is listed, so a folder cannot be moved into another"


@pytest.mark.xfail(strict=True, reason=MOVE_DIALOG_EMPTY)
def test_a_folder_can_be_moved_into_another_and_back_to_the_root(signed_in: Page):
    create_folder(signed_in, "Mover")
    target = create_folder(signed_in, "Target")
    open_projects(signed_in)

    folder_action(signed_in, "Mover", "Move")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_label("Target").click()
    dialog.get_by_role("button", name="Move Folder").click()
    expect(toast(signed_in, "Folder moved")).to_be_visible()
    assert folder_named(signed_in, "Mover")["parent_id"] == target["id"]

    expand_folder(signed_in, "Target")
    folder_action(signed_in, "Mover", "Move")
    dialog = signed_in.get_by_role("dialog")
    dialog.get_by_label("Root (No parent)").click()
    dialog.get_by_role("button", name="Move Folder").click()
    expect(toast(signed_in, "Folder moved").last).to_be_visible()
    assert folder_named(signed_in, "Mover")["parent_id"] is None


@pytest.mark.xfail(strict=True, reason=MOVE_DIALOG_EMPTY)
def test_the_move_dialog_does_not_offer_a_folders_own_descendants(signed_in: Page):
    three_levels(signed_in)
    create_folder(signed_in, "Elsewhere")
    open_projects(signed_in)

    folder_action(signed_in, "Level 1", "Move")
    dialog = signed_in.get_by_role("dialog")

    expect(dialog.get_by_label("Elsewhere")).to_be_visible()
    expect(dialog.get_by_label("Level 2")).to_have_count(0)
    expect(dialog.get_by_label("Level 3")).to_have_count(0)


def test_moving_a_folder_takes_its_projects_and_subfolders_along(signed_in: Page):
    mover = create_folder(signed_in, "Mover")
    create_folder(signed_in, "Mover child", parent_id=mover["id"])
    create_project(signed_in, "Rides along", flat_document(), folder_id=mover["id"])
    target = create_folder(signed_in, "Target")
    assert signed_in.request.post(f"/api/v1/folders/{mover['id']}/move", data={"new_parent_id": target["id"]}).ok
    open_projects(signed_in)

    expand_folder(signed_in, "Target")
    expand_folder(signed_in, "Mover")

    expect(folder_row(signed_in, "Mover child")).to_be_visible()
    select_folder(signed_in, "Mover")
    expect(project_card(signed_in, "Rides along")).to_be_visible()


def test_a_folder_cannot_be_moved_into_its_own_descendant(signed_in: Page):
    top, _, bottom = three_levels(signed_in)

    r = signed_in.request.post(f"/api/v1/folders/{top['id']}/move", data={"new_parent_id": bottom["id"]})

    assert r.status == 400, r.text()
    assert folder_named(signed_in, "Level 1")["parent_id"] is None
