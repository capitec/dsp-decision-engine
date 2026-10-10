"""Sharing projects and folders between two users: who sees what, who may save, who may delete."""

import re

import pytest
from playwright.sync_api import Page, expect

import node_ui
from flat_rules_ui import open_flat_rules
from platform_fixtures import Other, other  # noqa: F401  (the fixture)
from platform_ui import (
    expand_folder,
    folder_row,
    open_projects,
    project_card,
    project_action,
    select_folder,
    share_folder_with,
    share_project_with,
    toast,
)
from seed import create_folder, create_project, get_project, list_folders, list_projects, share_folder, share_project
from test_flat_rule_params import flat_document

PUT = lambda r: r.request.method == "PUT" and "/api/v1/projects/" in r.url  # noqa: E731


def attempt_save(page: Page) -> int:
    """Press Save; the server's status."""
    with page.expect_response(PUT) as reply:
        page.get_by_role("button", name="Save", exact=True).click()
    return reply.value.status


def open_editor(page: Page, kind: str, project: str) -> None:
    if kind == "flat":
        open_flat_rules(page, project)
    else:
        page.goto(f"/projects/{project}")
        expect(page.get_by_role("button", name="Add Node")).to_be_visible(timeout=20_000)


def make_edit(page: Page, kind: str) -> None:
    if kind == "flat":
        page.get_by_role("button", name="Add Rule").click()
        expect(page.get_by_test_id("rule-card")).to_have_count(2)
    else:
        node_ui.nudge(page, "root")


def seed_shared(owner: Page, other: Other, kind: str, mode: str) -> str:
    if kind == "flat":
        project = create_project(owner, "Shared", flat_document())["id"]
    else:
        project = create_project(owner, "Shared", node_ui.loan_document(), project_type="tree")["id"]
    share_project(owner, project, other.id, mode)
    return project


def stored_document(owner: Page, project: str) -> dict:
    return get_project(owner, project)["project_data"]


# ---- a project shared with one user ---------------------------------------------------------------------------------

@pytest.mark.smoke
def test_a_project_shared_through_the_dialog_appears_for_the_other_user(signed_in: Page, other: Other):
    create_project(signed_in, "For you", flat_document())
    open_projects(signed_in)

    share_project_with(signed_in, "For you", other.email)

    open_projects(other.page)
    expect(project_card(other.page, "For you")).to_be_visible()
    open_flat = project_card(other.page, "For you").get_by_role("button", name="Open Project")
    open_flat.click()
    expect(other.page.get_by_role("button", name="Add Rule")).to_be_visible()


def test_the_other_user_sees_nothing_before_a_share(signed_in: Page, other: Other):
    create_project(signed_in, "Private", flat_document())
    create_folder(signed_in, "Private folder")

    open_projects(other.page)

    expect(other.page.get_by_text("No projects found")).to_be_visible()
    assert list_projects(other.page)["total"] == 0
    assert list_folders(other.page) == []


def test_the_dialog_lists_a_share_and_removing_it_takes_the_project_away(signed_in: Page, other: Other):
    create_project(signed_in, "Revocable", flat_document())
    open_projects(signed_in)
    share_project_with(signed_in, "Revocable", other.email, "Read & Write")
    dialog = signed_in.get_by_role("dialog")

    dialog.get_by_role("tab", name=re.compile("Manage Shares")).click()
    row = dialog.get_by_test_id("share-row")
    expect(row).to_contain_text(other.email)
    expect(row).to_contain_text("Read & Write")
    row.get_by_test_id("share-remove").click()

    expect(dialog.get_by_test_id("share-row")).to_have_count(0)
    open_projects(other.page)
    expect(project_card(other.page, "Revocable")).to_have_count(0)


def test_overview_of_the_other_user_counts_and_labels_a_shared_project(signed_in: Page, other: Other):
    project = create_project(signed_in, "Counted share", flat_document())["id"]
    share_project(signed_in, project, other.id, "read")

    other.page.goto("/")

    expect(other.page.get_by_text("Total Projects").locator("xpath=following-sibling::*")).to_have_text("1")
    recent = other.page.get_by_role("heading", name="Counted share").locator("xpath=ancestor::div[.//text()='read'][1]")
    expect(recent.get_by_text("read", exact=True)).to_be_visible()
    expect(recent.get_by_text("Owner")).to_have_count(0)


@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_read_write_user_can_save_and_the_owner_sees_it(signed_in: Page, other: Other, kind: str):
    project = seed_shared(signed_in, other, kind, "write")
    open_editor(other.page, kind, project)

    make_edit(other.page, kind)

    assert attempt_save(other.page) == 200
    expect(toast(other.page, "Saved")).to_be_visible()
    assert stored_document(signed_in, project) != (flat_document() if kind == "flat" else node_ui.loan_document())


@pytest.mark.parametrize("kind", ["flat", "tree"])
def test_a_read_only_users_save_is_refused_with_a_message_and_stores_nothing(signed_in: Page, other: Other, kind: str):
    project = seed_shared(signed_in, other, kind, "read")
    before = stored_document(signed_in, project)
    open_editor(other.page, kind, project)

    make_edit(other.page, kind)

    assert attempt_save(other.page) == 403
    expect(toast(other.page, "Save Failed")).to_contain_text("Insufficient permissions")
    expect(other.page.get_by_text("Unsaved changes")).to_be_visible()
    assert stored_document(signed_in, project) == before


@pytest.mark.parametrize("mode", ["read", "write"])
def test_a_user_the_project_is_shared_with_cannot_delete_it(signed_in: Page, other: Other, mode: str):
    project = create_project(signed_in, "Not yours", flat_document())["id"]
    share_project(signed_in, project, other.id, mode)

    reply = other.page.request.delete(f"/api/v1/projects/{project}")

    assert reply.status == 403
    assert get_project(signed_in, project)["name"] == "Not yours"


def test_deleting_a_shared_project_from_the_list_tells_the_user_and_keeps_it(signed_in: Page, other: Other):
    project = create_project(signed_in, "Not yours", flat_document())["id"]
    share_project(signed_in, project, other.id, "write")
    open_projects(other.page)

    project_action(other.page, "Not yours", "Delete")
    other.page.get_by_role("alertdialog").get_by_role("button", name="Delete Project").click()

    expect(other.page.get_by_text("Failed to delete project")).to_be_visible()
    assert get_project(signed_in, project)["name"] == "Not yours"


def test_a_read_only_user_cannot_share_the_project_onward(signed_in: Page, other: Other):
    project = create_project(signed_in, "Closed", flat_document())["id"]
    share_project(signed_in, project, other.id, "read")

    reply = other.page.request.post(f"/api/v1/projects/{project}/shares",
                                    data={"share_type": "user", "user_id": other.id, "access_mode": "admin"})

    assert reply.status == 403


def test_removing_a_share_takes_the_access_away(signed_in: Page, other: Other):
    project = create_project(signed_in, "Temporary", flat_document())["id"]
    share = share_project(signed_in, project, other.id, "write")
    assert other.page.request.get(f"/api/v1/projects/{project}").ok
    assert signed_in.request.delete(f"/api/v1/projects/{project}/shares/{share['id']}").ok

    assert other.page.request.get(f"/api/v1/projects/{project}").status in (403, 404)


# ---- another user's unshared project ---------------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["projects", "flat-rules"])
def test_opening_another_users_unshared_project_by_url_shows_no_data(signed_in: Page, other: Other, route: str):
    project = create_project(signed_in, "Secret plans", flat_document())["id"]

    other.page.goto(f"/{route}/{project}")

    expect(other.page.get_by_text(re.compile("not found|forbidden|no access|permission|not authorized", re.I)).first).to_be_visible()
    expect(other.page.get_by_text("Secret plans")).to_have_count(0)
    expect(other.page.get_by_role("button", name="Add Rule")).to_have_count(0)


@pytest.mark.parametrize("verb,path", [("get", ""), ("put", ""), ("get", "/export"), ("get", "/shares")])
def test_the_api_refuses_another_users_unshared_project(signed_in: Page, other: Other, verb: str, path: str):
    project = create_project(signed_in, "Secret plans", flat_document())["id"]
    options = {"data": {"name": "taken over"}} if verb == "put" else {}

    reply = getattr(other.page.request, verb)(f"/api/v1/projects/{project}{path}", **options)

    assert reply.status in (403, 404), reply.text()
    assert get_project(signed_in, project)["name"] == "Secret plans"


def test_another_users_unshared_project_cannot_be_deleted(signed_in: Page, other: Other):
    project = create_project(signed_in, "Secret plans", flat_document())["id"]

    assert other.page.request.delete(f"/api/v1/projects/{project}").status in (403, 404)
    assert get_project(signed_in, project)["name"] == "Secret plans"


def test_a_missing_project_id_is_a_clear_not_found(signed_in: Page):
    signed_in.goto("/flat-rules/00000000-0000-4000-8000-000000000000")

    expect(signed_in.get_by_text(re.compile("not found", re.I)).first).to_be_visible()


def test_a_malformed_project_id_in_the_url_is_a_clear_error(signed_in: Page):
    signed_in.goto("/projects/not-a-uuid")

    expect(signed_in.get_by_text(re.compile("error|not found|invalid", re.I)).first).to_be_visible()
    expect(signed_in.get_by_role("button", name="Add Node")).to_have_count(0)


# ---- folders ----------------------------------------------------------------------------------------------------------

FOLDER_PROJECTS_MISSING = "a folder share does not put the projects inside the folder in the recipient's project list (only owned and directly shared projects are listed)"
FOLDER_CHILDREN_MISSING = "the folder list omits subfolders reachable only through a shared parent folder"


def test_a_folder_shared_through_the_dialog_shows_the_folder_to_the_other_user(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Team folder")["id"]
    project = create_project(signed_in, "Inside shared", flat_document(), folder_id=folder)["id"]
    open_projects(signed_in)

    share_folder_with(signed_in, "Team folder", other.email)

    open_projects(other.page)
    expect(folder_row(other.page, "Team folder")).to_be_visible()
    assert other.page.request.get(f"/api/v1/projects/{project}").ok


@pytest.mark.xfail(strict=True, reason=FOLDER_PROJECTS_MISSING)
def test_the_projects_in_a_shared_folder_are_listed_for_the_other_user(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Team folder")["id"]
    create_project(signed_in, "Inside shared", flat_document(), folder_id=folder)
    share_folder(signed_in, folder, other.id, "read")
    open_projects(other.page)

    select_folder(other.page, "Team folder")

    expect(project_card(other.page, "Inside shared")).to_be_visible()


def test_a_read_only_folder_is_marked_with_a_lock_for_the_other_user(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Look only")["id"]
    share_folder(signed_in, folder, other.id, "read")

    open_projects(other.page)

    expect(folder_row(other.page, "Look only").get_by_title("Read-only access")).to_be_visible()


def test_a_folder_share_gives_access_to_projects_in_its_subfolders(signed_in: Page, other: Other):
    top = create_folder(signed_in, "Top")
    low = create_folder(signed_in, "Low", parent_id=top["id"])
    project = create_project(signed_in, "Deep inside", flat_document(), folder_id=low["id"])["id"]
    share_folder(signed_in, top["id"], other.id, "read")

    assert other.page.request.get(f"/api/v1/projects/{project}").ok


@pytest.mark.xfail(strict=True, reason=FOLDER_CHILDREN_MISSING)
def test_the_other_user_can_open_a_subfolder_of_a_shared_folder(signed_in: Page, other: Other):
    top = create_folder(signed_in, "Top")
    create_folder(signed_in, "Low", parent_id=top["id"])
    share_folder(signed_in, top["id"], other.id, "read")
    open_projects(other.page)
    expect(folder_row(other.page, "Top").get_by_label("Expand folder")).to_be_visible()

    expand_folder(other.page, "Top")

    expect(folder_row(other.page, "Low")).to_be_visible()


def test_a_project_in_a_read_only_folder_cannot_be_saved_by_the_other_user(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Look only")["id"]
    project = create_project(signed_in, "Inside", flat_document(), folder_id=folder)["id"]
    share_folder(signed_in, folder, other.id, "read")

    reply = other.page.request.put(f"/api/v1/projects/{project}", data={"name": "changed"})

    assert reply.status == 403
    assert get_project(signed_in, project)["name"] == "Inside"


def test_a_project_in_a_read_write_folder_can_be_saved_by_the_other_user(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Both write")["id"]
    project = create_project(signed_in, "Inside", flat_document(), folder_id=folder)["id"]
    share_folder(signed_in, folder, other.id, "write")

    reply = other.page.request.put(f"/api/v1/projects/{project}", data={"name": "changed"})

    assert reply.ok
    assert get_project(signed_in, project)["name"] == "changed"


def test_a_read_only_folder_cannot_get_new_subfolders_or_be_deleted_by_the_other_user(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Look only")["id"]
    share_folder(signed_in, folder, other.id, "read")

    created = other.page.request.post("/api/v1/folders", data={"name": "Sneaky", "parent_id": folder})
    deleted = other.page.request.delete(f"/api/v1/folders/{folder}")

    assert (created.status, deleted.status) == (403, 403)
    assert [f["name"] for f in list_folders(signed_in)] == ["Look only"]


def test_another_users_unshared_folder_is_refused_by_the_api(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Hidden")["id"]

    reads = other.page.request.get(f"/api/v1/folders/{folder}")
    renames = other.page.request.put(f"/api/v1/folders/{folder}", data={"name": "mine now"})

    assert reads.status in (403, 404), reads.text()
    assert renames.status in (403, 404), renames.text()
    assert [f["name"] for f in list_folders(signed_in)] == ["Hidden"]


def test_removing_a_folder_share_takes_the_folder_and_its_projects_away(signed_in: Page, other: Other):
    folder = create_folder(signed_in, "Brief access")["id"]
    project = create_project(signed_in, "Inside", flat_document(), folder_id=folder)["id"]
    share = share_folder(signed_in, folder, other.id, "read")
    assert other.page.request.get(f"/api/v1/projects/{project}").ok
    assert signed_in.request.delete(f"/api/v1/folders/{folder}/shares/{share['id']}").ok

    open_projects(other.page)

    expect(other.page.get_by_text("No folders yet")).to_be_visible()
    expect(project_card(other.page, "Inside")).to_have_count(0)
    assert list_folders(other.page) == []
