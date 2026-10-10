"""Project list, folder and sharing actions of the platform pages: each does what a user does, nothing more."""

from playwright.sync_api import Locator, Page, expect

import node_ui
from flat_rules_ui import open_flat_rules
from seed import create_project, get_project


def open_projects(page: Page) -> None:
    page.goto("/projects")
    expect(page.get_by_role("heading", level=1, name="All Projects")).to_be_visible()
    expect(page.get_by_text("Loading projects...")).to_be_hidden()


def toast(page: Page, text: str) -> Locator:
    """The toast whose title or description contains `text`."""
    return page.locator("li[data-swipe-direction]").filter(has_text=text)


def project_card(page: Page, name: str) -> Locator:
    return page.get_by_test_id("project-card").filter(has=page.get_by_text(name, exact=True))


def project_action(page: Page, name: str, item: str) -> None:
    """Open the card's menu of the project `name` and pick `item` (Share, Export, Rename, Move to Folder, Delete)."""
    project_card(page, name).get_by_test_id("project-list-menu").click()
    page.get_by_role("menuitem", name=item, exact=True).click()


def open_project(page: Page, name: str) -> None:
    project_card(page, name).get_by_role("button", name="Open Project").click()


def rename_project(page: Page, name: str, new_name: str, description: str | None = None) -> None:
    """Rename dialog of the project list."""
    project_action(page, name, "Rename")
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Enter project name").fill(new_name)
    if description is not None:
        dialog.get_by_placeholder("Enter project description (optional)").fill(description)
    dialog.get_by_role("button", name="Save Changes").click()
    expect(dialog).to_be_hidden()


def delete_project(page: Page, name: str) -> None:
    project_action(page, name, "Delete")
    page.get_by_role("alertdialog").get_by_role("button", name="Delete Project").click()


def folder_row(page: Page, name: str) -> Locator:
    """The folder's row in the sidebar tree (nested folders are visible only once their parent is expanded)."""
    return page.get_by_test_id("folder-row").filter(has=page.get_by_role("button", name=name, exact=True))


def folder_action(page: Page, name: str, item: str) -> None:
    """Open the folder row's menu and pick `item` (Create Subfolder, Rename, Share, Move, Delete)."""
    row = folder_row(page, name)
    row.hover()
    row.get_by_test_id("folder-menu").click()
    page.get_by_role("menuitem", name=item, exact=True).click()


def expand_folder(page: Page, name: str) -> None:
    folder_row(page, name).get_by_label("Expand folder").click()


def select_folder(page: Page, name: str) -> None:
    """Open the folder in the sidebar: the list shows its projects and the heading shows its name."""
    folder_row(page, name).get_by_role("button", name=name, exact=True).click()
    expect(page.get_by_role("heading", level=1, name=name, exact=True)).to_be_visible()


def select_all_projects(page: Page) -> None:
    page.get_by_text("All Projects", exact=True).first.click()
    expect(page.get_by_role("heading", level=1, name="All Projects")).to_be_visible()


def create_folder(page: Page, name: str, description: str | None = None, parent: str | None = None,
                  wait: bool = True) -> None:
    """Sidebar New Folder dialog; `parent` is the folder name chosen in the Parent Folder select. `wait`: until it closed."""
    page.get_by_role("button", name="New Folder").click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Enter folder name").fill(name)
    if description is not None:
        dialog.get_by_placeholder("Enter folder description (optional)").fill(description)
    if parent is not None:
        dialog.get_by_label("Parent Folder").click()
        page.get_by_role("option", name=parent, exact=True).click()
    dialog.get_by_role("button", name="Create Folder").click()
    if wait:
        expect(dialog).to_be_hidden()


def confirm_folder_delete(page: Page, button: str = "Delete Folder", option: str | None = None) -> None:
    """Pick a delete option (the dialog's radio label) when given, press `button`, and wait for the server's answer."""
    dialog = page.get_by_role("alertdialog")
    if option is not None:
        dialog.get_by_label(option).click()
    with page.expect_response(lambda r: r.request.method == "DELETE" and "/api/v1/folders/" in r.url):
        dialog.get_by_role("button", name=button).click()


def share_project_with(page: Page, project: str, email: str, access: str = "Read Only") -> None:
    """Share dialog of a project card: search the user by email, pick the access level, press Share."""
    project_action(page, project, "Share")
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Search by email...").fill(email)
    if access != "Read Only":
        dialog.get_by_role("combobox").click()
        page.get_by_role("option", name=access, exact=True).click()
    dialog.get_by_role("button", name="Share", exact=True).click()
    expect(dialog.get_by_role("tab", name="Manage Shares (1)")).to_be_visible()


def share_folder_with(page: Page, folder: str, email: str, access: str = "Read Only") -> None:
    """Share dialog of a folder row: search the user, pick the access level, press Share."""
    folder_action(page, folder, "Share")
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Enter email or name...").fill(email)
    if access != "Read Only":
        dialog.get_by_label("Access Level").click()
        page.get_by_role("option", name=access).click()
    dialog.get_by_role("button", name="Share", exact=True).click()
    expect(toast(page, "Share added")).to_be_visible()


def open_with_edit(page: Page, kind: str, name: str = "Editing") -> str:
    """Seed and open a project of `kind` ('flat' or 'tree') and make one unsaved edit; returns the project id."""
    from test_flat_rule_params import flat_document

    if kind == "flat":
        project = create_project(page, name, flat_document())["id"]
        open_flat_rules(page, project)
        page.get_by_role("button", name="Add Rule").click()
        expect(page.get_by_test_id("rule-card")).to_have_count(2)
    else:
        project = node_ui.open_tree(page, node_ui.loan_document(), name)
        node_ui.nudge(page, "root")
        expect(page.get_by_text("Unsaved changes")).to_be_visible()
    return project


def edit_is_stored(page: Page, project: str, kind: str) -> bool:
    """True when the edit made by open_with_edit is in the stored document."""
    data = get_project(page, project)["project_data"]
    if kind == "flat":
        return len(data["rules"]) == 2
    return next(n for n in data["nodes"] if n["id"] == "root")["position"] != node_ui.loan_document()["nodes"][0]["position"]


def press_save(page: Page) -> None:
    page.get_by_role("button", name="Save", exact=True).click()


def import_project(page: Page, filename: str, content: str | bytes, name: str | None = "Imported", submit: bool = True) -> Locator:
    """Projects page -> Import Project -> choose an in-memory file -> name -> submit (`submit`); returns the dialog."""
    page.get_by_role("button", name="Import Project").click()
    dialog = page.get_by_role("dialog")
    buffer = content.encode() if isinstance(content, str) else content
    dialog.locator("input[type=file]").set_input_files({"name": filename, "mimeType": "application/json", "buffer": buffer})
    if name is not None:
        dialog.get_by_placeholder("Enter project name").fill(name)
    if submit:
        dialog.locator("button[type=submit]").click()
    return dialog
