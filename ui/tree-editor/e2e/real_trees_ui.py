"""What a user does with an existing tree document in either editor: open, save, export, import from a file."""

import json
import re
from pathlib import Path

from playwright.sync_api import Locator, Page, Response, expect

import flat_rules_ui
import node_ui
import tree_ui


def items(page: Page, flat: bool) -> Locator:
    """The rule cards of the flat-rule editor, or the nodes on the tree editor's canvas."""
    return page.get_by_test_id("rule-card") if flat else page.locator(".react-flow__node")


def open_project(page: Page, project_id: str, flat: bool, timeout: float = 15_000) -> None:
    """Open the project in the editor its type belongs to and wait until it has loaded."""
    page.goto(f"/flat-rules/{project_id}" if flat else f"/projects/{project_id}")
    wait_until_loaded(page, flat, timeout)


def wait_until_loaded(page: Page, flat: bool, timeout: float = 15_000) -> None:
    """Wait for the editor's Add Rule or Add Node button."""
    expect(page.get_by_role("button", name="Add Rule" if flat else "Add Node")).to_be_visible(timeout=timeout)
    if flat:
        flat_rules_ui.wait_until_connected(page)


def save(page: Page, flat: bool) -> dict:
    """Save as it is; returns the server's reply. The tree editor, and a live flat-rule editor, send their form of the
    document only with an edit, so this first moves a node, or renames a rule and back."""
    if not flat:
        tree_ui.drag(page, items(page, flat).first, 40, 40)
        expect(page.get_by_text("Unsaved changes")).to_be_visible()
    elif flat_rules_ui.LIVE:
        name = flat_rules_ui.rule_names(page)[0]
        flat_rules_ui.rename_rule(page, f"{name} renamed")
        flat_rules_ui.rename_rule(page, name)
    return flat_rules_ui.save(page)


def export(page: Page, flat: bool, tmp_path: Path) -> dict:
    """Export through the editor (the Export button, or Export Project in the tree editor's menu)."""
    if flat:
        return flat_rules_ui.export(page, tmp_path)
    node_ui.open_menu(page)
    with page.expect_download() as download:
        page.get_by_role("menuitem", name="Export Project").click()
    path = tmp_path / "export.json"
    download.value.save_as(path)
    return json.loads(path.read_text())


def import_file(page: Page, path: Path, name: str) -> Response:
    """Projects -> Import Project -> Browse Files -> name -> Import Project; returns the server's reply."""
    page.get_by_role("button", name="Projects").click()
    page.get_by_role("button", name="Import Project").click()
    dialog = page.get_by_role("dialog", name="Import Project")
    with page.expect_file_chooser() as chooser:
        dialog.get_by_role("button", name="Browse Files").click()
    chooser.value.set_files(path)
    dialog.get_by_placeholder("Enter project name").fill(name)
    with page.expect_response("**/api/v1/projects/import") as reply:
        dialog.get_by_role("button", name="Import Project").click()
    return reply.value


def opened_project(page: Page) -> tuple[str, bool]:
    """The id of the project the browser shows, and whether it is in the flat-rule editor."""
    page.wait_for_url(re.compile(r"/(flat-rules|projects)/[0-9a-f-]{36}$"))
    kind, project_id = page.url.rsplit("/", 2)[-2:]
    return project_id, kind == "flat-rules"
