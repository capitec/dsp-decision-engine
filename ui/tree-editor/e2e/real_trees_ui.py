"""What a user does with an existing tree document in either editor: open, save, export, import from a file."""

import json
import re
from pathlib import Path

from playwright.sync_api import Locator, Page, Response, expect

import flat_rules_ui


def items(page: Page, flat: bool) -> Locator:
    """The rule cards of the flat-rule editor, or the nodes on the tree editor's canvas."""
    return page.get_by_test_id("rule-card") if flat else page.locator(".react-flow__node")


def open_project(page: Page, project_id: str, flat: bool, timeout: float = 15_000) -> None:
    """Open the project in the editor its type belongs to and wait until it has loaded."""
    page.goto(f"/flat-rules/{project_id}" if flat else f"/projects/{project_id}")
    wait_until_loaded(page, flat, timeout)


def wait_until_loaded(page: Page, flat: bool, timeout: float = 15_000) -> None:
    expect(page.get_by_role("button", name="Add Rule" if flat else "Add Node")).to_be_visible(timeout=timeout)


def save(page: Page, flat: bool) -> dict:
    """Save as it is (the tree editor only saves a change, so move a node first); returns the server's reply."""
    if not flat:
        node = items(page, flat).first.bounding_box()
        page.mouse.move(node["x"] + node["width"] / 2, node["y"] + node["height"] / 2)
        page.mouse.down()
        page.mouse.move(node["x"] + node["width"] / 2 + 40, node["y"] + node["height"] / 2 + 40, steps=5)
        page.mouse.up()
        expect(page.get_by_text("Unsaved changes")).to_be_visible()
    with page.expect_response(lambda r: r.request.method == "PUT" and "/api/v1/projects/" in r.url) as reply:
        page.get_by_role("button", name="Save", exact=True).click()
    assert reply.value.ok, reply.value.text()
    return reply.value.json()


def export(page: Page, flat: bool, tmp_path: Path) -> dict:
    """Export through the editor (the Export button, or Export Project in the tree editor's menu)."""
    if flat:
        return flat_rules_ui.export(page, tmp_path)
    page.locator("button").filter(has=page.locator("svg.lucide-ellipsis-vertical")).last.click()
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
