"""Flat-rule editor actions: each does what a user does, nothing more."""

import json
import re

from playwright.sync_api import Locator, Page, expect


def open_flat_rules(page: Page, project_id: str) -> None:
    page.goto(f"/flat-rules/{project_id}")
    expect(page.get_by_role("button", name="Add Rule")).to_be_visible()


def rule_card(page: Page, rule: int = 0) -> Locator:
    return page.get_by_test_id("rule-card").nth(rule)


def add_parameter(page: Page, name: str, dtype: str = "Float64", value: str | None = None) -> None:
    """Manage Parameters -> new row -> name, type, default -> Add -> Save Changes."""
    page.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    dialog = page.get_by_role("dialog", name="Manage Parameters")
    first = dialog.get_by_role("button", name="Add Your First Parameter")
    (first if first.is_visible() else dialog.get_by_role("button", name="Add Parameter")).click()
    row = dialog.locator("tbody tr").filter(has=page.get_by_placeholder("parameter_name"))
    row.get_by_placeholder("parameter_name").fill(name)
    if dtype != "String":
        row.get_by_role("combobox").click()
        page.get_by_role("option", name=dtype, exact=True).click()
    if value is not None:
        row.locator("td").nth(2).locator("input").first.fill(value)
    row.get_by_role("button", name="Add", exact=True).click()
    dialog.get_by_role("button", name="Save Changes").click()
    expect(dialog).to_be_hidden()


def set_parameter_default(page: Page, name: str, value: str) -> None:
    """Manage Parameters -> the row of `name` -> new default -> Save Changes."""
    page.get_by_role("button", name=re.compile(r"^Manage Parameters")).click()
    dialog = page.get_by_role("dialog", name="Manage Parameters")
    row = dialog.locator("tbody tr").filter(has=page.locator(f"input[value='{name}']"))
    row.locator("td").nth(2).locator("input").first.fill(value)
    dialog.get_by_role("button", name="Save Changes").click()
    expect(dialog).to_be_hidden()


def set_node_type(page: Page, scope: Locator, kind: str, index: int = 0) -> None:
    """Leaf, Unary, Cases or Composite for the `index`-th node of `scope`."""
    scope.get_by_test_id("node-type").nth(index).click()
    page.get_by_role("option", name=kind, exact=True).click()


def choose_feature(page: Page, scope: Locator, feature: str) -> None:
    scope.get_by_text("Select or search feature...").first.click()
    page.get_by_role("option", name=feature, exact=True).click()


def choose_output(page: Page, scope: Locator, label: str, index: int = 0) -> None:
    """Pick the output record whose label contains `label` in the `index`-th leaf of `scope`."""
    scope.get_by_test_id("leaf-output").nth(index).get_by_role("combobox").click()
    page.get_by_role("option").filter(has_text=label).first.click()


def use_parameter(scope: Locator, name: str, index: int = 0) -> None:
    """Click the `index`-th # (next to a value), then type the parameter's name."""
    toggle = scope.get_by_role("button", name="#").nth(index)
    holder = toggle.locator("xpath=..")
    toggle.click()
    box = holder.get_by_placeholder("parameter_name")
    box.fill(name)
    box.press("Escape")


def save(page: Page) -> dict:
    """Click Save; returns the server's reply (with `validation_errors`)."""
    with page.expect_response(lambda r: r.request.method == "PUT" and "/api/v1/projects/" in r.url) as reply:
        page.get_by_role("button", name="Save", exact=True).click()
    assert reply.value.ok, reply.value.text()
    return reply.value.json()


def export(page: Page, tmp_path) -> dict:
    """Click Export and return the downloaded document; fails if the export is refused."""
    with page.expect_download() as download:
        page.get_by_role("button", name="Export").click()
    path = tmp_path / "export.json"
    download.value.save_as(path)
    return json.loads(path.read_text())
