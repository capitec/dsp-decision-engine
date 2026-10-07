import json
import re

import polars as pl
from decider import TreeConfig
from playwright.sync_api import Page, expect

from tree_ui import build_loan_tree, create_tree_project, save


def open_menu(page: Page) -> None:
    page.locator("button").filter(has=page.locator("svg.lucide-ellipsis-vertical")).last.click()


def export_project(page: Page, tmp_path) -> dict:
    open_menu(page)
    with page.expect_download() as download:
        page.get_by_role("menuitem", name="Export Project").click()
    path = tmp_path / "export.json"
    download.value.save_as(path)
    return json.loads(path.read_text())


def test_new_tree_project_is_listed(signed_in: Page):
    create_tree_project(signed_in, "Listed tree")
    signed_in.get_by_role("button", name="Back").click()
    expect(signed_in.get_by_text("Listed tree", exact=True)).to_be_visible()


def test_saved_tree_survives_reload(signed_in: Page):
    create_tree_project(signed_in, "Persisted tree")
    build_loan_tree(signed_in)
    save(signed_in)
    signed_in.reload()
    expect(signed_in.locator(".react-flow__node")).to_have_count(3)
    expect(signed_in.locator(".react-flow__edge")).to_have_count(2)
    expect(signed_in.get_by_text("<= 1000")).to_be_visible()


def test_exported_tree_runs_in_decider(signed_in: Page, tmp_path):
    create_tree_project(signed_in, "Exported tree")
    build_loan_tree(signed_in, limit=1000)
    save(signed_in)

    document = export_project(signed_in, tmp_path)

    tree = TreeConfig.load({"type": "tree", "name": "loan", "tree": document})
    out = tree.run(pl.DataFrame({"amount": [500.0, 1000.0, 1500.0]}))
    assert out["decision"].to_list() == ["approve", "approve", "decline"]


def test_exported_tree_imports_back(signed_in: Page, tmp_path):
    create_tree_project(signed_in, "Round trip")
    build_loan_tree(signed_in)
    save(signed_in)
    exported = tmp_path / "round_trip.json"
    exported.write_text(json.dumps(export_project(signed_in, tmp_path)))

    signed_in.get_by_role("button", name="Back").click()
    signed_in.get_by_role("button", name="Import Project").click()
    signed_in.locator("input[type=file]").set_input_files(exported)
    signed_in.get_by_placeholder("Enter project name").fill("Imported copy")
    signed_in.get_by_role("dialog").locator("button[type=submit]").click()

    signed_in.goto("/projects")
    card = signed_in.get_by_text("Imported copy", exact=True).locator("xpath=ancestor::div[.//button[normalize-space()='Open Project']][1]")
    card.get_by_role("button", name="Open Project").click()
    expect(signed_in.locator(".react-flow__node")).to_have_count(3)


def test_deleted_project_leaves_the_list(signed_in: Page):
    create_tree_project(signed_in, "Doomed tree")
    open_menu(signed_in)
    signed_in.get_by_role("menuitem", name="Delete Project").click()
    signed_in.get_by_role("alertdialog").get_by_role("button", name="Delete").click()

    expect(signed_in).to_have_url(re.compile(r"/projects$"))
    expect(signed_in.get_by_text("Doomed tree", exact=True)).to_be_hidden()


def test_new_project_starts_without_unsaved_changes(signed_in: Page):
    create_tree_project(signed_in, "Clean start")
    signed_in.wait_for_load_state("networkidle")
    expect(signed_in.get_by_text("Unsaved changes")).to_be_hidden()
    expect(signed_in.get_by_role("button", name="Save")).to_be_disabled()


def test_projects_page_loads_folders_right_after_sign_up(signed_in: Page):
    signed_in.get_by_role("button", name="Projects").click()
    signed_in.wait_for_load_state("networkidle")
    expect(signed_in.get_by_text("Failed to load folders")).to_be_hidden()
