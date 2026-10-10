"""Importing and exporting project files: bad files are refused in the dialog, good ones import once per attempt."""

import json
import re
import time

import pytest
from playwright.sync_api import Page, expect

import node_ui
from flat_rules_ui import export as export_from_editor
from platform_ui import import_project, open_projects
from seed import create_project, export_document, get_project, list_projects
from test_flat_rule_params import flat_document

BAD_FILES = {
    "txt_file": ("notes.txt", '{"type": "prioritized_flat_rule"}', "Only JSON files are supported"),
    "invalid_json": ("tree.json", '{"type": "tree", ', "Invalid JSON"),
    "empty_file": ("tree.json", "", "Invalid JSON"),
    "not_a_tree": ("tree.json", '{"a": 1}', "not a tree or flat-rule document"),
    "v1_tree": ("tree.json", '{"type": "v1-tree", "root": {}}', "v1 tree documents are no longer supported"),
    "json_array": ("tree.json", "[1, 2, 3]", "must be a JSON object"),
    "json_null": ("tree.json", "null", "must be a JSON object"),
    "json_string": ("tree.json", '"just text"', "must be a JSON object"),
    "binary_junk": ("tree.json", bytes([0xFF, 0xFE, 0x00, 0x80, 0x81]), "Failed to read file"),
}
LARGE_RULES = 521


def error_text(dialog) -> str:
    return dialog.locator("p.text-red-700").inner_text()


def canonical(document: dict) -> str:
    return json.dumps({k: v for k, v in document.items() if k != "metadata"}, sort_keys=True)


@pytest.mark.parametrize("filename,content,message", BAD_FILES.values(), ids=BAD_FILES.keys())
def test_a_file_that_is_not_a_project_is_refused_in_the_dialog_and_creates_nothing(signed_in: Page, filename, content, message):
    open_projects(signed_in)

    dialog = import_project(signed_in, filename, content, submit=not filename.endswith(".txt"))  # a .txt file is refused on choosing it

    expect(dialog.locator("p.text-red-700")).to_be_visible()
    assert message in error_text(dialog)
    expect(dialog).to_be_visible()
    assert list_projects(signed_in)["total"] == 0
    assert "/projects" in signed_in.url and "/flat-rules/" not in signed_in.url


def test_an_import_without_a_name_cannot_be_submitted(signed_in: Page):
    open_projects(signed_in)

    dialog = import_project(signed_in, "tree.json", json.dumps(flat_document()), name="")

    expect(dialog).to_be_visible()
    assert list_projects(signed_in)["total"] == 0


@pytest.mark.smoke
def test_importing_a_flat_rule_file_opens_it_in_the_flat_rule_editor(signed_in: Page):
    open_projects(signed_in)

    import_project(signed_in, "rules.json", json.dumps(flat_document()), name="Imported rules")

    expect(signed_in).to_have_url(re.compile(r"/flat-rules/"))
    expect(signed_in.get_by_role("heading", name="Imported rules")).to_be_visible()
    expect(signed_in.get_by_role("button", name="Add Rule")).to_be_visible()


def test_importing_a_tree_file_opens_it_in_the_node_editor(signed_in: Page):
    open_projects(signed_in)

    import_project(signed_in, "tree.json", json.dumps(node_ui.loan_document()), name="Imported tree")

    expect(signed_in).to_have_url(re.compile(r"/projects/[0-9a-f-]{36}$"))
    expect(signed_in.locator(".react-flow__node")).to_have_count(3, timeout=20_000)


def test_importing_the_same_file_twice_makes_two_independent_projects(signed_in: Page):
    document = flat_document()
    content = json.dumps(document)
    open_projects(signed_in)
    import_project(signed_in, "rules.json", content, name="First copy")
    expect(signed_in).to_have_url(re.compile(r"/flat-rules/"))
    signed_in.goto("/projects")
    import_project(signed_in, "rules.json", content, name="Second copy")
    expect(signed_in).to_have_url(re.compile(r"/flat-rules/"))

    items = {p["name"]: p["id"] for p in list_projects(signed_in)["items"]}
    assert set(items) == {"First copy", "Second copy"}
    assert items["First copy"] != items["Second copy"]
    renamed = signed_in.request.put(f"/api/v1/projects/{items['First copy']}", data={"name": "Changed"})
    assert renamed.ok
    other = get_project(signed_in, items["Second copy"])
    assert other["name"] == "Second copy"
    assert signed_in.request.delete(f"/api/v1/projects/{items['First copy']}").ok
    assert get_project(signed_in, items["Second copy"])["project_data"] == other["project_data"]


def test_an_export_imports_back_into_an_identical_project(signed_in: Page):
    project = create_project(signed_in, "Round trip", flat_document())["id"]
    exported = export_document(signed_in, project)
    open_projects(signed_in)

    import_project(signed_in, "round.json", json.dumps(exported), name="Round trip copy")
    expect(signed_in).to_have_url(re.compile(r"/flat-rules/"))

    copy = next(p["id"] for p in list_projects(signed_in)["items"] if p["name"] == "Round trip copy")
    assert canonical(export_document(signed_in, copy)) == canonical(exported)


# ---- a very large file --------------------------------------------------------------------------------------------------------

def large_document() -> dict:
    rule = {"type": "unary", "then": {"type": "leaf", "result_idx": 0},
            "condition": {"type": "unary", "op": "<=", "feature": "amount", "threshold": 1000}}
    document = flat_document(*[rule] * LARGE_RULES)
    for i, entry in enumerate(document["rules"]):
        entry["meta"]["name"] = f"Rule {i + 1}"
    return document


def without_editor_additions(value):
    """The document minus what the editor adds on opening it: node ids and rule positions."""
    if isinstance(value, dict):
        return {k: without_editor_additions(v) for k, v in value.items() if k not in ("id", "position")}
    if isinstance(value, list):
        return [without_editor_additions(v) for v in value]
    return value


@pytest.mark.slow
def test_a_521_rule_document_imports_and_exports_unchanged_within_a_minute(signed_in: Page):
    document = large_document()
    started = time.monotonic()
    open_projects(signed_in)

    import_project(signed_in, "large.json", json.dumps(document), name="Large")
    expect(signed_in).to_have_url(re.compile(r"/flat-rules/"), timeout=60_000)
    project = signed_in.url.rsplit("/", 1)[1]

    same = canonical(export_document(signed_in, project)) == canonical(document)
    assert same, "the stored export differs from the imported document"
    assert time.monotonic() - started < 60
    assert len(get_project(signed_in, project)["project_data"]["rules"]) == LARGE_RULES


@pytest.mark.slow
def test_a_521_rule_document_exported_from_the_editor_keeps_every_rule(signed_in: Page, tmp_path):
    document = large_document()
    project = create_project(signed_in, "Large", document)["id"]
    started = time.monotonic()
    signed_in.goto(f"/flat-rules/{project}")
    expect(signed_in.get_by_role("button", name="Add Rule")).to_be_visible(timeout=60_000)

    exported = export_from_editor(signed_in, tmp_path)

    assert time.monotonic() - started < 90
    same = canonical(without_editor_additions(exported)) == canonical(without_editor_additions(document))
    assert same, "the editor's export lost or changed rule content"


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="opening a project in the flat-rule editor and exporting it adds a fresh random id to every node and a position to every rule, so an unchanged document does not export unchanged")
def test_a_521_rule_document_exported_from_the_editor_is_canonically_equal(signed_in: Page, tmp_path):
    document = large_document()
    project = create_project(signed_in, "Large", document)["id"]
    signed_in.goto(f"/flat-rules/{project}")
    expect(signed_in.get_by_role("button", name="Add Rule")).to_be_visible(timeout=60_000)

    exported = export_from_editor(signed_in, tmp_path)

    same = canonical(exported) == canonical(document)
    assert same, "the editor's export differs from the stored document"
