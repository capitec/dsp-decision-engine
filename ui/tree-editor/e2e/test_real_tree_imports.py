"""Real tree files imported the way a user does it (the Import Project dialog), exported the way a user does it
(the editor's Export), and the documents in the repo that are not plain tree documents."""

import json

import pytest
from playwright.sync_api import Page, expect

import real_trees_ui as ui
from canonical import as_lines
from real_trees import BY_ID, REPO, TREES
from round_trip import expected_items
from seed import export_document, get_project
from tree_runs import assert_runs_alike, probe_frame


def some(*tree_ids: str, smoke: str = "") -> list:
    """The trees as cases; `smoke` is the one in the smoke subset."""
    return [pytest.param(tree_id, id=tree_id, marks=[pytest.mark.smoke] if tree_id == smoke else [])
            for tree_id in tree_ids]


def import_through_the_dialog(page: Page, tree, tmp_path) -> tuple[str, bool]:
    path = tmp_path / f"{tree.id}.json"
    path.write_text(json.dumps(tree.document, indent=2))
    assert ui.import_file(page, path, tree.id).ok
    project_id, flat = ui.opened_project(page)
    ui.wait_until_loaded(page, flat)
    return project_id, flat


@pytest.mark.parametrize("tree_id", some("stored-rich-flat", "stored-rich-v3", "campaign-23", smoke="stored-rich-v3"))
def test_tree_file_imported_through_the_dialog_opens_in_its_editor(signed_in: Page, tmp_path, tree_id):
    tree = BY_ID[tree_id]
    project_id, flat = import_through_the_dialog(signed_in, tree, tmp_path)

    assert flat == tree.flat
    expect(ui.items(signed_in, flat)).to_have_count(expected_items(tree), timeout=15_000)
    project = get_project(signed_in, project_id)
    assert project["validation_errors"] == [] and project["project_data"] == tree.document


@pytest.mark.parametrize("tree_id", some("mcp-flat-rules", "stored-rich-v3", "campaign-23"))
def test_editor_export_downloads_what_the_api_exports(signed_in: Page, tmp_path, tree_id):
    project_id, flat = import_through_the_dialog(signed_in, BY_ID[tree_id], tmp_path)

    assert as_lines(ui.export(signed_in, flat, tmp_path)) == as_lines(export_document(signed_in, project_id))


@pytest.mark.parametrize("tree", [pytest.param(t, id=t.id) for t in TREES if t.file is not None])
def test_step_document_file_imports_its_tree(signed_in: Page, tree):
    reply = ui.import_file(signed_in, tree.file, tree.id)
    assert reply.ok, reply.text()
    project_id = reply.json()["project_id"]
    project = get_project(signed_in, project_id)
    assert project["validation_errors"] == [] and project["project_data"] == tree.document
    # The editor holds the tree; the step settings (path_output, feature_types, ...) stay in the file around it.
    assert_runs_alike(tree.wrap(export_document(signed_in, project_id)), tree.step, probe_frame(tree.step))


def test_fraud_rule_set_is_refused_with_a_clear_message(signed_in: Page):
    reply = ui.import_file(signed_in, REPO / "examples/transaction-fraud/configs/0.1.0/rule_set.json", "rules")
    assert reply.status == 400
    # The dialog also shows the file's name (rule_set.json), so look for the message itself.
    expect(signed_in.get_by_role("dialog", name="Import Project")).to_contain_text(
        "a 'rule_set' document is not a tree or flat-rule document")
