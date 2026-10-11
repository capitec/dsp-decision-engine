"""Every tree document in the repo (real_trees.py) goes through the editor and comes back as it went in."""

import pytest
from decikit import TreeConfig
from decikit.steps.trees._encode import feature_kinds, kind_of
from playwright.sync_api import Page

from canonical import as_lines, own_parts
from real_trees import BY_ID, TREES, tree_files
from round_trip import cases, expected_items, round_trip

NAMED = [tree for tree in TREES if (tree.document.get("metadata") or {}).get("name")]


@pytest.mark.parametrize("tree", cases({}))
def test_real_tree_opens_and_saves_without_errors(page: Page, tree):
    trip = round_trip(page, tree)
    assert trip.imported["validation_errors"] == []
    assert trip.shown == expected_items(tree)
    assert trip.saved["validation_errors"] == [] and trip.saved_message == "Project Saved"
    assert trip.shown_after_reload == trip.shown
    assert trip.page_errors == []


@pytest.mark.parametrize("tree", cases({}))
def test_real_tree_export_matches_its_source(page: Page, tree):
    export = round_trip(page, tree).export
    assert as_lines(export) == as_lines(tree.document)
    assert own_parts(export, tree.document) == own_parts(tree.document, tree.document)


@pytest.mark.parametrize("tree", cases({}, NAMED))
def test_real_tree_export_keeps_the_documents_own_name(page: Page, tree):
    assert round_trip(page, tree).export["metadata"]["name"] == tree.document["metadata"]["name"]


@pytest.mark.parametrize("tree", cases({}))
def test_real_tree_export_types_the_features_it_adds_as_decikit_reads_them(page: Page, tree):
    known = tree.document.get("input_schema") or []
    added = [entry for entry in round_trip(page, tree).export.get("input_schema") or [] if entry not in known]
    kinds = feature_kinds(TreeConfig.load(tree.step)._tree(), tree.step.get("feature_types", {}))
    assert [(name, kind_of(dtype, name)) for name, dtype in added] == [(name, kinds.get(name)) for name, _ in added]


@pytest.mark.smoke
def test_flat_rules_with_parameters_and_leaf_ids_come_back_unchanged(page: Page):
    tree = BY_ID["docs-guide-fraud-rules"]
    assert as_lines(round_trip(page, tree).export) == as_lines(tree.document)


def test_every_tree_json_file_in_the_repo_is_in_the_list():
    assert tree_files() == {tree.file for tree in TREES if tree.file}
