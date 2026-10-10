"""Every tree's export runs in decikit exactly as its source does: same params, same columns, same values."""

import pytest
from decikit import TreeConfig
from playwright.sync_api import Page

from canonical import as_lines
from real_trees import BY_ID
from round_trip import TOO_BIG, cases, round_trip
from seed import export_document, import_document
from tree_runs import assert_runs_alike, probe_frame


@pytest.mark.parametrize("tree", cases({}))
def test_real_tree_export_runs_like_its_source(page: Page, tree):
    exported = tree.wrap(round_trip(page, tree).export)
    assert TreeConfig.load(exported).parameters() == TreeConfig.load(tree.step).parameters()
    assert_runs_alike(exported, tree.step, probe_frame(tree.step))


@pytest.mark.parametrize("tree_id", sorted(TOO_BIG))
def test_rule_set_too_big_to_open_exports_and_runs_like_its_source(signed_in: Page, tree_id):
    # The editor cannot open these, but importing and exporting them must still keep every rule and how it runs.
    tree = BY_ID[tree_id]
    imported = import_document(signed_in, tree.id, tree.document, f"{tree.id}.json")
    assert imported["validation_errors"] == []
    export = export_document(signed_in, imported["project_id"])
    assert as_lines(export) == as_lines(tree.document)
    assert_runs_alike(tree.wrap(export), tree.step, probe_frame(tree.step))
