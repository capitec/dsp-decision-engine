"""Every tree's export runs in decikit exactly as its source does: same params, same columns, same values."""

import pytest
from decikit import TreeConfig
from playwright.sync_api import Page

from canonical import as_lines
from real_trees import BY_ID, RealTree
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


@pytest.mark.xfail(strict=True, reason="the flat-rule editor names an unnamed rule 'Rule N' on open, which renames its "
                                       "output columns in `all` mode (rule_0.hit -> Rule 1.hit)")
def test_unnamed_rules_in_all_mode_keep_their_output_columns(page: Page):
    def rule(feature: str) -> dict:
        return {"rule": {"type": "unary", "condition": {"op": ">", "feature": feature, "threshold": 1},
                         "then": {"type": "leaf", "result_idx": 0}}}

    tree = RealTree("all-mode-unnamed", "test_real_trees_run.py", {"type": "tree", "name": "t", "tree": {
        "type": "prioritized_flat_rule", "mode": "all", "rules": [rule("a"), rule("b")],
        "output": {"data": [{"hit": True}], "default": {"hit": False}, "dtypes": [["hit", "Boolean"]]}}})
    assert_runs_alike(tree.wrap(round_trip(page, tree).export), tree.step, probe_frame(tree.step))
