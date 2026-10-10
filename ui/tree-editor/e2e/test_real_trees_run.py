"""Every tree's export runs in decikit exactly as its source does: same params, same columns, same values."""

import pytest
from decikit import TreeConfig
from playwright.sync_api import Page

from round_trip import cases, round_trip
from tree_runs import assert_runs_alike, probe_frame


@pytest.mark.parametrize("tree", cases({}))
def test_real_tree_export_runs_like_its_source(page: Page, tree):
    exported = tree.wrap(round_trip(page, tree).export)
    assert TreeConfig.load(exported).parameters() == TreeConfig.load(tree.step).parameters()
    assert_runs_alike(exported, tree.step, probe_frame(tree.step))
