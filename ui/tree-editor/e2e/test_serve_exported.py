"""Exported trees served: a project whose `build(tree)` takes the export by name scores like the library does,
in-process for every tree and through a real `decikit build` + `decikit serve` for a representative few."""

import os

import pytest
from playwright.sync_api import Page

import serve_project as project
from real_trees import BY_ID
from round_trip import SLOW, cases, round_trip
from tree_runs import probe_frame, run

pytestmark = pytest.mark.serve

# flat rules with params, a v3 tree, the campaign tree, a 200-rule set; one port each, from E2E_SERVE_PORT on
PORT = int(os.environ.get("E2E_SERVE_PORT", 8650))
SERVED = [pytest.param(tree_id, PORT + i, id=tree_id, marks=[pytest.mark.slow] if tree_id in SLOW else [])
          for i, tree_id in enumerate(["stored-rich-flat", "stored-rich-v3", "campaign-23", "generated-200"])]


def exported_project(page: Page, tree, nulls: float = 0.1):
    exported = tree.wrap(round_trip(page, tree).export)
    frame = probe_frame(tree.step, 50, nulls)
    expected = run(exported, frame).drop("probe_row").to_dicts()
    records = project.records(frame)
    return project.make(f"served_{tree.id.replace('-', '_')}", exported, records), records, expected


@pytest.mark.parametrize("tree", cases({}))
def test_exported_tree_scored_in_process_matches_the_library(page: Page, tree):
    root, records, expected = exported_project(page, tree)
    handler = project.in_process(root)
    assert [project.run(handler, record) for record in records] == expected


@pytest.mark.parametrize("tree_id, port", SERVED)
def test_exported_tree_served_by_decikit_serve_matches_in_process(page: Page, tree_id, port):
    # Compiled, each new set of null inputs in a record compiles again (30 s a time for 200 rules), so only the
    # first record is null; the in-process test covers the nulls.
    root, records, _ = exported_project(page, BY_ID[tree_id], nulls=0)
    handler = project.in_process(root)
    project.build(root)
    with project.serving(root, port) as post:
        for record in records:
            assert post(record) == project.run(handler, record)
