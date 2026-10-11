"""One tree's trip through the editor (import, open, save, reload, export), done once per test session."""

import uuid
from dataclasses import dataclass, field

import pytest
from playwright.sync_api import Page, expect

import real_trees_ui as ui
from real_trees import TREES, RealTree
from seed import export_document, import_document
from tree_ui import register


@dataclass
class RoundTrip:
    imported: dict  # the import reply, with `validation_errors`
    shown: int  # rule cards or nodes on screen once opened
    saved: dict  # the save reply
    saved_message: str  # the toast after Save
    shown_after_reload: int
    export: dict  # GET /export after the reload
    page_errors: list = field(default_factory=list)


SLOW = {"fraud-shadow", "generated-200"}
# Not run: opening them takes the browser past the 3 GB the e2e runs are capped at, which kills the whole run.
TOO_BIG = {tree_id: "the flat-rule editor renders all 521 rules at once and runs out of memory (3 GB) opening them"
           for tree_id in ("fraud-live", "generated-521")}


def cases(known: dict[str, str], trees: list[RealTree] = TREES) -> list:
    """Every tree (or `trees`) as a test case, expected to fail where `known` says why."""
    return [pytest.param(tree, id=tree.id, marks=[
        *([pytest.mark.xfail(run=False, reason=TOO_BIG[tree.id])] if tree.id in TOO_BIG else []),
        *([pytest.mark.xfail(strict=True, reason=known[tree.id])] if tree.id in known else []),
        *([pytest.mark.slow] if tree.id in SLOW else []),
    ]) for tree in trees]


# A tree that fails its trip fails every check of it with the same error, without trying again.
_done: dict[str, RoundTrip | AssertionError] = {}


def expected_items(tree: RealTree) -> int:
    if not tree.flat:
        return len(tree.document["nodes"])
    return len(tree.document.get("rules") or [tree.document["rule"]])  # a `flat_rule` holds one


def round_trip(page: Page, tree: RealTree) -> RoundTrip:
    """A fresh user imports the tree's document as the import dialog does, opens it, saves, reloads and exports."""
    if tree.id not in _done:
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            _done[tree.id] = _trip(page, tree, errors)
        except Exception as error:
            _done[tree.id] = AssertionError(f"{tree.id} did not get through the editor: {error}\npage errors: {errors}")
    if isinstance(_done[tree.id], AssertionError):
        raise _done[tree.id]
    return _done[tree.id]


def _trip(page: Page, tree: RealTree, errors: list) -> RoundTrip:
    page.on("dialog", lambda dialog: dialog.accept())
    register(page, f"{uuid.uuid4().hex[:10]}@example.com")
    imported = import_document(page, tree.id, tree.document, f"{tree.id}.json")
    # The flat-rule editor renders every rule at once, so a big rule set takes a while to open; a loaded
    # machine takes several seconds over a small one.
    timeout = 15_000 + 250 * expected_items(tree)
    page.set_default_timeout(timeout)
    ui.open_project(page, imported["project_id"], tree.flat, timeout)
    expect(ui.items(page, tree.flat).first).to_be_visible(timeout=timeout)
    shown = ui.items(page, tree.flat).count()
    saved = ui.save(page, tree.flat)
    message = page.get_by_text("Project Saved").or_(page.get_by_text("Saved with validation issues")).first
    message = message.inner_text(timeout=timeout)
    page.reload()
    ui.wait_until_loaded(page, tree.flat, timeout)
    expect(ui.items(page, tree.flat).first).to_be_visible(timeout=timeout)
    return RoundTrip(imported, shown, saved, message, ui.items(page, tree.flat).count(),
                     export_document(page, imported["project_id"]), errors)
