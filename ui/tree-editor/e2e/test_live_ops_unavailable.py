"""A browser that cannot load the live-editing wasm still opens both editors and saves over REST, and says so.

Runs with live editing on or off: either way that browser saves with Save.
"""

from playwright.sync_api import Page, expect

import node_ui
from flat_docs import SMALL, document
from flat_rules_ui import open_flat_rules, rename_rule, rule_names
from seed import create_project, get_project

UNAVAILABLE = "Live editing is unavailable in this browser"


def block_wasm(page: Page) -> None:
    page.route("**/*.wasm", lambda route: route.abort())


def test_without_the_wasm_the_flat_rule_editor_says_so_and_saves_over_rest(signed_in: Page):
    project = create_project(signed_in, "No wasm", document(SMALL, SMALL))["id"]
    block_wasm(signed_in)

    open_flat_rules(signed_in, project)

    expect(signed_in.get_by_test_id("live-ops-failed")).to_contain_text(UNAVAILABLE)
    expect(signed_in.get_by_test_id("live-status")).to_be_hidden()
    rename_rule(signed_in, "Saved without wasm")
    reply = node_ui.save(signed_in)
    assert reply["version"] == 2
    assert [r["meta"]["name"] for r in get_project(signed_in, project)["project_data"]["rules"]][0] == "Saved without wasm"
    signed_in.reload()
    open_flat_rules(signed_in, project)
    assert rule_names(signed_in)[0] == "Saved without wasm"


def test_without_the_wasm_the_node_editor_saves_over_rest(signed_in: Page):
    block_wasm(signed_in)
    project = node_ui.open_tree(signed_in, node_ui.loan_document())

    expect(signed_in.get_by_test_id("live-ops-failed")).to_contain_text(UNAVAILABLE)
    node_ui.select_node(signed_in, "root")
    node_ui.set_threshold(signed_in, "250")
    node_ui.save(signed_in)

    assert node_ui.saved_data(signed_in, project)["condition"]["threshold"] == 250
