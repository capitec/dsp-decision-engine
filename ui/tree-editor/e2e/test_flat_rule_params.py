"""Parameters in flat rules (and the node editor's ranges): they save valid, export, and run with their defaults."""

import polars as pl
import pytest
from decikit import TreeConfig
from playwright.sync_api import Page, expect

from flat_rules_ui import (
    add_parameter,
    choose_feature,
    choose_output,
    export,
    open_flat_rules,
    rule_card,
    save,
    set_node_type,
    set_parameter_default,
    use_parameter,
)
from seed import create_project

THEN_APPROVE = {"type": "leaf", "result_idx": 0}
OTHERWISE_DEFAULT = {"type": "leaf", "result_idx": -1}
LIMIT = {"limit": {"type": "Float64", "default_value": 1000}}


def flat_document(*rules: dict, parameters: dict | None = None, mode: str = "first_match") -> dict:
    return {
        "type": "prioritized_flat_rule",
        "mode": mode,
        "parameters": parameters or {},
        "input_schema": [["amount", "Float64"], ["band", "String"]],
        "output": {
            "dtypes": [["decision", "String"]],
            "type_defs": {},
            "data": [{"decision": "approve"}],
            "default": {"decision": "decline"},
        },
        "rules": [{"meta": {"name": f"Rule {i + 1}"}, "rule": r} for i, r in enumerate(rules or [OTHERWISE_DEFAULT])],
    }


def run(document: dict, amounts: list[float], params: dict | None = None) -> list:
    tree = TreeConfig.load({"type": "tree", "name": "rules", "tree": document})
    out = tree.run(pl.DataFrame({"amount": amounts}), **({"params": {"rules": params}} if params else {}))
    return out.drop("amount").to_dicts()


def decisions(document: dict, amounts: list[float], params: dict | None = None) -> list[str]:
    return [row["decision"] for row in run(document, amounts, params)]


def open_seeded(page: Page, *rules: dict, parameters: dict | None = None, mode: str = "first_match") -> str:
    project = create_project(page, "Params", flat_document(*rules, parameters=parameters, mode=mode))["id"]
    open_flat_rules(page, project)
    return project


def unary_then_approve(page: Page) -> None:
    """Rule 1's root becomes `amount <= <value>` -> approve."""
    card = rule_card(page)
    set_node_type(page, card, "Unary")
    choose_feature(page, card, "amount")
    card.get_by_role("button", name="Add THEN").click()
    choose_output(page, card, "approve")


@pytest.mark.smoke
def test_threshold_from_a_new_parameter_saves_valid_exports_and_runs(signed_in: Page, tmp_path):
    open_seeded(signed_in)
    add_parameter(signed_in, "limit", "Float64", "1000")
    unary_then_approve(signed_in)
    use_parameter(rule_card(signed_in), "limit")

    assert save(signed_in)["validation_errors"] == []
    expect(signed_in.get_by_text("Project Saved", exact=True)).to_be_visible()
    document = export(signed_in, tmp_path)

    assert document["rules"][0]["rule"]["condition"]["threshold"] == {"param": "limit"}
    assert document["parameters"]["limit"]["default_value"] == 1000
    assert decisions(document, [500.0, 1500.0]) == ["approve", "decline"]
    assert decisions(document, [500.0, 1500.0], {"limit": 2000.0}) == ["approve", "approve"]


@pytest.mark.parametrize("dtype, value, expected", [
    ("Int64", "1000", 1000),
    ("Float64", "999.5", 999.5),
    ("String", "1000", "1000"),
])
def test_parameter_default_of_each_type_exports_and_runs(signed_in: Page, tmp_path, dtype, value, expected):
    open_seeded(signed_in)
    add_parameter(signed_in, "limit", dtype, value)
    unary_then_approve(signed_in)
    use_parameter(rule_card(signed_in), "limit")

    assert save(signed_in)["validation_errors"] == []
    document = export(signed_in, tmp_path)
    assert document["parameters"]["limit"]["default_value"] == expected
    assert decisions(document, [500.0, 1500.0]) == ["approve", "decline"]


def test_parameter_picked_from_the_suggestions_saves_valid(signed_in: Page, tmp_path):
    open_seeded(signed_in)
    add_parameter(signed_in, "limit", "Float64", "1000")
    unary_then_approve(signed_in)
    card = rule_card(signed_in)
    card.get_by_role("button", name="#").first.click()
    signed_in.get_by_text("Float64: 1000").click()

    assert save(signed_in)["validation_errors"] == []
    assert export(signed_in, tmp_path)["rules"][0]["rule"]["condition"]["threshold"] == {"param": "limit"}


def test_parameter_rule_reopened_and_saved_again_stays_valid(signed_in: Page, tmp_path):
    open_seeded(signed_in)
    add_parameter(signed_in, "limit", "Float64", "1000")
    unary_then_approve(signed_in)
    use_parameter(rule_card(signed_in), "limit")
    save(signed_in)

    signed_in.reload()
    expect(rule_card(signed_in).get_by_placeholder("parameter_name")).to_have_value("limit")
    assert save(signed_in)["validation_errors"] == []
    assert decisions(export(signed_in, tmp_path), [500.0, 1500.0]) == ["approve", "decline"]


def test_parameter_inside_a_composite_saves_valid_and_runs(signed_in: Page, tmp_path):
    composite = {"type": "composite", "op": "and", "then": THEN_APPROVE, "conditions": [
        {"type": "unary", "op": ">", "feature": "amount", "threshold": 0},
        {"type": "unary", "op": "<=", "feature": "amount", "threshold": 5}]}
    open_seeded(signed_in, composite, parameters=LIMIT)
    use_parameter(rule_card(signed_in), "limit", index=1)

    assert save(signed_in)["validation_errors"] == []
    document = export(signed_in, tmp_path)
    assert document["rules"][0]["rule"]["conditions"][1]["threshold"] == {"param": "limit"}
    assert decisions(document, [-1.0, 500.0, 1500.0]) == ["decline", "approve", "decline"]


def test_parameter_as_a_between_bound_saves_valid_and_runs(signed_in: Page, tmp_path):
    between = {"type": "unary", "then": THEN_APPROVE,
               "condition": {"type": "unary", "op": "between", "feature": "amount", "min": 0, "max": 5}}
    open_seeded(signed_in, between, parameters=LIMIT)
    use_parameter(rule_card(signed_in), "limit", index=1)

    assert save(signed_in)["validation_errors"] == []
    document = export(signed_in, tmp_path)
    assert document["rules"][0]["rule"]["condition"]["max"] == {"param": "limit"}
    assert decisions(document, [-1.0, 1000.0, 1000.5]) == ["decline", "approve", "decline"]


def test_parameter_as_isin_values_saves_valid_and_runs(signed_in: Page, tmp_path):
    isin = {"type": "unary", "then": THEN_APPROVE,
            "condition": {"type": "unary", "op": "isin", "feature": "amount", "values": [1, 2]}}
    open_seeded(signed_in, isin, parameters=LIMIT)
    use_parameter(rule_card(signed_in), "limit")

    assert save(signed_in)["validation_errors"] == []
    document = export(signed_in, tmp_path)
    assert document["rules"][0]["rule"]["condition"]["values"] == {"param": "limit"}
    assert decisions(document, [1.0, 1000.0]) == ["decline", "approve"]


def test_parameter_as_a_manual_range_bound_saves_valid_and_runs(signed_in: Page, tmp_path):
    ranges = {"type": "cases", "op": "ranges", "strict": False, "feature": "amount", "otherwise": 1,
              "conditions": [{"when": {"min": None, "max": 5}, "then": 0}, {"when": {"min": 5, "max": None}, "then": 1}],
              "branches": [THEN_APPROVE, OTHERWISE_DEFAULT]}
    open_seeded(signed_in, ranges, parameters=LIMIT)
    use_parameter(rule_card(signed_in), "limit", index=1)

    assert save(signed_in)["validation_errors"] == []
    document = export(signed_in, tmp_path)
    assert document["rules"][0]["rule"]["conditions"][0]["when"]["max"] == {"param": "limit"}
    assert decisions(document, [500.0, 1500.0]) == ["approve", "decline"]


def test_one_parameter_used_by_two_rules_in_all_mode_runs_both(signed_in: Page, tmp_path):
    below = {"type": "unary", "then": THEN_APPROVE,
             "condition": {"type": "unary", "op": "<=", "feature": "amount", "threshold": 5}}
    above = {"type": "unary", "then": THEN_APPROVE,
             "condition": {"type": "unary", "op": ">", "feature": "amount", "threshold": 5}}
    open_seeded(signed_in, below, above, parameters=LIMIT, mode="all")
    use_parameter(rule_card(signed_in, 0), "limit")
    use_parameter(rule_card(signed_in, 1), "limit")

    assert save(signed_in)["validation_errors"] == []
    rows = run(export(signed_in, tmp_path), [500.0, 1500.0])
    assert rows == [{"Rule 1.decision": "approve", "Rule 2.decision": "decline"},
                    {"Rule 1.decision": "decline", "Rule 2.decision": "approve"}]


def test_parameter_default_a_threshold_cannot_use_is_reported_and_blocks_the_export(signed_in: Page, tmp_path):
    rule = {"type": "unary", "then": THEN_APPROVE,
            "condition": {"type": "unary", "op": "<=", "feature": "amount", "threshold": {"param": "limit"}}}
    open_seeded(signed_in, rule, parameters={"limit": {"type": "String", "default_value": ""}})

    assert [e["loc"] for e in save(signed_in)["validation_errors"]] == [["parameters", "limit"]]
    expect(signed_in.get_by_test_id("parameter-errors")).to_contain_text("parameter 'limit'")
    signed_in.get_by_role("button", name="Export").click()
    expect(signed_in.get_by_text("Export failed", exact=True)).to_be_visible()

    set_parameter_default(signed_in, "limit", "1000")
    assert save(signed_in)["validation_errors"] == []
    expect(signed_in.get_by_test_id("parameter-errors")).to_be_hidden()
    assert decisions(export(signed_in, tmp_path), [500.0, 1500.0]) == ["approve", "decline"]


STRICT_RANGES = {"type": "cases", "op": "ranges", "strict": True, "feature": "amount", "otherwise": 2,
                 "conditions": [{"when": {"min": None, "max": {"param": "limit"}}, "then": 0},
                                {"when": {"min": None, "max": 5000}, "then": 1}],
                 "branches": [THEN_APPROVE, OTHERWISE_DEFAULT, OTHERWISE_DEFAULT]}


def test_strict_ranges_with_a_parameter_bound_stay_editable(signed_in: Page, tmp_path):
    crashes = []
    signed_in.on("pageerror", lambda e: crashes.append(str(e)))
    project = create_project(signed_in, "Strict", flat_document(STRICT_RANGES, parameters=LIMIT))
    signed_in.goto(f"/flat-rules/{project['id']}")

    expect(signed_in.get_by_role("button", name="Add Rule")).to_be_visible()
    expect(rule_card(signed_in).get_by_text("$limit")).to_be_visible()  # the second band starts where the first ends
    assert crashes == []
    assert save(signed_in)["validation_errors"] == []
    assert decisions(export(signed_in, tmp_path), [500.0, 1500.0]) == ["approve", "decline"]


def test_strict_range_bound_set_to_a_parameter_keeps_the_editor_alive(signed_in: Page):
    crashes = []
    signed_in.on("pageerror", lambda e: crashes.append(str(e)))
    literal = {**STRICT_RANGES, "conditions": [{"when": {"min": None, "max": 1000}, "then": 0},
                                              {"when": {"min": None, "max": 5000}, "then": 1}]}
    open_seeded(signed_in, literal, parameters=LIMIT)
    use_parameter(rule_card(signed_in), "limit")

    expect(rule_card(signed_in).get_by_text("$limit")).to_be_visible()
    assert crashes == []
    assert save(signed_in)["validation_errors"] == []


def test_node_editor_strict_ranges_with_a_parameter_bound_open(signed_in: Page):
    crashes = []
    signed_in.on("pageerror", lambda e: crashes.append(str(e)))

    def leaf(i: int, idx: int) -> dict:
        return {"id": f"leaf{i}", "position": {"x": 150.0 * i, "y": 250.0}, "data": {"type": "leaf", "result_idx": idx}}

    document = {
        "type": "ui-tree", "formatVersion": 3, "parameters": LIMIT, "subtrees": [],
        "input_schema": [["amount", "Float64"]],
        "output": {"dtypes": [["decision", "String"]], "type_defs": {}, "data": [{"decision": "approve"}],
                   "default": {"decision": "decline"}},
        "nodes": [{"id": "bands", "position": {"x": 150.0, "y": 0.0}, "data": {
            "type": "cases", "op": "ranges", "strict": True, "feature": "amount",
            "conditions": [{"max": {"param": "limit"}}, {"max": 5000}]}}, leaf(0, 0), leaf(1, -1), leaf(2, -1)],
        "edges": [{"id": f"bands:leaf{i}", "source": "bands", "target": f"leaf{i}", "data": {"sourceIndex": [i]}}
                  for i in range(3)],
    }
    project = create_project(signed_in, "Strict node", document, project_type="tree")
    assert project["validation_errors"] == []
    signed_in.goto(f"/projects/{project['id']}")
    signed_in.locator(".react-flow__node").first.click()

    expect(signed_in.get_by_text("$limit").first).to_be_visible()
    assert crashes == []


def test_computed_feature_suggestions_are_expressions_the_engine_accepts(signed_in: Page):
    open_seeded(signed_in, parameters=LIMIT)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Unary")
    card.get_by_title("Switch to computed feature").click()
    expression = card.get_by_placeholder("Expression (e.g., feature1 + feature2)")

    expression.fill("amount - p")
    expect(card.get_by_text("p.limit")).to_have_count(0)  # decikit expressions have no parameters
    expression.fill("ab")
    expect(card.get_by_text("abs", exact=True)).to_be_visible()
    expression.fill("da")
    expect(card.get_by_text("date", exact=True)).to_have_count(0)  # nor polars functions


def test_computed_feature_takes_a_decimal_point(signed_in: Page):
    open_seeded(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Unary")
    card.get_by_title("Switch to computed feature").click()
    expression = card.get_by_placeholder("Expression (e.g., feature1 + feature2)")

    expression.press_sequentially("amount * 0.5")
    expect(expression).to_have_value("amount * 0.5")
    assert save(signed_in)["validation_errors"] == []
