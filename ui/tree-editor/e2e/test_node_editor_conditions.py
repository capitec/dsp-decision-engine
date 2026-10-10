"""Node-editor conditions: every node kind and operator the sidebar offers saves a document decikit runs."""

import operator

import pytest
from playwright.sync_api import Page

from node_ui import (
    choose_feature,
    loan_document,
    open_tree,
    outcome,
    run,
    save,
    saved_data,
    select_node,
    set_operator,
    set_threshold,
    sidebar,
    two_leaves,
    unary,
)
from seed import export_document


@pytest.mark.parametrize("symbol, op, compare", [
    pytest.param("≤", "<=", operator.le, marks=pytest.mark.smoke), ("<", "<", operator.lt), ("==", "==", operator.eq),
    ("!=", "!=", operator.ne), (">", ">", operator.gt), ("≥", ">=", operator.ge),
])
def test_unary_comparison_operator_saves_and_decides_at_the_boundary(signed_in: Page, symbol, op, compare):
    project = open_tree(signed_in, loan_document(limit=7, op="<="))
    select_node(signed_in, "root")
    set_operator(signed_in, symbol)
    set_threshold(signed_in, "1000")
    assert save(signed_in)["validation_errors"] == []

    condition = saved_data(signed_in, project)["condition"]
    assert (condition["op"], condition["threshold"]) == (op, 1000)
    amounts = [999.0, 1000.0, 1001.0]
    assert run(export_document(signed_in, project), amount=amounts) == outcome([compare(a, 1000) for a in amounts])


@pytest.mark.parametrize("lower, upper, truth", [
    ("100", None, [False, True, True, True]),
    (None, "1000", [True, True, True, False]),
    ("100", "1000", [False, True, True, False]),
])
def test_between_with_an_open_end_saves_and_includes_both_bounds(signed_in: Page, lower, upper, truth):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_operator(signed_in, "Between")
    bounds = sidebar(signed_in).get_by_placeholder("null")
    if lower:
        bounds.nth(0).fill(lower)
    if upper:
        bounds.nth(1).fill(upper)
    assert save(signed_in)["validation_errors"] == []

    condition = saved_data(signed_in, project)["condition"]
    assert (condition["min"], condition["max"]) == (int(lower) if lower else None, int(upper) if upper else None)
    assert run(export_document(signed_in, project), amount=[50.0, 100.0, 1000.0, 1001.0]) == outcome(truth)


@pytest.mark.parametrize("text, value, truth", [
    ("-5", -5, [False, True, True]), ("0.1", 0.1, [False, False, True]), ("1e15", 1e15, [True, True, True]),
])
def test_negative_decimal_and_very_large_thresholds_are_saved_exactly(signed_in: Page, text, value, truth):
    project = open_tree(signed_in, two_leaves(unary("root", ">=", threshold=1)))
    select_node(signed_in, "root")
    set_threshold(signed_in, text)
    assert save(signed_in)["validation_errors"] == []

    assert saved_data(signed_in, project)["condition"]["threshold"] == value
    amounts = [-10.0, 0.0, 0.5]
    expected = [a >= value for a in amounts]
    assert run(export_document(signed_in, project), amount=amounts) == outcome(expected)


@pytest.mark.xfail(strict=True, reason="typing inf or nan in a threshold shows it in the box but saves 0")
@pytest.mark.parametrize("text", ["inf", "-inf", "nan"])
def test_special_number_typed_in_a_threshold_is_not_silently_saved_as_zero(signed_in: Page, text):
    project = open_tree(signed_in, two_leaves(unary("root", threshold=5)))
    select_node(signed_in, "root")
    set_threshold(signed_in, text)
    save(signed_in)

    assert saved_data(signed_in, project)["condition"]["threshold"] != 0


@pytest.mark.xfail(strict=True, reason="a threshold too large for a float (1e400) becomes null in the saved document")
def test_threshold_beyond_float_range_is_not_saved_as_null(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root", threshold=5)))
    select_node(signed_in, "root")
    set_threshold(signed_in, "1e400")
    reply = save(signed_in)

    assert saved_data(signed_in, project)["condition"]["threshold"] is not None or reply["validation_errors"] == []


def test_isin_on_an_integer_feature_saves_the_values_and_decides(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_operator(signed_in, "IsIn")
    choose_feature(signed_in, "code", current="amount")
    values = sidebar(signed_in).locator("input").last  # the placeholder goes once the first value is a chip
    for value in ("1", "2"):
        values.fill(value)
        values.press("Enter")
    assert save(signed_in)["validation_errors"] == []

    assert saved_data(signed_in, project)["condition"]["values"] == [1, 2]
    assert run(export_document(signed_in, project), code=[1, 2, 3]) == ["approve", "approve", "review"]


@pytest.mark.parametrize("match_type, pattern, truth", [
    ("Exact", "gold", [True, False, False, False]),
    ("Starts With", "gold", [True, True, False, False]),
    ("Contains", "gold", [True, True, True, False]),
    ("Ends With", "gold", [True, False, True, False]),
    ("Regex", "^gol.*n$", [False, True, False, False]),
])
def test_string_match_type_saves_and_matches(signed_in: Page, match_type, pattern, truth):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_operator(signed_in, "String Match")
    choose_feature(signed_in, "band", current="amount")
    sidebar(signed_in).get_by_placeholder("Pattern 1").fill(pattern)
    sidebar(signed_in).get_by_role("combobox").filter(has_text="Exact").click()
    signed_in.get_by_role("option", name=match_type, exact=True).click()
    assert save(signed_in)["validation_errors"] == []

    assert saved_data(signed_in, project)["condition"]["match_type"] == match_type.lower().replace(" ", "_")
    assert run(export_document(signed_in, project), band=["gold", "golden", "a gold", "silver"]) == outcome(truth)


def test_string_match_case_insensitive_with_trimmed_whitespace_matches(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_operator(signed_in, "String Match")
    choose_feature(signed_in, "band", current="amount")
    sidebar(signed_in).get_by_placeholder("Pattern 1").fill("gold")
    sidebar(signed_in).get_by_label("Case Sensitive").click()
    sidebar(signed_in).get_by_label("Trim Whitespace").click()
    assert save(signed_in)["validation_errors"] == []

    condition = saved_data(signed_in, project)["condition"]
    assert (condition["case_sensitive"], condition["trim_whitespace"]) == (False, True)
    assert run(export_document(signed_in, project), band=[" GOLD ", "silver"]) == ["approve", "review"]


def test_string_match_with_two_patterns_matches_either(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_operator(signed_in, "String Match")
    choose_feature(signed_in, "band", current="amount")
    sidebar(signed_in).get_by_placeholder("Pattern 1").fill("gold")
    sidebar(signed_in).get_by_role("button", name="Add Pattern").click()
    sidebar(signed_in).get_by_placeholder("Pattern 2").fill("silver")
    assert save(signed_in)["validation_errors"] == []

    assert run(export_document(signed_in, project), band=["gold", "silver", "bronze"]) == ["approve", "approve", "review"]


@pytest.mark.parametrize("label, op, truth", [("Is True", "is_true", [True, False]), ("Is False", "is_false", [False, True])])
def test_boolean_check_saves_and_decides(signed_in: Page, label, op, truth):
    project = open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_operator(signed_in, label)
    choose_feature(signed_in, "flag", current="amount")
    assert save(signed_in)["validation_errors"] == []

    assert saved_data(signed_in, project)["condition"]["op"] == op
    assert run(export_document(signed_in, project), flag=[True, False]) == outcome(truth)


@pytest.mark.xfail(strict=True, reason="Is Null / Is Not Null are offered but decikit has no null operators: the saved tree is invalid")
@pytest.mark.parametrize("label", ["Is Null", "Is Not Null"])
def test_null_check_the_editor_offers_saves_a_valid_document(signed_in: Page, label):
    open_tree(signed_in, two_leaves(unary("root")))
    select_node(signed_in, "root")
    set_operator(signed_in, label)

    assert save(signed_in)["validation_errors"] == []


def test_computed_feature_is_saved_as_an_expression_and_evaluated(signed_in: Page):
    project = open_tree(signed_in, two_leaves(unary("root", threshold=1000)))
    select_node(signed_in, "root")
    sidebar(signed_in).get_by_title("Switch to computed feature").click()
    sidebar(signed_in).get_by_placeholder("feature or expression (e.g., feature1 + feature2)").fill("amount * 2")
    assert save(signed_in)["validation_errors"] == []

    assert saved_data(signed_in, project)["condition"]["feature"] == {"type": "computed", "expression": "amount * 2"}
    assert run(export_document(signed_in, project), amount=[400.0, 600.0]) == ["approve", "review"]


@pytest.mark.parametrize("expression, amounts, truth", [
    ("abs(amount)", [-900.0, 1500.0], [True, False]),
    ("min(amount, code)", [5000.0, 5000.0], [True, False]),
    ("amount - code", [1500.0, 1500.0], [False, True]),
])
def test_computed_feature_functions_over_features_run(signed_in: Page, expression, amounts, truth):
    project = open_tree(signed_in, two_leaves(unary("root", threshold=1000)))
    select_node(signed_in, "root")
    sidebar(signed_in).get_by_title("Switch to computed feature").click()
    sidebar(signed_in).get_by_placeholder("feature or expression (e.g., feature1 + feature2)").fill(expression)
    assert save(signed_in)["validation_errors"] == []

    codes = [500, 5000] if "min" in expression else [1, 600]
    assert run(export_document(signed_in, project), amount=amounts, code=codes) == outcome(truth)
