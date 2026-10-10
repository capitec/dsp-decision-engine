"""Every condition operator of the flat-rule editor, driven through the UI: what is saved and what decikit then does."""

import re

import pytest
from playwright.sync_api import Page, expect

from flat_docs import DEFAULT, decisions_for, open_doc
from flat_rules_ui import (
    OPERATOR_LABELS,
    add_isin_values,
    add_parameter,
    add_then,
    choose_feature,
    choose_match_type,
    choose_operator,
    fill_number,
    fill_pattern,
    number_input,
    rule_card,
    save,
    set_node_type,
    unary_rule,
    use_parameter,
)
from seed import export_document


def saved_condition(reply: dict) -> dict:
    return reply["project_data"]["rules"][0]["rule"]["condition"]


def approved(doc: dict, frame: dict) -> list[bool]:
    return [d == "approve" for d in decisions_for(doc, frame)]


@pytest.mark.smoke
def test_less_than_integer_threshold_saves_and_runs(signed_in: Page):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "<")
    fill_number(card, "1000")

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_condition(reply) == {"type": "unary", "op": "<", "feature": "amount", "threshold": 1000}
    assert approved(export_document(signed_in, project), {"amount": [999.0, 1000.0, 1001.0]}) == [True, False, False]


@pytest.mark.parametrize("op, text, threshold, amounts, expected", [
    ("<=", "-5", -5, [-6.0, -5.0, -4.0], [True, True, False]),
    ("==", "99.5", 99.5, [99.5, 99.0], [True, False]),
    ("!=", "1e15", 1e15, [1e15, 5.0], [False, True]),
    (">", "0", 0, [-1.0, 0.0, 1.0], [False, False, True]),
    (">=", "250", 250, [249.99, 250.0], [False, True]),
])
def test_comparison_operator_saves_its_threshold_and_runs(signed_in: Page, op, text, threshold, amounts, expected):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, op)
    fill_number(card, text)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_condition(reply)["op"] == op
    assert saved_condition(reply)["threshold"] == threshold
    assert approved(export_document(signed_in, project), {"amount": amounts}) == expected


@pytest.mark.parametrize("fields, bounds, amounts, expected", [
    ({0: "10", 1: "20"}, {"min": 10, "max": 20}, [9.0, 10.0, 15.0, 20.0, 21.0], [False, True, True, True, False]),
    ({0: "10"}, {"min": 10, "max": None}, [9.0, 10.0, 5000.0], [False, True, True]),
    ({1: "20"}, {"min": None, "max": 20}, [-5000.0, 20.0, 21.0], [True, True, False]),
])
def test_between_with_both_one_or_open_bounds_saves_and_runs(signed_in: Page, fields, bounds, amounts, expected):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "between")
    for index, text in fields.items():
        fill_number(card, text, index)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert {k: saved_condition(reply)[k] for k in ("min", "max")} == bounds
    assert approved(export_document(signed_in, project), {"amount": amounts}) == expected


@pytest.mark.xfail(strict=True, reason="clearing a numeric field stores 0 instead of leaving the bound open")
def test_clearing_a_between_bound_leaves_it_open(signed_in: Page):
    open_doc(signed_in)
    card = unary_rule(signed_in, "between")
    fill_number(card, "10", 0)
    fill_number(card, "20", 1)
    fill_number(card, "", 0)

    assert saved_condition(save(signed_in))["min"] is None


def test_isin_integers_save_sorted_and_run(signed_in: Page):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "isin", feature="count")
    add_isin_values(card, [3, 1, 2])

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_condition(reply)["values"] == [1, 2, 3]
    assert approved(export_document(signed_in, project), {"count": [0, 1, 3, 4]}) == [False, True, True, False]


@pytest.mark.xfail(strict=True, reason="the IsIn field truncates 1.5 to 1 without saying so")
def test_isin_decimal_value_is_not_silently_truncated(signed_in: Page):
    open_doc(signed_in)
    card = unary_rule(signed_in, "isin")
    add_isin_values(card, ["1.5"])

    assert saved_condition(save(signed_in))["values"] != [1]


def test_isin_offers_no_negation(signed_in: Page):
    open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Unary")
    card.get_by_test_id("unary-op").click()

    options = signed_in.get_by_role("option").all_inner_texts()
    assert "IsIn" in options
    assert not [o for o in options if "not in" in o.lower() or "notin" in o.lower().replace(" ", "")]


@pytest.mark.parametrize("match_type, pattern, bands, expected", [
    ("exact", "gold", ["gold", "silver", "golden"], [True, False, False]),
    ("starts_with", "gol", ["gold", "silver", "golden", "ngold"], [True, False, True, False]),
    ("contains", "ilv", ["gold", "silver", "ilv"], [False, True, True]),
    ("ends_with", "er", ["gold", "silver", "ers"], [False, True, False]),
    ("regex", "^g.*d$", ["gold", "golden", "gd"], [True, False, True]),
])
def test_string_match_type_saves_and_runs(signed_in: Page, match_type, pattern, bands, expected):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "string_match", feature="band")
    fill_pattern(card, pattern)
    choose_match_type(signed_in, card, match_type)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    saved = saved_condition(reply)
    assert (saved["match_type"], saved["patterns"]) == (match_type, [pattern])
    assert approved(export_document(signed_in, project), {"band": bands}) == expected


def test_string_match_case_insensitive_and_trimmed_saves_and_runs(signed_in: Page):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "string_match", feature="band")
    fill_pattern(card, "gold")
    card.get_by_role("checkbox", name="Case Sensitive").click()
    card.get_by_role("checkbox", name="Trim Whitespace").click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert (saved_condition(reply)["case_sensitive"], saved_condition(reply)["trim_whitespace"]) == (False, True)
    frame = {"band": ["GOLD", " Gold ", "gold", "silver"]}
    assert approved(export_document(signed_in, project), frame) == [True, True, True, False]


def test_string_match_with_two_patterns_matches_either(signed_in: Page):
    project = open_doc(signed_in)
    card = unary_rule(signed_in, "string_match", feature="band")
    fill_pattern(card, "gold", 0)
    card.get_by_role("button", name="Add Pattern").click()
    fill_pattern(card, "silver", 1)

    reply = save(signed_in)
    assert saved_condition(reply)["patterns"] == ["gold", "silver"]
    assert approved(export_document(signed_in, project), {"band": ["gold", "silver", "bronze"]}) == [True, True, False]


@pytest.mark.parametrize("op, expected", [("is_true", [True, False]), ("is_false", [False, True])])
def test_boolean_checks_save_and_run(signed_in: Page, op, expected):
    project = open_doc(signed_in)
    unary_rule(signed_in, op, feature="flag")

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_condition(reply) == {"type": "unary", "op": op, "feature": "flag"}
    assert approved(export_document(signed_in, project), {"flag": [True, False]}) == expected


@pytest.mark.xfail(strict=True, reason="the operator list offers Is Null and Is Not Null, which decikit's schema does not accept")
@pytest.mark.parametrize("op", ["is_null", "is_not_null"])
def test_a_null_check_is_either_not_offered_or_saves_a_valid_document(signed_in: Page, op):
    open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Unary")
    card.get_by_test_id("unary-op").click()
    expect(signed_in.get_by_role("option", name="Between")).to_be_visible()
    option = signed_in.get_by_role("option", name=OPERATOR_LABELS[op], exact=True)
    if option.count() == 0:
        return  # decikit has no null checks, so not offering one is right too
    option.click()
    choose_feature(signed_in, card, "amount")
    add_then(signed_in, card)

    assert save(signed_in)["validation_errors"] == []


def test_computed_feature_expression_saves_and_runs(signed_in: Page):
    project = open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Unary")
    card.get_by_title("Switch to computed feature").click()
    card.get_by_placeholder("Expression (e.g., feature1 + feature2)").fill("amount * 2 - 1")
    fill_number(card, "9")
    add_then(signed_in, card)

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_condition(reply)["feature"] == {"type": "computed", "expression": "amount * 2 - 1"}
    assert approved(export_document(signed_in, project), {"amount": [5.0, 6.0]}) == [True, False]


@pytest.mark.parametrize("text", ["inf", "-inf", "nan"])
def test_a_non_finite_threshold_is_refused_and_the_number_kept(signed_in: Page, text):
    open_doc(signed_in)
    card = unary_rule(signed_in, "<")
    fill_number(card, "5")
    fill_number(card, text)

    expect(number_input(card)).to_have_class(re.compile("border-red-500"))
    assert saved_condition(save(signed_in))["threshold"] == 5


@pytest.mark.xfail(strict=True, reason="a range bound offers inf, -inf and nan, then stores 0 for them")
@pytest.mark.parametrize("text", ["inf", "-inf", "nan"])
def test_a_non_finite_range_bound_is_not_saved_as_zero(signed_in: Page, text):
    bands = {"type": "cases", "op": "ranges", "strict": True, "feature": "amount", "otherwise": 0,
             "conditions": [{"when": {"min": None, "max": 10}, "then": 0}], "branches": [DEFAULT]}
    open_doc(signed_in, bands)
    fill_number(rule_card(signed_in), text)

    saved = save(signed_in)
    assert saved["project_data"]["rules"][0]["rule"]["conditions"][0]["when"]["max"] in (None, 10)


@pytest.mark.xfail(strict=True, reason="changing the operator resets the threshold to 0")
def test_changing_the_operator_keeps_a_literal_threshold(signed_in: Page):
    open_doc(signed_in)
    card = unary_rule(signed_in, "<=")
    fill_number(card, "1000")
    choose_operator(signed_in, card, "<")

    assert saved_condition(save(signed_in))["threshold"] == 1000


@pytest.mark.xfail(strict=True, reason="changing the operator replaces a parameter threshold with 0")
def test_changing_the_operator_keeps_a_parameter_threshold(signed_in: Page):
    open_doc(signed_in)
    add_parameter(signed_in, "limit", "Float64", "1000")
    card = unary_rule(signed_in, "<=")
    use_parameter(card, "limit")
    choose_operator(signed_in, card, ">")

    assert saved_condition(save(signed_in))["threshold"] == {"param": "limit"}


def test_changing_the_feature_keeps_the_operator_and_threshold(signed_in: Page):
    open_doc(signed_in)
    card = unary_rule(signed_in, ">=")
    fill_number(card, "7")
    card.get_by_role("combobox").filter(has_text="amount").click()
    signed_in.get_by_role("option", name="count", exact=True).click()

    saved = saved_condition(save(signed_in))
    assert (saved["feature"], saved["op"], saved["threshold"]) == ("count", ">=", 7)
    expect(number_input(card)).to_have_value("7")
