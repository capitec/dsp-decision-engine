"""Invalid input in the flat-rule editor: after Save the offending rule is marked with a message that says what to fix,
Export is refused with an explanation, and fixing the input clears the marker and lets the export through."""

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from flat_docs import APPROVE, open_doc
from flat_rules_ui import (
    add_then,
    choose_feature,
    choose_operator,
    export,
    fill_number,
    fill_parameter,
    open_tab,
    rule_card,
    save,
    set_node_type,
    unary_rule,
    use_parameter,
)

LIMIT = {"limit": {"type": "Float64", "default_value": 1000}}


def refused_export(page: Page, message: re.Pattern | str) -> None:
    page.get_by_role("button", name="Export").click()
    toast = page.get_by_role("status").filter(has_text="Export failed").last
    expect(toast).to_be_visible()
    expect(toast).to_contain_text(message)


def reported_on_the_rule(page: Page, message: re.Pattern | str) -> Locator:
    """After Save: the rule card is marked red and its error list names the problem."""
    reply = save(page)
    assert reply["validation_errors"], "the document was saved as valid"
    card = rule_card(page)
    expect(card).to_have_class(re.compile("border-red-500"))
    expect(card.get_by_test_id("rule-errors")).to_contain_text(message)
    expect(page.get_by_role("tab", name=re.compile("Rules")).get_by_text("!")).to_be_visible()
    return card


def fixed(page: Page, tmp_path) -> None:
    """After the fix: Save reports nothing, the marker is gone and Export downloads."""
    assert save(page)["validation_errors"] == []
    expect(page.get_by_test_id("rule-errors")).to_have_count(0)
    expect(rule_card(page)).not_to_have_class(re.compile("border-red-500"))
    assert export(page, tmp_path)["rules"]


@pytest.mark.smoke
def test_empty_composite_is_reported_and_fixed_by_adding_a_condition(signed_in: Page, tmp_path):
    open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Composite")
    add_then(signed_in, card)

    reported_on_the_rule(signed_in, "composite needs at least one condition")
    refused_export(signed_in, "composite needs at least one condition")
    card.get_by_role("button", name="Add Condition", exact=True).click()
    choose_feature(signed_in, card.get_by_test_id("condition-row"), "amount")
    fixed(signed_in, tmp_path)


def test_not_with_two_conditions_is_reported_and_fixed_by_switching_to_and(signed_in: Page, tmp_path):
    two = [{"type": "unary", "op": "<=", "feature": "amount", "threshold": 1},
           {"type": "unary", "op": ">", "feature": "amount", "threshold": 1}]
    open_doc(signed_in, {"type": "composite", "op": "not", "conditions": two, "then": APPROVE})

    # the invalid document is flagged as soon as it opens, and again after a save
    expect(rule_card(signed_in).get_by_test_id("rule-errors")).to_contain_text("NOT operator must have exactly 1")
    reported_on_the_rule(signed_in, "NOT operator must have exactly 1 condition")
    refused_export(signed_in, "NOT operator must have exactly 1 condition")
    choose_operator(signed_in, rule_card(signed_in), "and", testid="composite-op")
    fixed(signed_in, tmp_path)


def test_between_without_bounds_is_reported_and_fixed_by_giving_one(signed_in: Page, tmp_path):
    open_doc(signed_in)
    card = unary_rule(signed_in, "between")

    reported_on_the_rule(signed_in, "At least one of min or max must be specified")
    refused_export(signed_in, "At least one of min or max")
    fill_number(card, "10", 0)
    fixed(signed_in, tmp_path)


@pytest.mark.xfail(strict=True, reason="the message ends with `e.g. ''` and never says to pick a parameter")
def test_unfinished_parameter_choice_says_to_pick_a_parameter(signed_in: Page, tmp_path):
    open_doc(signed_in, parameters=LIMIT)
    card = unary_rule(signed_in, "<=")
    card.get_by_role("button", name="#").first.click()

    card = reported_on_the_rule(signed_in, "threshold")
    message = card.get_by_test_id("rule-errors").inner_text()
    assert "''" not in message and re.search(r"(choose|select|pick) a parameter|parameter .*(choose|select)", message, re.I)
    fill_parameter(card, "limit")
    fixed(signed_in, tmp_path)


def test_unfinished_parameter_choice_is_reported_on_the_threshold_and_export_refused(signed_in: Page, tmp_path):
    open_doc(signed_in, parameters=LIMIT)
    card = unary_rule(signed_in, "<=")
    card.get_by_role("button", name="#").first.click()

    reported_on_the_rule(signed_in, "threshold")
    refused_export(signed_in, "threshold")
    fill_parameter(card, "limit")
    fixed(signed_in, tmp_path)


@pytest.mark.xfail(strict=True, reason="a parameter name that is not defined saves as valid and exports")
def test_undefined_parameter_name_is_reported_and_names_the_parameter(signed_in: Page, tmp_path):
    open_doc(signed_in, parameters=LIMIT)
    card = unary_rule(signed_in, "<=")
    use_parameter(card, "ghost")

    reported_on_the_rule(signed_in, "ghost")
    refused_export(signed_in, "ghost")
    fill_parameter(card, "limit")
    fixed(signed_in, tmp_path)


def test_parameter_name_that_is_not_an_identifier_is_reported_with_a_suggestion(signed_in: Page, tmp_path):
    open_doc(signed_in, parameters=LIMIT)
    card = unary_rule(signed_in, "<=")
    use_parameter(card, "max limit")

    reported_on_the_rule(signed_in, re.compile(r"'max limit'.*max_limit"))
    refused_export(signed_in, "max limit")
    fill_parameter(card, "limit")
    fixed(signed_in, tmp_path)


@pytest.mark.xfail(strict=True, reason="a condition without a feature saves as valid and exports")
def test_condition_without_a_feature_is_reported_and_fixed_by_choosing_one(signed_in: Page, tmp_path):
    open_doc(signed_in)
    card = rule_card(signed_in)
    set_node_type(signed_in, card, "Unary")
    add_then(signed_in, card)

    reported_on_the_rule(signed_in, "feature")
    refused_export(signed_in, "feature")
    choose_feature(signed_in, card, "amount")
    fixed(signed_in, tmp_path)


@pytest.mark.xfail(strict=True, reason="the error has no rule path and names a node id, so nothing is marked after Save")
def test_leaf_pointing_past_the_output_records_is_reported_on_the_rule(signed_in: Page):
    open_doc(signed_in, {"type": "leaf", "result_idx": 9})

    card = reported_on_the_rule(signed_in, re.compile("result_idx|output record", re.I))
    assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}", card.get_by_test_id("rule-errors").inner_text())


def test_leaf_pointing_past_the_output_records_refuses_the_export_with_the_reason(signed_in: Page):
    open_doc(signed_in, {"type": "leaf", "result_idx": 9})

    assert save(signed_in)["validation_errors"]
    refused_export(signed_in, "result_idx 9 is not a row of the output")


@pytest.mark.xfail(strict=True, reason="deleting the last output column is reported nowhere in the editor after Save")
def test_no_output_columns_is_reported_on_the_outputs_tab(signed_in: Page, tmp_path):
    open_doc(signed_in)
    open_tab(signed_in, "Configure Outputs")
    signed_in.get_by_title("Delete field").click()

    assert save(signed_in)["validation_errors"]
    expect(signed_in.get_by_role("tab", name=re.compile("Configure Outputs")).get_by_text("!")).to_be_visible()
    expect(signed_in.get_by_role("tabpanel")).to_contain_text("no columns")
    refused_export(signed_in, "no columns")
    signed_in.get_by_role("button", name="Add First Column").click()
    signed_in.get_by_placeholder("Column name").fill("decision")
    signed_in.get_by_role("button", name="Add", exact=True).click()
    assert save(signed_in)["validation_errors"] == []


def test_no_output_columns_refuses_the_export_with_the_reason(signed_in: Page):
    open_doc(signed_in)
    open_tab(signed_in, "Configure Outputs")
    signed_in.get_by_title("Delete field").click()

    assert save(signed_in)["validation_errors"]
    refused_export(signed_in, "declares no columns")
