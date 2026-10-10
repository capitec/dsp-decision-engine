"""Output columns and records, the input schema, and what rules keep pointing at after they change."""

import pytest
from playwright.sync_api import Page, expect

from flat_docs import APPROVE, DEFAULT, open_doc, run_frame
from flat_rules_ui import (
    add_output_column,
    delete_output_column,
    open_tab,
    output_cell,
    rename_output_column,
    rule_card,
    save,
    set_output_column_type,
)
from seed import export_document

THREE = {"dtypes": [["decision", "String"]], "type_defs": {}, "default": {"decision": "decline"},
         "data": [{"decision": "approve"}, {"decision": "review"}, {"decision": "escalate"}]}
SCORED = {"dtypes": [["decision", "String"], ["score", "Int64"]], "type_defs": {}, "default": {"decision": "decline", "score": 0},
          "data": [{"decision": "approve", "score": 5}, {"decision": "review", "score": 7}]}


def rule_with_result(idx: int) -> dict:
    return {"type": "unary", "condition": {"type": "unary", "op": ">=", "feature": "amount", "threshold": 0},
            "then": {"type": "leaf", "result_idx": idx}}


def saved_output(reply: dict) -> dict:
    return reply["project_data"]["output"]


def test_added_output_column_reaches_every_record_and_the_decikit_output(signed_in: Page):
    project = open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    add_output_column(signed_in, "note")
    output_cell(signed_in, 0, 1).fill("n1")

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    out = saved_output(reply)
    assert out["dtypes"] == [["decision", "String"], ["note", "String"]]
    assert out["data"][0] == {"decision": "approve", "note": "n1"}
    assert run_frame(export_document(signed_in, project), {"amount": [1.0]}) == [{"decision": "approve", "note": "n1"}]


def test_renamed_output_column_is_renamed_in_every_record_and_in_decikit(signed_in: Page):
    project = open_doc(signed_in, APPROVE)
    open_tab(signed_in, "Configure Outputs")
    rename_output_column(signed_in, "decision", "verdict")

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    out = saved_output(reply)
    assert (out["dtypes"], out["data"], out["default"]) == (
        [["verdict", "String"]], [{"verdict": "approve"}], {"verdict": "decline"})
    assert run_frame(export_document(signed_in, project), {"amount": [1.0]}) == [{"verdict": "approve"}]


def test_deleted_output_column_leaves_the_other_columns_valid(signed_in: Page):
    project = open_doc(signed_in, APPROVE, output=SCORED)
    open_tab(signed_in, "Configure Outputs")
    delete_output_column(signed_in, "score")

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_output(reply)["dtypes"] == [["decision", "String"]]
    assert run_frame(export_document(signed_in, project), {"amount": [1.0]}) == [{"decision": "approve"}]


@pytest.mark.xfail(strict=True, reason="changing a column's type overwrites every record's value with the default")
def test_changing_an_output_column_type_keeps_the_values_that_convert(signed_in: Page):
    open_doc(signed_in, APPROVE, output=SCORED)
    open_tab(signed_in, "Configure Outputs")
    set_output_column_type(signed_in, "score", "Float64")

    out = saved_output(save(signed_in))
    assert out["dtypes"][1] == ["score", "Float64"]
    assert [row["score"] for row in out["data"]] == [5, 7]


def test_added_record_can_be_chosen_by_a_leaf_and_runs(signed_in: Page):
    project = open_doc(signed_in, DEFAULT)
    open_tab(signed_in, "Configure Outputs")
    signed_in.get_by_role("button", name="Add Record").click()
    output_cell(signed_in, 1, 0).fill("review")
    open_tab(signed_in, "Rules")
    card = rule_card(signed_in)
    card.get_by_test_id("leaf-output").get_by_role("combobox").click()
    signed_in.get_by_role("option").filter(has_text="review").first.click()

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert saved_output(reply)["data"][-1] == {"decision": "review"}
    assert reply["project_data"]["rules"][0]["rule"]["result_idx"] == 1
    assert run_frame(export_document(signed_in, project), {"amount": [1.0]}) == [{"decision": "review"}]


def test_the_first_record_has_no_delete_button(signed_in: Page):
    open_doc(signed_in, APPROVE, output=THREE)
    open_tab(signed_in, "Configure Outputs")

    expect(signed_in.get_by_test_id("output-record-delete")).to_have_count(2)
    expect(signed_in.locator("tbody tr").first.get_by_text("Default")).to_be_visible()


def test_deleting_a_record_repoints_leaves_to_the_same_remaining_records(signed_in: Page):
    # first_match with one condition per rule: rule 1 -> record 1, rule 2 -> record 2 (different amount bands)
    low = {"type": "unary", "condition": {"type": "unary", "op": "<=", "feature": "amount", "threshold": 10},
           "then": {"type": "leaf", "result_idx": 1}}
    high = {"type": "unary", "condition": {"type": "unary", "op": ">", "feature": "amount", "threshold": 10},
            "then": {"type": "leaf", "result_idx": 2}}
    project = open_doc(signed_in, low, high, output=THREE)
    open_tab(signed_in, "Configure Outputs")
    signed_in.get_by_test_id("output-record-delete").first.click()  # record 1 ("review")

    reply = save(signed_in)
    assert reply["validation_errors"] == []
    assert [r["rule"]["then"]["result_idx"] for r in reply["project_data"]["rules"]] == [-1, 1]
    assert saved_output(reply)["data"] == [{"decision": "approve"}, {"decision": "escalate"}]
    rows = run_frame(export_document(signed_in, project), {"amount": [5.0, 50.0]})
    assert rows == [{"decision": "decline"}, {"decision": "escalate"}]


@pytest.mark.xfail(strict=True, reason="editing the first output record also overwrites the default row")
def test_editing_the_first_record_leaves_the_default_row_alone(signed_in: Page):
    open_doc(signed_in, APPROVE, output=THREE)
    open_tab(signed_in, "Configure Outputs")
    output_cell(signed_in, 0, 0).fill("approved")

    out = saved_output(save(signed_in))
    assert (out["data"][0]["decision"], out["default"]["decision"]) == ("approved", "decline")


def test_input_fields_can_be_added_renamed_retyped_reordered_and_deleted(signed_in: Page):
    open_doc(signed_in, DEFAULT)
    open_tab(signed_in, "Configure Inputs")
    rows = signed_in.get_by_test_id("schema-field-row")
    signed_in.get_by_role("button", name="Add Field").click()
    new = rows.last
    expect(new.get_by_placeholder("Field name")).to_have_value("field_5")
    new.get_by_placeholder("Field name").fill("income")
    new.get_by_role("combobox").click()
    signed_in.get_by_role("option", name="Int64", exact=True).click()
    new.get_by_role("button").first.click()  # move up

    reply = save(signed_in)
    schema = reply["project_data"]["input_schema"]
    assert reply["validation_errors"] == []
    assert schema[-2:] == [["income", "Int64"], ["band", "String"]]

    rows.nth(len(schema) - 2).get_by_test_id("schema-field-remove").click()
    reply = save(signed_in)
    assert [name for name, _ in reply["project_data"]["input_schema"]] == ["amount", "count", "flag", "band"]


@pytest.mark.xfail(strict=True, reason="renaming an input field leaves rules reading the old name, unreported")
def test_renaming_an_input_field_updates_or_flags_the_conditions_that_use_it(signed_in: Page):
    open_doc(signed_in, rule_with_result(0))
    open_tab(signed_in, "Configure Inputs")
    signed_in.get_by_test_id("schema-field-row").first.get_by_placeholder("Field name").fill("balance")

    reply = save(signed_in)
    followed = reply["project_data"]["rules"][0]["rule"]["condition"]["feature"] == "balance"
    assert followed or reply["validation_errors"]


@pytest.mark.xfail(strict=True, reason="deleting an input field a condition reads is not reported")
def test_deleting_an_input_field_flags_the_conditions_that_use_it(signed_in: Page):
    open_doc(signed_in, rule_with_result(0))
    open_tab(signed_in, "Configure Inputs")
    signed_in.get_by_test_id("schema-field-row").first.get_by_test_id("schema-field-remove").click()

    reply = save(signed_in)
    assert reply["project_data"]["rules"][0]["rule"]["condition"]["feature"] != "amount" or reply["validation_errors"]
