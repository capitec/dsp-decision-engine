"""Struct outputs flatten to `column.field` columns; lists of values and of struct records are list columns."""
import polars as pl
import pytest

from decider._engine import Engine
from decider.steps.trees import TreeConfig

TYPE_DEFS = {
    "act": {"name": "Act", "type": "struct", "definition": {
        "fields": [["Outcome", "String"], ["Code", "Int64"], ["Band", {"type": "Custom", "type_id": "band"}]],
        "display_field": "Outcome",
        "data": [{"Outcome": "Approve", "Code": 1, "Band": "low"},
                 {"Outcome": "Decline", "Code": 2, "Band": "high"}]}},
    "band": {"name": "Band", "type": "categorical", "definition": {"categories": ["low", "high"]}},
}
ACT = {"type": "Custom", "type_id": "act"}
APPROVE = {"Outcome": "Approve", "Code": 1, "Band": "low"}
DECLINE = {"Outcome": "Decline", "Code": 2, "Band": "high"}
DF = pl.DataFrame({"score": [0.9, 0.1]})


def _tree(dtypes, data, default=None, condition=None, type_defs=TYPE_DEFS):
    condition = condition or {"op": ">=", "feature": "score", "threshold": 0.5}
    return TreeConfig(name="t", tree={
        "type": "ui-tree", "formatVersion": 3,
        "nodes": [{"id": "a", "data": {"type": "unary", "condition": condition}},
                  {"id": "hi", "data": {"type": "leaf", "result_idx": 0}},
                  {"id": "lo", "data": {"type": "leaf", "result_idx": 1}}],
        "edges": [{"source": "a", "target": "hi", "data": {"sourceIndex": [0]}},
                  {"source": "a", "target": "lo", "data": {"sourceIndex": [1]}}],
        "output": {"dtypes": dtypes, "type_defs": type_defs, "data": data, "default": default}})


def test_a_struct_column_becomes_one_column_per_field(run):
    out = run(_tree([["Action", ACT]], [{"Action": {"$key": "Approve"}}, {"Action": {"$key": "Decline"}}]), DF)
    assert out.drop("score").to_dicts() == [
        {"Action.Outcome": "Approve", "Action.Code": 1, "Action.Band": "low"},
        {"Action.Outcome": "Decline", "Action.Code": 2, "Action.Band": "high"}]
    assert out.schema["Action.Code"] == pl.Int64


@pytest.mark.parametrize("value", [{"$key": "Decline"}, "Decline", DECLINE, {**DECLINE, "Extra": 1}])
def test_a_struct_value_is_a_key_reference_a_bare_key_or_the_record(run, value):
    out = run(_tree([["Action", ACT]], [{"Action": "Approve"}, {"Action": value}]), DF)
    assert out["Action.Outcome"].to_list() == ["Approve", "Decline"]
    assert "Action.Extra" not in out.columns


def test_a_null_struct_is_null_in_every_field(run):
    out = run(_tree([["Action", ACT]], [{"Action": None}, {}]), DF)
    assert out.drop("score").to_dicts() == [{"Action.Outcome": None, "Action.Code": None, "Action.Band": None}] * 2


def test_an_inline_struct_type_flattens_too(run):
    out = run(_tree([["Limit", [["amount", "Float64"], ["term", "Int64"]]]],
                    [{"Limit": {"amount": 100.0, "term": 12}}, {"Limit": {"amount": 50.0, "term": 6}}]), DF)
    assert out["Limit.amount"].to_list() == [100.0, 50.0] and out["Limit.term"].to_list() == [12, 6]


def test_a_list_of_struct_records_is_one_list_column(run):
    tree = _tree([["Actions", {"type": "List", "inner": ACT}]],
                 [{"Actions": [{"$key": "Approve"}, "Decline"]}, {"Actions": []}], default={"Actions": None})
    out = run(tree, DF)
    assert out["Actions"].to_list() == [[APPROVE, DECLINE], []]
    assert out.schema["Actions"] == pl.List(pl.Struct({"Outcome": pl.String, "Code": pl.Int64, "Band": pl.String}))


def test_a_list_of_values_is_a_list_column(run):
    out = run(_tree([["Tags", {"type": "List", "inner": "Int64"}]], [{"Tags": [1, 2]}, {"Tags": None}]), DF)
    assert out["Tags"].to_list() == [[1, 2], None]
    assert out.schema["Tags"] == pl.List(pl.Int64)


def test_a_tree_reading_strings_returns_lists_too(run):
    tree = _tree([["Actions", {"type": "List", "inner": ACT}]], [{"Actions": ["Approve"]}, {"Actions": ["Decline"]}],
                 condition={"op": "string_match", "feature": "channel", "patterns": ["app"], "match_type": "exact"})
    out = run(tree, pl.DataFrame({"channel": ["app", "web"]}))
    assert out["Actions"].to_list() == [[APPROVE], [DECLINE]]


def test_flat_rules_with_every_rule_return_structs_and_lists(run):
    leaf = {"type": "leaf", "result_idx": 0}
    rule = {"type": "unary", "condition": {"op": ">=", "feature": "score", "threshold": 0.5}, "then": leaf}
    tree = TreeConfig(name="t", tree={
        "type": "prioritized_flat_rule", "mode": "all",
        "rules": [{"meta": {"name": "high"}, "rule": rule}],
        "output": {"dtypes": [["Action", ACT], ["Actions", {"type": "List", "inner": ACT}]], "type_defs": TYPE_DEFS,
                   "data": [{"Action": "Approve", "Actions": ["Approve"]}],
                   "default": {"Action": "Decline", "Actions": []}}})
    out = run(tree, DF)
    assert out["high.Action.Outcome"].to_list() == ["Approve", "Decline"]
    assert out["high.Actions"].to_list() == [[APPROVE], []]


@pytest.mark.parametrize("mode", ["interpreted", "stepped", "fused"])
def test_score_returns_records_a_caller_may_edit(mode):
    exe = Engine().bind(_tree([["Actions", {"type": "List", "inner": ACT}]], [{"Actions": ["Approve"]}, {}]), mode=mode)
    exe.score({"score": 0.9})["Actions"][0]["Code"] = 99
    assert exe.score({"score": 0.9})["Actions"] == [APPROVE]


def test_list_outputs_run_in_the_shared_kernel():
    tree = _tree([["Actions", {"type": "List", "inner": ACT}], ["Action", ACT]], [{"Actions": ["Approve"]}, {}])
    assert Engine().bind(tree, mode="fused").fallbacks() == {}


def test_an_unknown_struct_key_is_refused():
    with pytest.raises(ValueError, match="'Refer' is not a record of its struct type"):
        _tree([["Action", ACT]], [{"Action": "Refer"}, {}]).parameters()


def test_a_struct_nested_in_a_struct_is_refused():
    defs = {**TYPE_DEFS, "outer": {"name": "Outer", "type": "struct", "definition": {
        "fields": [["inner", ACT]], "display_field": "inner", "data": []}}}
    with pytest.raises(ValueError, match="not a nested struct"):
        _tree([["x", {"type": "Custom", "type_id": "outer"}]], [{}, {}], type_defs=defs).parameters()


def test_flattened_columns_that_clash_are_refused():
    with pytest.raises(ValueError, match=r"\['Action.Code'\] appear twice"):
        _tree([["Action", ACT], ["Action.Code", "Int64"]], [{}, {}]).parameters()
