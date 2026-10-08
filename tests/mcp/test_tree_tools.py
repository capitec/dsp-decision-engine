from __future__ import annotations

import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from decider import ConfigurableStep
from decider._tooling.mcp import Policy, build_server
from decider._tooling.mcp import _trees as t

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "ui/vscode/examples/bank/trees"
LIVE = ROOT / "tests/mcp/data/flat_rules.json"

SMALL = {"type": "tree", "name": "risk", "tree": {
    "nodes": [{"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "ratio", "threshold": 0.7}}},
              {"id": "high", "data": {"type": "leaf", "result_idx": 0}}],
    "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
    "output": {"data": [{"risk": 1}], "default": {"risk": 0}, "dtypes": [["risk", "Int64"]]}}}


def _write(tmp_path, doc, name="t.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(doc, indent=2) + "\n")
    return str(p)


def _copy(tmp_path, src: Path) -> str:
    return str(shutil.copy(src, tmp_path / src.name))


def _run(path, **cols):
    return ConfigurableStep.load(json.loads(Path(path).read_text())).run(pl.DataFrame(cols))


def _tree(doc):
    return doc["tree"]


def test_view_shows_conditions_params_branches_and_leaf_rows(tmp_path):
    view = t.tree_view(_copy(tmp_path, BANK / "bureau_grade.json"))
    assert view["problems"] == []
    out = view["outline"]
    assert "excellent: bureau_score >= {excellent_threshold=720.0}" in out
    assert "[0 true] excellent_leaf: leaf -> row 0 {\"grade\": 1}" in out
    assert "[1 false] good: bureau_score >= {good_threshold=660.0}" in out
    assert "[1 false] -> default row {\"grade\": 4}" in out
    fair = next(n for n in view["nodes"] if n["id"] == "fair")
    assert fair["parents"] == ["good"]
    assert fair["children"] == [{"branch": 0, "label": "true", "to": "fair_leaf"}, {"branch": 1, "label": "false", "to": None}]


def test_view_reads_every_condition_kind(tmp_path):
    doc = json.loads(json.dumps(SMALL))
    cases = [
        {"type": "unary", "condition": {"op": "between", "feature": "a", "min": 1, "max": {"param": "hi"}}},
        {"type": "unary", "condition": {"op": "isin", "feature": "c", "values": [1, 2]}},
        {"type": "unary", "condition": {"op": "string_match", "feature": "s", "patterns": ["ab"], "match_type": "contains", "case_sensitive": False}},
        {"type": "unary", "condition": {"op": "is_true", "feature": "flag"}},
        {"type": "unary", "condition": {"op": "<", "feature": {"type": "computed", "expression": "x - y"}, "threshold": 0}},
        {"type": "composite", "op": "or", "conditions": [{"op": "is_false", "feature": "f"}, {"op": "==", "feature": "n", "threshold": 3}]},
        {"type": "cases", "op": "ranges", "feature": "score", "conditions": [{"max": 30}, {"min": 30, "max": 60}, {"min": 60}]},
    ]
    for i, data in enumerate(cases):
        doc["tree"]["nodes"].append({"id": f"n{i}", "data": data})
    path = _write(tmp_path, doc)
    v = t.tree_view(path)
    assert v["problems"] == [] and v["unreachable"] == [f"n{i}" for i in range(len(cases))]
    summaries = {n["id"]: n["condition"] for n in t.tree_view(_write(tmp_path, _chain(doc, cases), "c.json"))["nodes"]}
    assert summaries["n0"] == "a between 1 and {hi}"
    assert summaries["n1"] == "c in [1, 2]"
    assert summaries["n2"] == "s contains [\"ab\"] ignoring case"
    assert summaries["n3"] == "flag is true"
    assert summaries["n4"] == "(x - y) < 0"
    assert summaries["n5"] == "(f is false OR n == 3)"
    assert summaries["n6"] == "score ranges cases"


def _chain(doc, cases):
    """Wire n0..n6 into one chain under root's branch 1, each on its own false branch; the last is a cases node."""
    doc = json.loads(json.dumps(doc))
    prev, branch = "root", 1
    for i in range(len(cases)):
        doc["tree"]["edges"].append({"source": prev, "target": f"n{i}", "data": {"sourceIndex": branch}})
        prev, branch = f"n{i}", 1
    return doc


def test_cases_branches_are_labelled_by_their_band(tmp_path):
    path = _write(tmp_path, {"type": "tree", "name": "band", "tree": {
        "nodes": [{"id": "b", "data": {"type": "cases", "op": "ranges", "feature": "score",
                                       "conditions": [{"max": 30}, {"min": 30, "max": 60}, {"min": 60}]}},
                  {"id": "mid", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "b", "target": "mid", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"band": "mid"}], "default": {"band": "other"}, "dtypes": [["band", "String"]]}}})
    out = t.tree_view(path)["outline"]
    assert "[0 [-inf, 30)] -> default row" in out
    assert "[1 [30, 60)] mid: leaf -> row 0" in out
    assert "[3 otherwise]" in out


def test_view_prints_a_shared_node_once_and_references_it_after(tmp_path):
    doc = json.loads(json.dumps(SMALL))
    doc["tree"]["nodes"].append({"id": "second", "data": {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 3}}})
    doc["tree"]["edges"] += [{"source": "root", "target": "second", "data": {"sourceIndex": 1}},
                             {"source": "second", "target": "high", "data": {"sourceIndex": 0}}]
    view = t.tree_view(_write(tmp_path, doc))
    assert view["outline"].count("high: leaf") == 1 and "high  (shown above)" in view["outline"]
    assert next(n for n in view["nodes"] if n["id"] == "high")["parents"] == ["root", "second"]


def test_view_of_a_subtree_starts_at_the_node(tmp_path):
    view = t.tree_view(_copy(tmp_path, BANK / "bureau_grade.json"), node="good")
    assert "excellent" not in view["outline"] and "good: bureau_score" in view["outline"]
    assert {n["id"] for n in view["nodes"]} == {"good", "good_leaf", "fair", "fair_leaf"}


def test_view_of_an_unknown_node_lists_the_ids(tmp_path):
    view = t.tree_view(_copy(tmp_path, BANK / "bureau_grade.json"), node="nope")
    assert "no node or rule 'nope'" in view["problems"][0] and "excellent" in view["problems"][0]


def test_view_of_flat_rules_lists_each_rule_with_its_mode(tmp_path):
    path = _copy(tmp_path, LIVE)
    view = t.tree_view(path, node="CF-0001")
    assert "mode all" in view["outline"] and "rule CF-0001:" in view["outline"]
    assert "model_score_available is true" in view["outline"] and "model_score > 570" in view["outline"]
    assert [r["name"] for r in t.tree_view(path)["rules"]][:2] == ["CF-0001", "CF-0002"]


def test_adding_a_node_and_an_output_row_changes_the_answer(tmp_path):
    path = _copy(tmp_path, BANK / "collections_risk.json")
    before = _run(path, missed_payments_12m=[0.0], judgements=[0.0])["collections_band"].to_list()
    assert before == [1]
    # judgements <= 0 reaches the empty branch 1 of has_judgement: give it a new answer
    rows = t.tree_set_output(path, {"collections_band": 9})
    assert rows["saved"] and rows["result_idx"] == 2
    added = t.tree_add_node(path, "has_judgement", 1, {"type": "unary", "condition": {"op": "<", "feature": "judgements", "threshold": 0.0}})
    assert added["saved"], added
    leaf = t.tree_add_node(path, added["id"], 0, {"type": "leaf", "result_idx": 2})
    assert leaf["saved"], leaf
    out = _run(path, missed_payments_12m=[0.0, 0.0], judgements=[-1.0, 0.0])
    assert out["collections_band"].to_list() == [9, 1]


def test_a_node_gets_a_readable_unique_id(tmp_path):
    path = _write(tmp_path, SMALL)
    data = {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 3}}
    assert t.tree_add_node(path, "root", 1, data)["id"] == "age_lt"
    assert t.tree_add_node(path, "age_lt", 0, {"type": "leaf", "result_idx": 0})["id"] == "leaf_0"
    assert t.tree_add_node(path, "age_lt", 1, {"type": "leaf", "result_idx": 0}, id="other")["id"] == "other"


def test_adding_to_an_occupied_branch_is_refused_and_says_how_to_replace(tmp_path):
    path = _write(tmp_path, SMALL)
    before = Path(path).read_text()
    r = t.tree_add_node(path, "root", 0, {"type": "leaf", "result_idx": -1})
    assert not r["saved"] and "already leads to 'high'" in r["problems"][0] and "tree_remove_node" in r["problems"][0]
    assert Path(path).read_text() == before


def test_adding_on_a_branch_the_node_does_not_have_lists_its_branches(tmp_path):
    path = _write(tmp_path, SMALL)
    r = t.tree_add_node(path, "root", 2, {"type": "leaf"})
    assert "0 (true), 1 (false)" in r["problems"][0]
    r = t.tree_add_node(path, "high", 0, {"type": "leaf"})
    assert "leaf has no branches" in r["problems"][0]


def test_a_leaf_pointing_at_a_missing_output_row_says_to_add_the_row_first(tmp_path):
    path = _write(tmp_path, SMALL)
    r = t.tree_add_node(path, "root", 1, {"type": "leaf", "result_idx": 4})
    assert not r["saved"] and "tree_set_output" in r["problems"][0]


def test_a_second_root_is_refused(tmp_path):
    r = t.tree_add_node(_write(tmp_path, SMALL), None, 0, {"type": "leaf"})
    assert not r["saved"] and "already has a root" in r["problems"][0]


def test_a_root_can_be_added_to_an_empty_tree(tmp_path):
    doc = json.loads(json.dumps(SMALL))
    doc["tree"]["nodes"], doc["tree"]["edges"] = [], []
    path = _write(tmp_path, doc)
    r = t.tree_add_node(path, None, 0, {"type": "leaf", "result_idx": 0}, id="only")
    assert r["saved"], r
    assert _run(path, ratio=[0.1])["risk"].to_list() == [1]


def test_an_edit_that_breaks_the_tree_leaves_the_file_unchanged(tmp_path):
    path = _write(tmp_path, SMALL)
    before = Path(path).read_text()
    r = t.tree_add_node(path, "root", 1, {"type": "unary", "condition": {"op": "~~", "feature": "x", "threshold": 1}})
    assert not r["saved"] and r["problems"]
    assert Path(path).read_text() == before
    r = t.tree_set_node(path, "root", {"type": "unary", "condition": {"op": ">", "feature": "ratio"}})
    assert not r["saved"] and Path(path).read_text() == before


def test_a_node_with_unknown_type_says_what_types_exist(tmp_path):
    r = t.tree_add_node(_write(tmp_path, SMALL), "root", 1, {"type": "mystery"})
    assert 'needs "type"' in r["problems"][0]


def test_setting_a_node_changes_what_it_tests(tmp_path):
    path = _write(tmp_path, SMALL)
    assert t.tree_set_node(path, "root", {"type": "unary", "condition": {"op": "<", "feature": "ratio", "threshold": 0.2}})["saved"]
    assert _run(path, ratio=[0.1, 0.5])["risk"].to_list() == [1, 0]


def test_setting_a_node_to_a_leaf_is_refused_while_it_has_children(tmp_path):
    path = _write(tmp_path, SMALL)
    r = t.tree_set_node(path, "root", {"type": "leaf", "result_idx": 0})
    assert not r["saved"] and "0 -> high" in r["problems"][0] and "tree_remove_node" in r["problems"][0]


def test_removing_a_node_drops_its_private_subtree(tmp_path):
    path = _copy(tmp_path, BANK / "bureau_grade.json")
    r = t.tree_remove_node(path, "good")
    assert r["saved"] and r["removed"] == ["fair", "fair_leaf", "good", "good_leaf"]
    assert r["now_default"] == [{"parent": "excellent", "branch": 1}]
    assert _run(path, bureau_score=[500.0, 800.0])["grade"].to_list() == [4, 1]


def test_removing_a_node_keeps_a_descendant_with_another_parent(tmp_path):
    doc = json.loads(json.dumps(SMALL))
    doc["tree"]["nodes"].append({"id": "second", "data": {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 3}}})
    doc["tree"]["edges"] += [{"source": "root", "target": "second", "data": {"sourceIndex": 1}},
                             {"source": "second", "target": "high", "data": {"sourceIndex": 0}}]
    path = _write(tmp_path, doc)
    r = t.tree_remove_node(path, "second")
    assert r["saved"] and r["removed"] == ["second"] and r["kept_shared"] == ["high"]
    assert "high: leaf" in t.tree_view(path)["outline"]


def test_removing_the_only_root_is_refused(tmp_path):
    r = t.tree_remove_node(_write(tmp_path, SMALL), "root")
    assert not r["saved"] and "empty" in r["problems"][0]


def test_output_rows_are_appended_replaced_and_default_set(tmp_path):
    path = _write(tmp_path, SMALL)
    assert t.tree_set_output(path, {"risk": 5})["result_idx"] == 1
    assert t.tree_set_output(path, {"risk": 6}, index=0)["result_idx"] == 0
    assert t.tree_set_output(path, {"risk": -1}, index=-1)["result_idx"] == -1
    out = json.loads(Path(path).read_text())["tree"]["output"]
    assert out["data"] == [{"risk": 6}, {"risk": 5}] and out["default"] == {"risk": -1}
    assert not t.tree_set_output(path, {"risk": 1}, index=7)["saved"]


def test_a_new_output_column_needs_a_dtype_and_the_error_suggests_one(tmp_path):
    path = _write(tmp_path, SMALL)
    r = t.tree_set_output(path, {"risk": 2, "reason": "late"})
    assert not r["saved"] and "dtypes={'reason': 'String'}" in r["problems"][0]
    r = t.tree_set_output(path, {"risk": 2, "reason": "late"}, dtypes={"reason": "String"})
    assert r["saved"] and r["added_dtypes"] == ["reason"]
    assert json.loads(Path(path).read_text())["tree"]["output"]["dtypes"] == [["risk", "Int64"], ["reason", "String"]]


RULES = {"type": "tree", "name": "rules", "tree": {
    "type": "prioritized_flat_rule", "mode": "first_match",
    "rules": [{"meta": {"name": "big"}, "rule": {"type": "unary", "condition": {"op": ">", "feature": "amount", "threshold": 100},
                                                "then": {"type": "leaf", "result_idx": 0}}},
              {"meta": {"name": "small"}, "rule": {"type": "unary", "condition": {"op": ">", "feature": "amount", "threshold": 10},
                                                  "then": {"type": "leaf", "result_idx": 1}}}],
    "output": {"data": [{"tier": "big"}, {"tier": "small"}, {"tier": "mid"}], "default": {"tier": "none"}, "dtypes": [["tier", "String"]]}}}

MID = {"meta": {"name": "mid"}, "rule": {"type": "unary", "condition": {"op": ">", "feature": "amount", "threshold": 50},
                                         "then": {"type": "leaf", "result_idx": 2}}}


def _names(path):
    return [r["meta"]["name"] for r in json.loads(Path(path).read_text())["tree"]["rules"]]


def test_adding_a_rule_at_a_position_keeps_the_order_and_changes_first_match(tmp_path):
    path = _write(tmp_path, RULES)
    assert _run(path, amount=[500.0, 60.0, 5.0])["tier"].to_list() == ["big", "small", "none"]
    r = t.tree_add_rule(path, MID, at=1)
    assert r["saved"] and r["index"] == 1 and "position matters" in r["note"]
    assert _names(path) == ["big", "mid", "small"]
    assert _run(path, amount=[500.0, 60.0, 20.0])["tier"].to_list() == ["big", "mid", "small"]


def test_adding_a_rule_appends_by_default(tmp_path):
    path = _write(tmp_path, RULES)
    assert t.tree_add_rule(path, MID)["saved"] and _names(path) == ["big", "small", "mid"]


def test_a_duplicate_rule_name_is_refused(tmp_path):
    path = _write(tmp_path, RULES)
    r = t.tree_add_rule(path, {**MID, "meta": {"name": "big"}})
    assert not r["saved"] and "tree_remove_rule" in r["problems"][0]
    assert not t.tree_add_rule(path, {"rule": MID["rule"]})["saved"]
    assert not t.tree_add_rule(path, MID, at=9)["saved"]


def test_removing_a_rule_keeps_the_others_in_order(tmp_path):
    path = _write(tmp_path, RULES)
    t.tree_add_rule(path, MID, at=1)
    r = t.tree_remove_rule(path, "mid")
    assert r["saved"] and r["removed_index"] == 1 and _names(path) == ["big", "small"]
    assert "no rule named 'zzz'" in t.tree_remove_rule(path, "zzz")["problems"][0]


def test_rules_of_a_real_rule_set_can_be_added_and_removed(tmp_path):
    path = _copy(tmp_path, LIVE)
    n = len(_names(path))
    rule = {"meta": {"name": "CF-NEW"}, "rule": {"type": "unary", "condition": {"op": ">", "feature": "amount", "threshold": {"param": "cf_new", "default": 1.0}},
                                                "then": {"type": "leaf", "result_idx": 0}}}
    r = t.tree_add_rule(path, rule, at=0)
    assert r["saved"] and _names(path)[0] == "CF-NEW" and r["rules"] == n + 1
    assert t.tree_remove_rule(path, "CF-NEW")["saved"] and len(_names(path)) == n


def test_node_tools_refuse_flat_rules_and_rule_tools_refuse_v3(tmp_path):
    flat, v3 = _write(tmp_path, RULES), _write(tmp_path, SMALL, "v3.json")
    assert "tree_add_rule" in t.tree_add_node(flat, "x", 0, {"type": "leaf"})["problems"][0]
    assert "tree_add_node" in t.tree_add_rule(v3, MID)["problems"][0]
    assert "one rule" in t.tree_add_rule(_write(tmp_path, {**RULES, "tree": {
        "type": "flat_rule", "rule": RULES["tree"]["rules"][0], "output": RULES["tree"]["output"]}}, "one.json"), MID)["problems"][0]


def test_the_tools_are_registered_on_the_server():
    import asyncio
    names = {x.name for x in asyncio.run(build_server(Policy()).list_tools())}
    assert {"tree_view", "tree_add_node", "tree_set_node", "tree_remove_node", "tree_set_output",
            "tree_add_rule", "tree_remove_rule"} <= names
