"""Writes flat-rule-errors.json and tree-errors.json from decikit's own validation. Rerun from ui/tree-editor/backend: PYTHONPATH=. SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv run --no-project --with-requirements requirements.txt --with-editable ../../.. python ../ui/src/utils/__tests__/fixtures/generate.py"""
import copy
import json
from pathlib import Path

from dsp_decikit.services.documents import validation_errors

HERE = Path(__file__).parent
STORED = json.loads((HERE / "../../../../../backend/tests/data/stored_documents.json").resolve().read_text())

LEAF = {"type": "leaf", "result_idx": 0}
UNARY = {"type": "unary", "feature": "x", "op": ">", "threshold": 1}


def flat(*rules):
    doc = copy.deepcopy(STORED["rich_flat"])
    doc["rules"] = [{"meta": {"name": f"R{i}"}, "rule": r} for i, r in enumerate(rules)]
    return doc


def unary_rule(condition, **extra):
    return {"type": "unary", "condition": condition, "then": LEAF, "otherwise": None, **extra}


def tree(nodes, edges):
    doc = copy.deepcopy(STORED["rich_v3"])
    doc["nodes"] = [{"id": i, "position": {"x": 0, "y": 0}, "data": d} for i, d in nodes.items()]
    doc["edges"] = [
        {"id": f"{s}->{t}", "source": s, "target": t, "data": {"sourceIndex": idx}} for s, t, idx in edges
    ]
    doc["subtrees"] = [{"id": next(iter(nodes)), "name": "Main"}]
    return doc


def unary_node(condition):
    return {"type": "unary", "condition": condition}


def leaf_node(idx=0):
    return {"type": "leaf", "result_idx": idx}


flat_docs = {
    "valid": flat(unary_rule(UNARY)),
    "ui_only_node_fields": flat(
        unary_rule(UNARY, editMode=False, isCollapsed=True, then={**LEAF, "editMode": False}),
        unary_rule({**UNARY, "threshold": {"param": "hi", "editMode": True}}),
    ),
    "empty_composite": flat(unary_rule(UNARY), {"type": "composite", "op": "and", "conditions": [], "then": LEAF}),
    "not_with_two_conditions": flat(
        {"type": "composite", "op": "not", "conditions": [UNARY, UNARY], "then": LEAF, "otherwise": None}
    ),
    "between_without_bounds": flat(unary_rule({"type": "unary", "feature": "x", "op": "between"})),
    "empty_param_name": flat(unary_rule({**UNARY, "threshold": {"param": ""}})),
    "bad_op": flat(unary_rule({**UNARY, "op": "~~"})),
    "result_idx_out_of_range": flat(unary_rule(UNARY, then={"type": "leaf", "result_idx": 9})),
    "cases_otherwise_out_of_range": flat(
        {
            "type": "cases", "feature": "x", "op": "ranges", "end_logic": "lower_inclusive", "strict": False,
            "conditions": [{"when": {"min": None, "max": 0.5}, "then": 0}],
            "branches": [LEAF, LEAF], "otherwise": 7,
        }
    ),
    "nested_composite_bad_threshold": flat(
        {
            "type": "composite", "op": "and",
            "conditions": [UNARY, {"type": "unary", "feature": "x", "op": ">", "threshold": {"param": "1bad"}}],
            "then": LEAF, "otherwise": None,
        }
    ),
    "cases_when_missing_patterns": flat(
        {
            "type": "cases", "feature": "s", "op": "string_match", "match_type": "exact", "case_sensitive": True,
            "trim_whitespace": False, "conditions": [{"when": {}, "then": 0}], "branches": [LEAF, LEAF], "otherwise": 1,
        }
    ),
    "parameter_not_an_object": {**flat(unary_rule(UNARY)), "parameters": {"p": 5}},
    "output_dtypes_not_a_list": {
        **flat(unary_rule(UNARY)),
        "output": {"dtypes": "A", "type_defs": {}, "data": [], "default": None},
    },
    "input_schema_entry_malformed": {**flat(unary_rule(UNARY)), "input_schema": [["x"]]},
}
cmp_node = unary_node(UNARY)
tree_docs = {
    "valid": tree({"a": cmp_node, "l0": leaf_node(0), "l1": leaf_node(1)}, [("a", "l0", [0]), ("a", "l1", [1])]),
    "empty_composite": tree(
        {"a": {"type": "composite", "op": "and", "conditions": []}, "l0": leaf_node()}, [("a", "l0", [0])]
    ),
    "not_with_two_conditions": tree(
        {"a": {"type": "composite", "op": "not", "conditions": [UNARY, UNARY]}, "l0": leaf_node(), "l1": leaf_node(1)},
        [("a", "l0", [0]), ("a", "l1", [1])],
    ),
    "between_without_bounds": tree(
        {"a": unary_node({"type": "unary", "feature": "x", "op": "between"}), "l0": leaf_node(), "l1": leaf_node(1)},
        [("a", "l0", [0]), ("a", "l1", [1])],
    ),
    "empty_param_name": tree(
        {"a": unary_node({**UNARY, "threshold": {"param": ""}}), "l0": leaf_node(), "l1": leaf_node(1)},
        [("a", "l0", [0]), ("a", "l1", [1])],
    ),
    "bad_op": tree(
        {"a": unary_node({**UNARY, "op": "~~"}), "l0": leaf_node(), "l1": leaf_node(1)},
        [("a", "l0", [0]), ("a", "l1", [1])],
    ),
    "result_idx_out_of_range": tree({"a": cmp_node, "l0": leaf_node(9), "l1": leaf_node(1)}, [("a", "l0", [0]), ("a", "l1", [1])]),
    "cycle": tree(
        {"a": cmp_node, "b": unary_node({**UNARY, "threshold": 2}), "l0": leaf_node()},
        [("a", "b", [0]), ("b", "a", [0]), ("a", "l0", [1]), ("b", "l0", [1])],
    ),
    "unknown_child_id": tree({"a": cmp_node, "l0": leaf_node()}, [("a", "l0", [0]), ("a", "ghost", [1])]),
}

for name, docs in (("flat-rule-errors", flat_docs), ("tree-errors", tree_docs)):
    out = {case: validation_errors(doc) for case, doc in docs.items()}
    (HERE / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(name, {k: len(v) for k, v in out.items()})
