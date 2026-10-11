import copy
import json
import random
import sys
import uuid
from pathlib import Path

import pytest

from dsp_decikit.services import live_ops
from dsp_decikit.services.live_ops import apply, diff, rebase_orders

TREE_EDITOR = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TREE_EDITOR / "e2e"), str(TREE_EDITOR / "merge-core-bench")]
from real_trees import TREES  # noqa: E402
from run_vectors import load_cases, run_case  # noqa: E402


def same(a, b):
    # Strict: `==` would take True for 1.
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# The vitest suite runs the same vectors through the WebAssembly build.
CASES = load_cases("op_level.json")


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_shared_case(case):
    run_case(live_ops, case)


def test_apply_result_does_not_share_values_with_the_ops():
    ops = [{"op": "add", "path": "/a", "value": {"x": [1]}}]
    doc, _ = apply({}, ops)
    doc["a"]["x"].append(2)
    assert ops[0]["value"] == {"x": [1]}


# Seeded random edits of the kinds the editors make.
def _new_id(rng):
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _thresholds(node):
    if isinstance(node, dict):
        value = node.get("threshold")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            yield node
        for v in node.values():
            yield from _thresholds(v)
    elif isinstance(node, list):
        for v in node:
            yield from _thresholds(v)


def _change_threshold(doc, rng):
    found = list(_thresholds(doc))
    if found:
        rng.choice(found)["threshold"] = round(rng.uniform(0, 1000), 2)


def _edit_output_row(doc, rng):
    data = (doc.get("output") or {}).get("data")
    if not isinstance(data, list):
        return
    rows = [r for r in data if isinstance(r, dict) and r]
    if rows and rng.random() < 0.7:
        row = rng.choice(rows)
        row[rng.choice(sorted(row))] = f"v{rng.randrange(1000)}"
    else:
        data.append(copy.deepcopy(rows[0]) if rows else {})


def _add_parameter(doc, rng):
    doc["parameters"] = {**(doc.get("parameters") or {}),
                         f"p{rng.randrange(10**6)}": {"type": "Float64", "default_value": rng.random()}}


def _rename_rule(doc, rng):
    rule = rng.choice(doc["rules"])
    rule["meta"] = {**(rule.get("meta") or {}), "name": f"R-{rng.randrange(1000)}"}


def _add_rule(doc, rng):
    rules = doc["rules"]
    new = copy.deepcopy(rng.choice(rules)) if rules else {"meta": {"name": "new"}, "rule": {"type": "leaf"}}
    if isinstance(new.get("rule"), dict) and "id" in new["rule"]:
        new["rule"]["id"] = _new_id(rng)
    rules.insert(rng.randrange(len(rules) + 1), new)


def _delete_rule(doc, rng):
    doc["rules"].pop(rng.randrange(len(doc["rules"])))


def _move_rule(doc, rng):
    rules = doc["rules"]
    rules.insert(rng.randrange(len(rules)), rules.pop(rng.randrange(len(rules))))


def _move_node(doc, rng):
    node = rng.choice(doc["nodes"])
    node["position"] = {"x": rng.uniform(0, 2000), "y": rng.uniform(0, 2000)}


def _add_edge(doc, rng):
    edges = doc.setdefault("edges", [])
    edge = {"source": rng.choice(doc["nodes"])["id"], "target": rng.choice(doc["nodes"])["id"],
            "data": {"sourceIndex": 0}}
    if rng.random() < 0.5:
        edge["id"] = _new_id(rng)
    edges.insert(rng.randrange(len(edges) + 1), edge)


def _edit(doc, rng):
    edits = [_change_threshold, _edit_output_row, _add_parameter]
    if isinstance(doc.get("rules"), list):
        edits.append(_add_rule)
        if doc["rules"]:
            edits += [_rename_rule, _delete_rule, _move_rule]
    if doc.get("nodes"):
        edits += [_move_node, _add_edge]
    rng.choice(edits)(doc, rng)


def _stamp_rule_ids(doc, rng):
    # As the flat-rule editor does at load.
    for rule in doc.get("rules") or []:
        if isinstance(rule.get("rule"), dict):
            rule["rule"].setdefault("id", _new_id(rng))


@pytest.mark.parametrize("stamped", [False, True], ids=["as-stored", "rule-ids-stamped"])
@pytest.mark.parametrize("tree", TREES, ids=[t.id for t in TREES])
def test_diff_then_apply_rebuilds_every_edit_of_a_real_tree(tree, stamped):
    rng = random.Random(f"{tree.id}-{stamped}")
    doc = copy.deepcopy(tree.document)
    if stamped:
        _stamp_rule_ids(doc, rng)
    for _ in range(25):
        edited = copy.deepcopy(doc)
        for _ in range(rng.randint(1, 3)):
            _edit(edited, rng)
        before = json.dumps(doc, sort_keys=True)
        ops = diff(doc, edited)
        result, dropped = apply(doc, ops)
        assert dropped == []
        assert same(result, edited)
        assert json.dumps(doc, sort_keys=True) == before
        assert diff(edited, copy.deepcopy(edited)) == []
        doc = edited


def _rules(*keys):
    return {"rules": [{"rule": {"id": k}} for k in keys]}


def _order(doc):
    return [r["rule"]["id"] for r in doc["rules"]]


def test_my_insert_merged_onto_their_reorder_keeps_their_order():
    mine = [{"op": "add", "path": "/rules/d", "value": {"rule": {"id": "d"}}},
            {"op": "order", "path": "/rules", "value": ["a", "d", "b", "c"]}]

    merged, _ = apply(_rules("c", "b", "a"), rebase_orders(mine, _rules("a", "b", "c"), _rules("c", "b", "a")))

    assert _order(merged) == ["c", "b", "a", "d"]


def test_my_reorder_is_not_rebased():
    mine = [{"op": "order", "path": "/rules", "value": ["c", "a", "b"]}]

    assert rebase_orders(mine, _rules("a", "b", "c"), _rules("a", "e", "b", "c")) == mine
