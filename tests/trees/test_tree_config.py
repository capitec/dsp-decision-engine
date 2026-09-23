"""TreeConfig as a step: loading, params, breakpoints, recompiles, and the walkers against each other."""
import json
import os
import random
import subprocess
import sys

import polars as pl
import pytest

from decider import engine, flow
from decider.engine import Engine
from decider.engine.compile import kernel
from decider.engine.debug import NodeVisited
from decider.steps import ConfigurableStep
from decider.steps.trees import LeafNode, TreeConfig, walker
from decider.testing import assert_equivalent

RISK = {
    "nodes": [
        {"id": "root", "position": {"x": 3.0, "y": 4.0}, "data": {"type": "unary", "condition": {
            "op": ">", "feature": "ratio", "threshold": {"param": "hi_thresh", "default": 0.7}}}},
        {"id": "high", "data": {"type": "leaf", "result_idx": 0}},
    ],
    "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
    "output": {"data": [{"band": "high", "pts": 10}], "default": {"band": "low", "pts": 0},
               "dtypes": [["band", "String"], ["pts", "Int64"]]},
}
FRAME = pl.DataFrame({"ratio": [0.5, 0.9, 3.0]})


def _risk(**tree) -> TreeConfig:
    return TreeConfig(name="risk_tree", tree={**RISK, **tree})


def test_a_tree_config_loads_by_alias_and_round_trips_with_node_positions():
    cfg = ConfigurableStep.load({"type": "tree", "name": "risk_tree", "tree": RISK})
    assert isinstance(cfg, TreeConfig)
    again = TreeConfig.load(cfg.model_dump_json())
    assert again == cfg
    assert json.loads(again.model_dump_json())["tree"]["nodes"][0]["position"] == {"x": 3.0, "y": 4.0}


def test_the_tree_is_one_row_node_whose_params_are_its_param_refs():
    node = engine.to_ir(_risk())
    assert (node.kind, node.origin.path, node.origin.source) == ("row", "risk_tree", "decider.steps.trees:TreeConfig")
    assert node.reference is not None and node.fn is walker.walk
    assert [i.name for i in node.inputs] == ["ratio"]
    assert [o.name for o in node.outputs] == ["band", "pts"]
    assert _risk().parameters()["risk_tree"] == {"hi_thresh": {"type": "float", "default": 0.7}}


def test_outputs_follow_the_reached_leaf_and_retune_through_params(run):
    out = run(_risk(), FRAME)
    assert out["band"].to_list() == ["low", "high", "high"]
    assert out["pts"].to_list() == [0, 10, 10]
    assert run(_risk(), FRAME, {"risk_tree": {"hi_thresh": 2.0}})["band"].to_list() == ["low", "low", "high"]


def test_nodes_map_each_locator_to_its_node():
    assert set(_risk().nodes) == {"root", "high"}
    assert isinstance(_risk().nodes["high"].data, LeafNode)


def test_a_node_breakpoint_pauses_after_the_tree_and_counts_rows_per_node():
    s = Engine().bind(_risk()).session(FRAME)
    s.break_at("risk_tree#high")
    at = s.resume()
    assert (at.origin.path, at.when) == ("risk_tree", "after")
    visits = {e.origin.locator: e.rows for e in s.events if isinstance(e, NodeVisited)}
    assert visits == {"root": 3, "high": 2}


def test_retuning_a_param_or_editing_a_literal_never_recompiles():
    df = pl.DataFrame({"ratio": [0.5, 0.9]})
    Engine().bind(_risk(), mode="fused").run(df)
    before = (len(walker.walk.signatures), len(kernel._KERNELS))
    for thresh in (0.1, 0.6, 5.0):
        Engine().bind(_risk(), mode="fused").run(df, params={"risk_tree": {"hi_thresh": thresh}})
    edited = json.loads(json.dumps(RISK).replace('"pts": 10', '"pts": 99'))
    edited["nodes"][0]["data"]["condition"]["threshold"] = {"param": "hi_thresh", "default": 0.2}
    assert Engine().bind(_risk(**edited), mode="fused").run(df)["pts"].to_list() == [99, 99]
    assert (len(walker.walk.signatures), len(kernel._KERNELS)) == before


def test_v0_to_v2_documents_are_refused_by_name():
    with pytest.raises(ValueError, match="v1 tree documents are deprecated"):
        TreeConfig(name="t", tree={"formatVersion": 1, "nodes": []})


def test_feature_types_accept_polars_spellings_and_refuse_unknown_ones():
    assert TreeConfig(name="t", tree=RISK, feature_types={"ratio": "Float64"}).feature_types == {"ratio": "float"}
    with pytest.raises(ValueError, match="not a type a tree understands"):
        TreeConfig(name="t", tree=RISK, feature_types={"ratio": "decimal"})


# A corpus of random trees: flat rules, the same trees as v3 documents, and
# prioritized documents in both modes, over every node and condition kind.

FLOATS = [0.0, 0.25, 0.5, 0.75, 1.0]
EXPRESSIONS = ["f0 - f1", "max(f2, f3) * 2", "f0 / (f1 + 1)", "abs(f2 - 0.5) < 0.2 and f3 >= 0.5"]


def _threshold(rng, feature, name):
    value = rng.randrange(6) if feature.startswith("i") else rng.choice(FLOATS)
    if rng.random() < 0.3:
        return {"param": name, "default": value}
    return value


def _feature(rng):
    if rng.random() < 0.2:
        return {"type": "computed", "expression": rng.choice(EXPRESSIONS)}
    return rng.choice(["f0", "f1", "f2", "f3", "i0", "i1"])


def _condition(rng, depth, names):
    kind = rng.randrange(6 if depth < 2 else 5)
    f = _feature(rng)
    p = f"p{next(names)}" if isinstance(f, str) and f.startswith("i") else f"q{next(names)}"
    if kind == 0:
        return {"op": rng.choice(["<", "<=", "==", ">", ">=", "!="]), "feature": f, "threshold": _threshold(rng, str(f), p)}
    if kind == 1:
        return {"op": "between", "feature": f, "min": _threshold(rng, str(f), p), "max": 1.0 if isinstance(f, dict) or not f.startswith("i") else 5}
    if kind == 2 and isinstance(f, str):
        return {"op": "isin", "feature": f, "values": [_threshold(rng, f, p + "_a"), rng.choice(FLOATS) if f.startswith("f") else 3]}
    if kind == 3:
        return {"op": rng.choice(["is_true", "is_false"]), "feature": "b0"}
    if kind == 4:
        return {"op": ">", "feature": {"type": "computed", "expression": rng.choice(EXPRESSIONS)}, "threshold": rng.choice(FLOATS)}
    op = rng.choice(["and", "or", "not"])
    return {"type": "composite", "op": op,
            "conditions": [_condition(rng, depth + 1, names) for _ in range(1 if op == "not" else 2)]}


def _rule(rng, depth, names, leaves):
    if depth == 3 or rng.random() < 0.15:
        return None if rng.random() < 0.2 else {"type": "leaf", "result_idx": rng.randrange(-1, leaves)}
    shape = rng.randrange(4)
    kids = lambda n: [_rule(rng, depth + 1, names, leaves) for _ in range(n)]  # noqa: E731
    if shape == 0:
        cond = _condition(rng, 0, names)
        then, other = kids(2)
        if cond.get("type") == "composite":
            return {"type": "composite", "op": cond["op"], "conditions": cond["conditions"], "then": then, "otherwise": other}
        return {"type": "unary", "condition": cond, "then": then, "otherwise": other}
    branches = [b or {"type": "leaf", "result_idx": -1} for b in kids(3)]
    if shape == 1:
        return {"type": "cases", "op": "ranges", "feature": rng.choice(["f0", "f2"]), "strict": False,
                "end_logic": rng.choice(["lower_inclusive", "upper_inclusive"]),
                "conditions": [{"when": {"max": 0.5}, "then": 0}, {"when": {"min": 0.25, "max": 0.75}, "then": 1}],
                "otherwise": 2, "branches": branches}
    return {"type": "cases", "op": "isin", "feature": "i0",
            "conditions": [{"when": {"values": [0, 1]}, "then": 1}, {"when": {"values": [{"param": f"k{next(names)}", "default": 2}]}, "then": 0}],
            "otherwise": 2, "branches": branches}


def _to_v3(rule) -> tuple[list, list]:
    nodes, edges = [], []

    def add(r):
        nid = f"n{len(nodes)}"
        data = {k: v for k, v in r.items() if k not in ("then", "otherwise", "branches")}
        nodes.append({"id": nid, "data": data})
        if r["type"] == "cases":
            data["conditions"] = [c["when"] for c in r["conditions"]]
            children = [r["branches"][c["then"]] for c in r["conditions"]] + [r["branches"][r["otherwise"]]]
        else:
            children = [] if r["type"] == "leaf" else [r["then"], r["otherwise"]]
        for i, child in enumerate(children):
            if child is not None:
                edges.append({"source": nid, "target": add(child), "data": {"sourceIndex": [i]}})
        return nid

    add(rule)
    return nodes, edges


def _output(rng):
    labels = ["a", "b", None, "c"]
    rows = [{"label": rng.choice(labels[:3] + ["d"]), "pts": float(i), "band": i, "flag": i % 2 == 0} for i in range(4)]
    default = {"label": rng.choice(labels), "pts": None if rng.random() < 0.5 else -1.0, "band": -1, "flag": False}
    return {"data": rows, "default": default,
            "dtypes": [["label", "String"], ["pts", "Float64"], ["band", "Int64"], ["flag", "Boolean"]]}


def _frame(rng, n=120):
    return pl.DataFrame({
        **{f"f{k}": [rng.choice(FLOATS) for _ in range(n)] for k in range(4)},
        **{f"i{k}": [rng.randrange(6) for _ in range(n)] for k in range(2)},
        "b0": [rng.random() < 0.5 for _ in range(n)],
    })


def _config(name, doc, reads, path=None):
    types = {f: "int" for f in ("i0", "i1") if f in reads}
    return TreeConfig(name=name, tree=doc, feature_types=types, path_output=path)


def _reads(doc):
    text = json.dumps(doc)
    return {f for f in ("i0", "i1") if f'"{f}"' in text}


@pytest.mark.parametrize("seed", range(8))
def test_random_trees_agree_across_walkers_modes_and_formats(seed, run):
    rng = random.Random(seed)
    names = iter(range(10_000))
    df = _frame(rng)
    root = _rule(rng, 0, names, 4) or {"type": "leaf", "result_idx": 0}
    output = _output(rng)
    flat = {"type": "flat_rule", "rule": {"rule": root}, "output": output}
    nodes, edges = _to_v3(root)
    v3 = {"nodes": nodes, "edges": edges, "output": output}
    a = run(_config("t", flat, _reads(flat)), df)
    assert a.equals(run(_config("t", v3, _reads(flat)), df))
    refs = json.dumps(flat).split('"param": "')[1:]
    if refs:
        run(_config("t", flat, _reads(flat)), df, {"t": {r.split('"')[0]: 1 for r in refs}})

    rules = [{"meta": {"name": f"r{k}"} if k else {}, "rule": _rule(rng, 1, names, 4) or {"type": "leaf", "result_idx": -1}}
             for k in range(3)]
    for mode in ("first_match", "all"):
        doc = {"type": "prioritized_flat_rule", "mode": mode, "rules": rules, "output": output}
        out = run(_config("p", doc, _reads(doc), path="leaf"), df)
        if mode == "all":
            assert [c for c in out.columns if "." in c] == [
                f"{r}.{c}" for r in ("rule_0", "r1", "r2") for c in ("label", "pts", "band", "flag", "leaf")]


def _two_leaves_one_row(**config) -> TreeConfig:
    # Two leaves share output row 0, so only the path tells them apart; `x > 9` leads nowhere (the default).
    return TreeConfig(name="risk_tree", path_output="risk_leaf", **config, tree={
        "nodes": [
            {"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "x", "threshold": 5.0}}},
            {"id": "mid", "data": {"type": "unary", "condition": {"op": ">", "feature": "x", "threshold": 9.0}}},
            {"id": "low", "data": {"type": "leaf", "result_idx": 0}},
            {"id": "also_low", "data": {"type": "leaf", "result_idx": 0}},
        ],
        "edges": [{"source": "root", "target": "mid", "data": {"sourceIndex": 0}},
                  {"source": "root", "target": "low", "data": {"sourceIndex": 1}},
                  {"source": "mid", "target": "also_low", "data": {"sourceIndex": 1}}],
        "output": {"data": [{"pts": 1}], "default": {"pts": 0}, "dtypes": [["pts", "Int64"]]},
    })


def test_a_tree_reports_the_leaf_that_answered_in_every_mode():
    out = assert_equivalent(_two_leaves_one_row(), pl.DataFrame({"x": [1.0, 7.0, 10.0, None]}))
    assert out["risk_leaf"].to_list() == ["low", "also_low", None, "low"]   # a null x takes the otherwise branch
    assert out["pts"].to_list() == [1, 1, 0, 1]
    assert out.schema["risk_leaf"] == pl.String
    assert Engine().bind(_two_leaves_one_row(), mode="fused").score({"x": 7.0})["risk_leaf"] == "also_low"


def test_the_path_column_survives_an_explicit_emit_and_a_drop():
    pipeline = flow(_two_leaves_one_row(), name="scoring").emit("risk_leaf").drop("pts")
    out = assert_equivalent(pipeline, pl.DataFrame({"x": [1.0, 7.0]}))
    assert out.columns == ["x", "risk_leaf"] and out["risk_leaf"].to_list() == ["low", "also_low"]


def test_the_path_column_names_each_rules_leaf_in_all_mode_and_in_python_trees():
    def rule(name, pattern):
        return {"meta": {"name": name}, "rule": {"type": "unary", "condition": {
            "op": "string_match", "feature": "s", "patterns": [pattern], "match_type": "regex"},
            "then": {"id": f"{name}_yes", "type": "leaf", "result_idx": 0}}}

    tree = TreeConfig(name="t", path_output="leaf", tree={
        "type": "prioritized_flat_rule", "mode": "all", "rules": [rule("a", "^x"), rule("b", "y$")],
        "output": {"data": [{"hit": 1}], "default": {"hit": 0}, "dtypes": [["hit", "Int64"]]}})
    out = assert_equivalent(tree, pl.DataFrame({"s": ["xy", "x", "zz"]}))
    assert out["a.leaf"].to_list() == ["a_yes", "a_yes", None]
    assert out["b.leaf"].to_list() == ["b_yes", None, None]


def test_a_path_output_named_like_an_output_column_is_refused():
    with pytest.raises(ValueError, match="path_output 'pts'"):
        engine.to_ir(_two_leaves_one_row().model_copy(update={"path_output": "pts"}))


_CHILD = """
import contextlib, io, json
import polars as pl
from decider.engine import Engine
from decider.steps.trees import TreeConfig

tree = TreeConfig(name="t", tree=json.loads(%r))
log = io.StringIO()
with contextlib.redirect_stdout(log):  # NUMBA_DEBUG_CACHE prints to stdout
    out = Engine().bind(tree, mode="stepped").run(pl.DataFrame({"ratio": [0.5, 0.9], "s": ["ab", "cd"]}))
lines = [l for l in log.getvalue().splitlines() if "walk" in l]
print(json.dumps({"pts": out["pts"].to_list(), "saved": sum("data saved" in l for l in lines),
                  "loaded": sum("data loaded" in l for l in lines)}))
"""


def test_the_tree_walker_is_a_disk_cache_hit_in_a_fresh_process(tmp_path):
    tree = {**RISK, "nodes": [
        {"id": "root", "data": {"type": "composite", "op": "and", "conditions": [
            {"op": ">", "feature": "ratio", "threshold": {"param": "hi_thresh", "default": 0.7}},
            {"op": "string_match", "feature": "s", "patterns": ["c"], "match_type": "starts_with"}]}},
        {"id": "high", "data": {"type": "leaf", "result_idx": 0}}]}
    (tmp_path / "child.py").write_text(_CHILD % json.dumps(tree))  # once: numba's index is keyed by mtime
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba_cache"), NUMBA_DEBUG_CACHE="1")

    def fresh():
        proc = subprocess.run([sys.executable, "child.py"], cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout.splitlines()[-1])

    cold, warm = fresh(), fresh()
    assert cold["pts"] == warm["pts"] == [0, 10]
    assert (cold["saved"], cold["loaded"]) == (1, 0), cold
    assert (warm["saved"], warm["loaded"]) == (0, 1), warm


def test_a_string_feature_given_a_number_is_a_type_error_naming_it():
    cfg = TreeConfig(name="t", tree={
        "type": "flat_rule",
        "rule": {"rule": {"type": "unary", "condition": {"op": "string_match", "feature": "s", "patterns": ["a"]},
                          "then": {"type": "leaf", "result_idx": 0}}},
        "output": {"data": [{"r": 1}], "default": {"r": 0}, "dtypes": [["r", "Int64"]]},
    })
    for mode in ("interpreted", "stepped", "fused"):
        with pytest.raises(TypeError, match="'s' is a string input"):
            Engine().bind(cfg, mode=mode).score({"s": 1})
