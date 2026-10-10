"""`TreeConfig.path_columns`: columns computed per route at build time, every branch of a node told apart."""
import json
import tempfile
import types
from pathlib import Path

import polars as pl
import pytest

from decikit import load_flow
from decikit.exceptions import WiringError
from decikit.steps.trees import TreeConfig

# Branch 0 of root leads nowhere; branches 1 and 2 share one edge into `hi`, whose otherwise leads nowhere.
DOC = {
    "nodes": [
        {"id": "root", "data": {"type": "cases", "op": "ranges", "feature": "score", "strict": False,
                                "conditions": [{"max": 50.0}, {"min": 50.0, "max": 80.0}]}},
        {"id": "hi", "data": {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 30.0}}},
        {"id": "young", "data": {"type": "leaf", "result_idx": 0}},
    ],
    "edges": [{"source": "root", "target": "hi", "data": {"sourceIndex": [1, 2]}},
              {"source": "hi", "target": "young", "data": {"sourceIndex": 0}}],
    "output": {"data": [{"band": "young"}], "default": {"band": "other"}, "dtypes": [["band", "String"]]},
}
DF = pl.DataFrame({"score": [10.0, 60.0, 90.0, 90.0, None], "age": [20.0, 20.0, 20.0, 40.0, 20.0]})


def route_text(route, tree):
    return {"route": "|".join(f"{v.node_id}:{v.branch}" for v in route.visits), "depth": len(route.visits),
            "hit": route.result_idx >= 0}


def test_every_branch_taken_is_its_own_route(run):
    out = run(TreeConfig(name="t", tree=DOC, path_columns=route_text, path_output="leaf"), DF)
    assert out.select("band", "route", "depth", "hit").rows() == [
        ("other", "root:0", 1, False),
        ("young", "root:1|hi:0", 2, True),
        ("young", "root:2|hi:0", 2, True),
        ("other", "root:2|hi:1", 2, False),
        ("young", "root:2|hi:0", 2, True),
    ]
    assert out["leaf"].to_list() == ["root.0", "young", "young~2", "hi~2.1", "young~2"]


def test_a_visit_says_where_its_branch_leads():
    seen = []
    TreeConfig(name="t", tree=DOC, path_columns=lambda r, tree: seen.append(r) or {"n": 1}).nodes
    assert {tuple((v.node_id, v.branch, v.child) for v in r.visits) for r in seen} == {
        (("root", 0, None),), (("root", 1, "hi"), ("hi", 0, "young")), (("root", 1, "hi"), ("hi", 1, None)),
        (("root", 2, "hi"), ("hi", 0, "young")), (("root", 2, "hi"), ("hi", 1, None))}


def test_path_columns_never_reach_the_document():
    tree = TreeConfig(name="t", tree=DOC, path_columns=route_text)
    assert "path_columns" not in tree.model_dump()
    assert "path_columns" not in TreeConfig.model_json_schema()["properties"]


@pytest.mark.parametrize("fn, message", [
    (lambda r, t: {"a": 1} if r.result_idx < 0 else {"b": 1}, "same columns"),
    (lambda r, t: {"band": "x"}, "already output columns"),
    (lambda r, t: {"mixed": 1 if r.result_idx < 0 else "one"}, "one type"),
])
def test_bad_path_columns_are_refused(fn, message):
    with pytest.raises(ValueError, match=message):
        TreeConfig(name="t", tree=DOC, path_columns=fn).nodes


def test_a_first_match_tree_of_several_rules_is_refused():
    leaf = {"type": "unary", "condition": {"op": ">", "feature": "x", "threshold": 1.0},
            "then": {"type": "leaf", "result_idx": 0}}
    doc = {"type": "prioritized_flat_rule", "rules": [{"rule": leaf}, {"rule": leaf}],
           "output": {"data": [{"y": 1}], "dtypes": [["y", "Int64"]]}}
    with pytest.raises(ValueError, match="single rule"):
        TreeConfig(name="t", tree=doc, path_columns=route_text).nodes


def _project(step_node):
    project = Path(tempfile.mkdtemp())
    (project / "flow.json").write_text(json.dumps({"flow": "f", "steps": [step_node]}))
    components = types.ModuleType("components")
    # A component counts only where it is defined, so define it inside the module.
    exec("def route_text(route, tree):\n    return {'route': '|'.join(f'{v.node_id}:{v.branch}' for v in route.visits)}",
         components.__dict__)
    return project / "flow.json", components


def test_a_flow_file_names_the_path_columns_function(run):
    path, components = _project({"step": "risk", "path_columns": "route_text"})
    pipeline = load_flow(path, components=components, documents={"risk": {"type": "tree", "name": "risk", "tree": DOC}})
    assert run(pipeline, DF)["route"].to_list()[:2] == ["root:0", "root:1|hi:0"]


def test_a_flow_file_path_columns_must_name_a_function():
    path, components = _project({"step": "risk", "path_columns": "route_txt"})
    with pytest.raises(WiringError, match="Did you mean 'route_text'"):
        load_flow(path, components=components, documents={"risk": {"type": "tree", "name": "risk", "tree": DOC}})
