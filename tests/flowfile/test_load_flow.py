"""A flow file (YAML or JSON) builds the same `Step` tree as the equivalent hand-written Python."""
import json

import polars as pl
import yaml

from decikit import branch, flow, loop
from decikit._engine import step_map
from decikit.steps.flowfile import load_flow
from decikit.steps._configurable import ConfigurableStep
from decikit.steps.trees import expr_condition
from decikit.lab.testing import assert_equivalent
from tests.flowfile import components as comp

FRAME = pl.DataFrame({
    "score": [60.0, 10.0], "limit": [100.0, 100.0], "balance": [90.0, 0.0], "is_adult": [True, False],
})


def _write(tmp_path, name, doc):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc) if name.endswith((".yaml", ".yml")) else json.dumps(doc))
    return path


def _assert_same_shape(loaded, hand, frame):
    assert_equivalent(loaded, frame)
    assert loaded.parameters().defaults() == hand.parameters().defaults()
    assert set(step_map(loaded)) == set(step_map(hand))


CREDIT_DOC = {
    "flow": "credit",
    "components": "tests.flowfile.components",
    "steps": [
        {"branch": "by_eligibility", "when": "eligible", "modifies": ["limit"],
         "then": [{"step": "full_limit"}], "else": [{"step": "no_limit"}]},
        {"branch": "by_cap", "when": {"expr": "limit > 4000"}, "modifies": ["limit"], "then": [{"step": "trim"}]},
        {"loop": "payoff", "while": {"expr": "balance > 0"}, "carries": ["balance"], "max": 10,
         "steps": [{"step": "pay"}]},
    ],
}


def _credit_hand_written():
    by_eligibility = branch(comp.eligible, comp.full_limit, comp.no_limit, modifies=["limit"], name="by_eligibility")
    by_cap = branch(expr_condition("limit > 4000", name="by_cap_when"), comp.trim, modifies=["limit"], name="by_cap")
    payoff = loop(expr_condition("balance > 0", name="payoff_while"), comp.pay, carries=["balance"],
                  max_iterations=10, name="payoff")
    return flow(by_eligibility, by_cap, payoff, name="credit")


def test_yaml_flow_matches_the_equivalent_python_flow(tmp_path):
    loaded = load_flow(_write(tmp_path, "flow.yaml", CREDIT_DOC))
    _assert_same_shape(loaded, _credit_hand_written(), FRAME)


def test_json_flow_matches_the_equivalent_python_flow(tmp_path):
    loaded = load_flow(_write(tmp_path, "flow.json", CREDIT_DOC))
    _assert_same_shape(loaded, _credit_hand_written(), FRAME)


COND_DOC = {
    "flow": "cond",
    "components": "tests.flowfile.components",
    "steps": [
        {"branch": "adult_branch", "when": "is_adult", "modifies": ["limit"], "then": [{"step": "full_limit"}]},
        {"branch": "cap_branch", "when": "over_cap", "modifies": ["limit"],
         "then": [{"step": "trim"}], "else": [{"step": "no_limit"}]},
    ],
}


def _cond_hand_written():
    adult_branch = branch("is_adult", comp.full_limit, modifies=["limit"], name="adult_branch")
    cap_branch = branch(comp.over_cap, comp.trim, comp.no_limit, modifies=["limit"], name="cap_branch")
    return flow(adult_branch, cap_branch, name="cond")


def test_bare_column_and_component_step_conditions(tmp_path):
    loaded = load_flow(_write(tmp_path, "flow.yaml", COND_DOC))
    _assert_same_shape(loaded, _cond_hand_written(), FRAME)


AS_DOC = {
    "flow": "as_reuse",
    "components": "tests.flowfile.components",
    "steps": [{"step": "full_limit"}, {"step": "full_limit", "as": "full_limit_again"}],
}


def test_as_runs_the_same_step_twice(tmp_path):
    loaded = load_flow(_write(tmp_path, "flow.yaml", AS_DOC))
    hand = flow(comp.full_limit, comp.full_limit.named("full_limit_again"), name="as_reuse")
    _assert_same_shape(loaded, hand, FRAME)


SCORECARD_DOC = {"type": "scorecard", "name": "card", "variables": [{
    "type": "scored", "variable_name": "age", "default": {"value": 0},
    "bins": [{"value": 5, "upper_bound": {"param": "young", "default": 25}},
             {"value": 10, "lower_bound": {"param": "young", "default": 25}}]}]}


def test_step_resolves_a_document_by_name(tmp_path):
    loaded = load_flow(_write(tmp_path, "flow.yaml", {"flow": "scoring", "steps": [{"step": "card"}]}),
                        documents={"card": SCORECARD_DOC})
    hand = flow(ConfigurableStep.load(SCORECARD_DOC), name="scoring")
    _assert_same_shape(loaded, hand, pl.DataFrame({"age": [20.0, 40.0]}))


def test_flow_reference_from_a_root_at_the_project_root(tmp_path):
    _write(tmp_path, "flows/sub.yaml",
           {"flow": "sub", "components": "tests.flowfile.components", "steps": [{"step": "full_limit"}]})
    path = _write(tmp_path, "flow.yaml",
                  {"flow": "credit", "components": "tests.flowfile.components", "steps": [{"flow": "sub"}]})
    loaded = load_flow(path)
    hand = flow(flow(comp.full_limit, name="sub"), name="credit")
    _assert_same_shape(loaded, hand, FRAME)


def test_flow_reference_from_a_root_inside_the_flows_directory(tmp_path):
    _write(tmp_path, "flows/sub.yaml",
           {"flow": "sub", "components": "tests.flowfile.components", "steps": [{"step": "full_limit"}]})
    path = _write(tmp_path, "flows/credit.yaml",
                  {"flow": "credit", "components": "tests.flowfile.components", "steps": [{"flow": "sub"}]})
    loaded = load_flow(path)
    hand = flow(flow(comp.full_limit, name="sub"), name="credit")
    _assert_same_shape(loaded, hand, FRAME)


def test_a_referenced_file_may_name_the_components_passed_as_an_argument(tmp_path):
    _write(tmp_path, "flows/sub.yaml",
           {"flow": "sub", "components": "tests.flowfile.components", "steps": [{"step": "full_limit"}]})
    path = _write(tmp_path, "flow.yaml", {"flow": "credit", "steps": [{"flow": "sub"}]})
    loaded = load_flow(path, components=comp)
    _assert_same_shape(loaded, flow(flow(comp.full_limit, name="sub"), name="credit"), FRAME)


def test_suppress_checks_keys_reach_the_origins(tmp_path):
    from decikit._engine import to_ir

    doc = {
        "flow": "credit", "components": "tests.flowfile.components", "suppress_checks": ["DCD101"],
        "steps": [
            {"step": "eligible", "suppress_checks": ["int64_overflow"]},
            {"flow": "inner", "suppress_checks": ["shadowed_write"], "steps": [{"step": "full_limit"}]},
            {"branch": "b", "when": "eligible", "modifies": ["limit"], "suppress_checks": ["a"],
             "then": [{"step": "full_limit"}]},
            {"loop": "lp", "while": {"expr": "balance > 0"}, "carries": ["balance"], "max": 3,
             "suppress_checks": ["b"], "steps": [{"step": "pay"}]},
        ],
    }
    root = to_ir(load_flow(_write(tmp_path, "credit.yaml", doc)))
    assert root.origin.suppress_checks == ("DCD101",)
    by_path = {n.origin.path: n.origin.suppress_checks for n in _iter(root)}
    assert by_path["credit/eligible"] == ("int64_overflow",)
    assert by_path["credit/inner"] == ("shadowed_write",)
    assert by_path["credit/b"] == ("a",)
    assert by_path["credit/lp"] == ("b",)
    assert by_path["credit/inner/full_limit"] == ()


def _iter(root):
    from decikit._engine.ir.nodes import iter_nodes

    return iter_nodes(root)
