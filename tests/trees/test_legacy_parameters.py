"""Param references and computed features give decider_old's answers."""
import polars as pl

from decider.steps.trees import (
    FlatRuleDocument,
    LeafRule,
    ParameterInfo,
    RuleRoot,
    TreeConfig,
    TreeOutput,
    UnaryBetween,
    UnaryLessThan,
    UnaryRule,
)

THRESH = {"thresh": ParameterInfo(type="Float64", default_value=50.0)}


def _tree(condition, labels, default="default", other=1, **doc):
    out = TreeOutput(data=[{"r": lbl} for lbl in labels], default={"r": default}, dtypes=[("r", "String")])
    rule = UnaryRule(condition=condition, then=LeafRule(result_idx=0), otherwise=LeafRule(result_idx=other))
    return TreeConfig(name="t", tree=FlatRuleDocument(rule=RuleRoot(rule=rule), output=out, **doc))


def _below(**doc):
    return _tree(UnaryLessThan(feature="score", threshold={"key": "thresh"}), ("low", "high"), **doc)


def test_a_param_ref_uses_the_parameters_block_default(run):
    assert run(_below(parameters=THRESH), pl.DataFrame({"score": [30.0, 70.0]}))["r"].to_list() == ["low", "high"]


def test_the_params_document_overrides_the_default(run):
    out = run(_below(parameters=THRESH), pl.DataFrame({"score": [30.0, 70.0]}), {"t": {"thresh": 80.0}})
    assert out["r"].to_list() == ["low", "low"]


def test_each_run_takes_its_own_params(run):
    df = pl.DataFrame({"score": [30.0, 70.0]})
    assert run(_below(parameters=THRESH), df, {"t": {"thresh": 20.0}})["r"].to_list() == ["high", "high"]
    assert run(_below(parameters=THRESH), df, {"t": {"thresh": 80.0}})["r"].to_list() == ["low", "low"]


def test_both_between_bounds_can_be_params(run):
    tree = _tree(UnaryBetween(feature="score", min={"key": "lo"}, max={"key": "hi"}), ("in_range",), "out", -1,
                 parameters={"lo": ParameterInfo(type="Float64", default_value=20.0),
                             "hi": ParameterInfo(type="Float64", default_value=80.0)})
    assert run(tree, pl.DataFrame({"score": [10.0, 50.0, 90.0]}))["r"].to_list() == ["out", "in_range", "out"]


def test_a_computed_feature_combines_two_columns(run):
    tree = _tree(UnaryLessThan(feature={"type": "computed", "expression": "amount * quantity"}, threshold=100.0),
                 ("small", "large"))
    df = pl.DataFrame({"amount": [5.0, 20.0], "quantity": [10.0, 10.0]})
    assert run(tree, df)["r"].to_list() == ["small", "large"]
