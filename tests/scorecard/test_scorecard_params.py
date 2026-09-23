"""Scorecard edges and points are params: retuning never rebuilds IR, and the compiled kernels agree."""
from __future__ import annotations

import numpy as np
import polars as pl

from decider import engine
from decider.engine import Engine
from decider.engine.compile import Fallback, Kernel, compile_plan, default_bundle
from decider.engine.wiring import resolve
from decider.steps import ParamRef
from decider.steps.scorecard import (
    AdjustedVariable,
    BoundBin,
    ConstantScore,
    DefaultBin,
    ScorecardConfig,
    ScoredVariable,
    ValuesBin,
)

YOUNG = ParamRef(param="young", default=25.0)


def card() -> ScorecardConfig:
    return ScorecardConfig(name="card", variables=[
        AdjustedVariable(
            variable=ScoredVariable(
                variable_name="age",
                bins=[
                    ValuesBin(value=99.0, items=[1.0]),
                    BoundBin(value=5.0, upper_bound=YOUNG),
                    BoundBin(value=ParamRef(param="mid", default=10.0), lower_bound=YOUNG, upper_bound=60.0),
                    BoundBin(value=15.0, lower_bound=60.0),
                ],
                default=DefaultBin(value=0.0),
            ),
            scale=ParamRef(param="weight", default=2.0),
        ),
        ScoredVariable(variable_name="income", default=DefaultBin(value=-1.0),
                       bins=[BoundBin(value=1.0, upper_bound=10.0), BoundBin(value=2.0, lower_bound=10.0)]),
        ConstantScore(score=100.0),
    ])


AGE = [1.0, 20.0, 40.0, 70.0, None, float("nan")]
DF = pl.DataFrame({"age": AGE, "income": [5.0, 20.0, 40.0, 70.0, 0.0, 12.0]}, schema={"age": pl.Float64, "income": pl.Float64})
RETUNE = {"card": {"age": {"young": 50.0, "mid": 30.0}, "age_adjusted_score": {"weight": 1.0}}}


def test_parameters_report_the_referenced_edges_and_points():
    assert card().parameters().defaults() == {
        "card": {"age": {"young": 25.0, "mid": 10.0}, "age_adjusted_score": {"weight": 2.0}},
    }


def test_retuning_an_edge_and_points_changes_the_output_without_rebuilding_ir():
    sc = card()
    ir = engine.to_ir(sc)
    exe = Engine().bind(sc)
    before = exe.run(DF)
    assert before["age_score"].to_list() == [99.0, 5.0, 10.0, 15.0, 0.0, 15.0]
    assert before["score"].to_list() == [299.0, 112.0, 122.0, 132.0, 101.0, 132.0]
    after = exe.run(DF, params=RETUNE)
    assert after["age_score"].to_list() == [99.0, 5.0, 5.0, 15.0, 0.0, 15.0]
    assert after["score"].to_list() == [200.0, 107.0, 107.0, 117.0, 101.0, 117.0]
    assert engine.to_ir(sc) is ir


def _run_compiled(plan, inputs, valid, params):
    units = compile_plan(plan)
    leaves = {v.name: v for v in plan.versions if v.producer is None}
    values = {leaves[k].id: a for k, a in inputs.items()}
    masks = {leaves[k].id: m for k, m in valid.items()}
    bundles = {c.id: default_bundle(c.node)._replace(**params.get(c.node.origin.path, {}))
               for c in plan.calls if c.node.params}
    n = len(next(iter(inputs.values())))
    for call in plan.calls:
        unit = units[call.id]
        if unit.calls[0] is call:
            unit.run(values, masks, bundles, n)
    return units, {name: values[v.id].tolist() for name, v in plan.outputs.items() if v.producer is not None}


def test_compiled_kernels_give_the_interpreted_answers():
    sc = card()
    plan = resolve(sc)
    age = np.array([0.0 if a is None else a for a in AGE])
    inputs = {"age": age, "income": DF["income"].to_numpy()}
    valid = {"age": np.array([a is not None for a in AGE])}
    flat = {f"{p}/{node}": v for p, nodes in RETUNE.items() for node, v in nodes.items()}
    for params, frame in (({}, sc.run(DF)), (flat, sc.run(DF, params=RETUNE))):
        units, out = _run_compiled(plan, inputs, valid, params)
        assert all(isinstance(u, Kernel) for u in units.values())
        assert out == {k: frame[k].to_list() for k in out}


def test_string_values_bins_fall_back_to_python_when_compiled():
    sc = ScorecardConfig(name="card", variables=[ScoredVariable(
        variable_name="status", bins=[ValuesBin(value=50.0, items=["vip"])], default=DefaultBin(value=0.0))])
    units = compile_plan(resolve(sc))
    assert any(isinstance(u, Fallback) for u in units.values())
