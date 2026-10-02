from __future__ import annotations

import typing as t

import polars as pl

from decider import flow, param
from decider.experiments import (BasePlotType, Experiment, ExperimentDef, OutcomeCounts, RunStatus, Sankey,
                                 dump_yaml, load_yaml)
from decider.experiments.runner import run_experiment


def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def _pipeline():
    return flow(ratio, approved, name="demo")


def _frame():
    return pl.DataFrame({"income": [1000.0] * 4, "debt": [float(i * 200) for i in range(4)]})


def test_outcome_counts_plot_runs_through_the_builder():
    exp = (Experiment("demo-plots", _pipeline(), _frame())
           .scenario("baseline")
           .plot(OutcomeCounts(type="outcome_counts", column="approved")))
    result = exp.run()
    assert result.status is RunStatus.COMPLETED
    assert result.plots["outcome_counts"]["column"] == "approved"
    assert sum(result.plots["outcome_counts"]["series"].values()) == 4


def test_sankey_plot_matches_the_bare_function_it_wraps():
    exp = Experiment("demo-plots", _pipeline(), _frame()).scenario("baseline").plot(Sankey(type="sankey"))
    result = exp.run()
    # `result.sankey` (the hardcoded call) and `result.plots["sankey"]` (the registry
    # route) run `path_sankey` over the same baseline, so they must agree exactly.
    assert result.plots["sankey"] == result.sankey


def test_plot_resolves_by_registered_type_tag_through_yaml():
    def_ = ExperimentDef(name="plots", flow={"entry": "demo:build"},
                         plots=[{"type": "outcome_counts", "column": "approved"}])
    restored = ExperimentDef.model_validate(load_yaml(dump_yaml(def_.model_dump())))
    assert restored == def_
    assert isinstance(restored.plots[0], OutcomeCounts)


def test_a_custom_plot_type_only_needs_to_implement_visualize():
    class ConstantPlot(BasePlotType):
        type: t.Literal["constant_plot"] = "constant_plot"

        def visualize(self, step, frame, params, baseline, outputs):
            return {"kind": "constant", "value": 1}

    exp = Experiment("demo-plots", _pipeline(), _frame()).scenario("baseline").plot(ConstantPlot())
    result = exp.run()
    assert result.plots["constant_plot"] == {"kind": "constant", "value": 1}


def test_no_plots_means_no_plots_in_the_result():
    result = run_experiment(ExperimentDef(name="bare", flow={"entry": "demo:build"},
                                          scenarios=[{"name": "baseline"}]), _pipeline(), _frame())
    assert result.plots == {}
