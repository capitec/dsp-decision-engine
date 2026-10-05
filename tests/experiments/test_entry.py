from __future__ import annotations

import polars as pl

from decider import flow
from decider.experiments import Experiment, ExperimentDef, RunStatus, run, validate
from decider.experiments._entry import import_flow, infer_entry


def ratio(income: float, debt: float) -> float:
    return debt / income


def approved(ratio: float) -> bool:
    return ratio <= 0.4


pipeline = flow(ratio, approved, name="demo")


def build():
    return flow(ratio, approved, name="demo")


def test_entry_is_the_module_variable_holding_the_pipeline():
    assert infer_entry(pipeline) == f"{__name__}:pipeline"
    assert import_flow(infer_entry(pipeline)) is pipeline


def test_entry_is_not_inferred_from_a_factory_function_because_nothing_is_called():
    assert infer_entry(build()) == ""


def test_entry_cannot_be_inferred_for_a_pipeline_no_module_rebuilds():
    assert infer_entry(flow(ratio, name="other")) == ""


def test_validate_reports_a_revision_scenario_when_the_entry_cannot_be_inferred():
    def_ = ExperimentDef(name="x", flow={"entry": ""}, scenarios=[{"name": "old", "revision": "HEAD^"}])
    assert any("old" in p and "flow.entry" in p for p in validate(def_, flow(ratio, name="other")))
    assert any("flow.entry" in p for p in validate(def_, build()))
    assert validate(def_, pipeline) == []


def test_from_def_takes_the_protocol_fields_and_loads_flow_and_input(tmp_path):
    pl.DataFrame({"income": [1000.0, 1000.0], "debt": [200.0, 350.0]}).write_parquet(tmp_path / "loans.parquet")
    def_ = ExperimentDef(name="x", flow={"entry": f"{__name__}:pipeline"}, input={"path": str(tmp_path / "loans.parquet")},
                         determinism=False)
    result = Experiment.from_def(def_).run()

    assert result.status is RunStatus.COMPLETED
    assert result.manifest.input.row_count == 2
    assert run(def_).status is RunStatus.COMPLETED
