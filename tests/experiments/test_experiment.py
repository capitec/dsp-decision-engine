from __future__ import annotations

import polars as pl

from decider import flow, param
from decider.experiments import (Experiment, ExperimentDef, RunStatus, ScenarioStatus, dump_yaml, load_yaml,
                                 run_experiment, validate_scenarios)


def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def tier(approved: bool, income: float) -> float:
    return 1.0 if approved else (0.5 if income > 0 else 0.0)


def _pipeline():
    return flow(ratio, approved, tier, name="demo")


def _frame():
    return pl.DataFrame({"income": [1000.0] * 4, "debt": [float(i * 200) for i in range(4)]})


def test_experiment_runs_baseline_and_variant_with_a_manifest():
    pipeline = _pipeline()
    exp = (Experiment("demo-drift", pipeline, _frame(), entry="demo:build")
           .scenario("baseline")
           .scenario("limit_0.3", params={"demo": {"approved": {"limit": 0.3}}})
           .compare(outputs=("tier",)))
    result = exp.run()

    assert result.status is RunStatus.COMPLETED
    assert not result.nondeterministic
    assert [s.name for s in result.scenarios] == ["baseline", "limit_0.3"]
    assert result.scenarios[0].status is ScenarioStatus.COMPLETED
    assert result.scenarios[0].divergences == ()
    assert result.scenarios[1].divergences  # limit 0.3 changes tier on rows 2+
    assert result.manifest.kind == "experiment_run"
    assert result.manifest.input.row_count == 4
    assert result.manifest.revision.authored == "HEAD"
    assert result.manifest.reproducible
    assert result.preflight.ok


def test_undeclared_produced_override_is_rejected():
    pipeline = _pipeline()
    def_ = ExperimentDef(
        name="bad", flow={"entry": "demo:build"},
        scenarios=[{"name": "x", "overrides": {"ratio": 0.1}}],
    )
    problems = validate_scenarios(def_, pipeline)
    assert any("produced" in p for p in problems)

    declared = ExperimentDef(
        name="ok", flow={"entry": "demo:build"},
        scenarios=[{"name": "x", "at": "demo/ratio", "overrides": {"ratio": 0.1}}],
    )
    assert validate_scenarios(declared, pipeline) == []


def test_input_override_needs_no_declared_point():
    pipeline = _pipeline()
    def_ = ExperimentDef(
        name="ok", flow={"entry": "demo:build"},
        scenarios=[{"name": "x", "overrides": {"income": 500.0}}],
    )
    assert validate_scenarios(def_, pipeline) == []


def test_nondeterministic_run_has_a_first_class_non_reproducible_status():
    counter = {"n": 0}

    def drift(income: float, debt: float) -> float:
        counter["n"] += 1
        return (debt / income if income > 0 else 1.0) + counter["n"] * 1e-12

    pipeline = flow(drift, approved, tier, name="demo")
    def_ = ExperimentDef(name="impure", flow={"entry": "demo:build"},
                         scenarios=[{"name": "baseline"}], comparison={"outputs": ("tier",)})
    result = run_experiment(def_, pipeline, _frame())
    assert result.status is RunStatus.NON_REPRODUCIBLE
    assert result.nondeterministic


def test_resume_skips_a_completed_scenario_and_reruns_a_failed_one():
    pipeline = _pipeline()
    exp = (Experiment("demo-drift", pipeline, _frame())
           .scenario("baseline")
           .scenario("bad_cast", at="demo/ratio", overrides={"ratio": "not-a-float"})
           .compare(outputs=("tier",)))
    first = exp.run()
    assert first.scenarios[1].status is ScenarioStatus.FAILED
    second = run_experiment(exp.def_, pipeline, _frame(), resume=first)
    assert second.scenarios[0].status is ScenarioStatus.RESUMED
    assert second.scenarios[1].status is ScenarioStatus.FAILED


def test_yaml_round_trips_the_schema():
    pipeline = _pipeline()
    exp = (Experiment("demo-drift", pipeline, _frame(), entry="demo:build")
           .scenario("baseline")
           .scenario("limit_0.3", params={"demo": {"approved": {"limit": 0.3}}})
           .scenario("force_ratio", at="demo/ratio", overrides={"ratio": 0.1})
           .compare(outputs=("tier",)))
    restored = ExperimentDef.model_validate(load_yaml(dump_yaml(exp.def_.model_dump())))
    assert restored == exp.def_


def test_override_at_an_unknown_value_is_rejected():
    pipeline = _pipeline()
    def_ = ExperimentDef(
        name="bad", flow={"entry": "demo:build"},
        scenarios=[{"name": "x", "at": "demo/ratio", "overrides": {"not_a_value": 1.0}}],
    )
    problems = validate_scenarios(def_, pipeline)
    assert any("neither an input nor a produced value" in p for p in problems)
