from __future__ import annotations

import pytest

from decikit.exceptions import DecikitError
from decikit.lab.experiments import Experiment, ExperimentDef, validate
from decikit.lab.experiments._experiments import check_experiment_version
from decikit.lab.diff._revisions import compatible
from decikit.lab.experiments.metrics import Describe, Divergence, ValueCounts
from decikit.lab.experiments.plots import Bars, Changes

LIMIT_03 = {"demo": {"approved": {"limit": 0.3}}}


def _def(**kwargs):
    return ExperimentDef(name="x", flow={"entry": "demo:build"}, **kwargs)


def test_json_round_trips_the_definition_with_its_metrics_and_plots(pipeline, frame, tmp_path):
    exp = (Experiment("demo-drift", pipeline, frame)
           .scenario("limit_0.3", params=LIMIT_03)
           .scenario("force_ratio", at="demo/ratio", overrides={"ratio": 0.1})
           .metric(Divergence(column="tier", rtol=0.01), ValueCounts(column="approved", nodes="demo/approved"))
           .plot(Changes(metric="divergence"), Bars(metric="value_counts")))
    exp.def_.flow.entry = "demo:build"
    restored = ExperimentDef.model_validate_json(exp.save(tmp_path / "experiment.json").read_text())

    assert restored == exp.def_
    assert isinstance(restored.metrics[0], Divergence) and restored.metrics[0].rtol == 0.01
    assert isinstance(restored.plots[1], Bars)


def test_a_revision_scenario_without_a_flow_entry_is_invalid(pipeline):
    scenarios = [{"name": "old", "revision": "HEAD^"}]
    problems = validate(ExperimentDef(name="x", flow={"entry": ""}, scenarios=scenarios), pipeline)
    assert any("old" in p and "flow.entry" in p for p in problems)
    assert validate(_def(scenarios=scenarios), pipeline) == []


def test_undeclared_produced_override_is_rejected(pipeline):
    problems = validate(_def(scenarios=[{"name": "x", "overrides": {"ratio": 0.1}}]), pipeline)
    assert any("produced" in p for p in problems)
    assert validate(_def(scenarios=[{"name": "x", "at": "demo/ratio", "overrides": {"ratio": 0.1}}]), pipeline) == []


def test_input_override_needs_no_declared_point(pipeline):
    assert validate(_def(scenarios=[{"name": "x", "overrides": {"income": 500.0}}]), pipeline) == []


def test_override_of_an_unknown_value_is_rejected(pipeline):
    problems = validate(_def(scenarios=[{"name": "x", "at": "demo/ratio", "overrides": {"nope": 1.0}}]), pipeline)
    assert any("neither an input nor a produced value" in p for p in problems)


def test_baseline_is_reserved_for_the_unchanged_run(pipeline):
    assert any("unchanged run" in p for p in validate(_def(scenarios=[{"name": "baseline"}]), pipeline))


def test_a_plot_must_name_a_collected_metric_that_gives_the_summary_it_requires(pipeline):
    missing = _def(plots=[Changes(metric="divergence")])
    assert any("doesn't collect" in p for p in validate(missing, pipeline))

    wrong = _def(metrics=[ValueCounts(name="divergence", column="tier")], plots=[Changes(metric="divergence")])
    assert any("needs a Divergences summary" in p for p in validate(wrong, pipeline))


def test_two_unnamed_metrics_of_one_type_need_names(pipeline):
    clash = _def(metrics=[Divergence(column="tier"), Divergence(column="approved")])
    assert any("two metrics" in p for p in validate(clash, pipeline))
    named = _def(metrics=[Divergence(column="tier"), Divergence(name="approved", column="approved")])
    assert validate(named, pipeline) == []


def test_a_newer_experiment_version_is_refused():
    check_experiment_version("2")
    with pytest.raises(DecikitError, match="newer"):
        check_experiment_version("3")


def test_the_engine_window_is_a_version_specifier():
    assert compatible(">=1.0,<2.0", "1.4.2")
    assert not compatible(">=1.0,<2.0", "2.0.0")
    assert compatible("", "0.1")


def test_a_step_metric_must_name_a_step_that_exists():
    from example_projects.experiments.loan_pipeline import build

    def problems(**kwargs):
        return validate(ExperimentDef(name="x", flow={"entry": "e:f"}, metrics=[Describe(column="rate", **kwargs)]),
                        build())

    assert problems(node_id="03ed37e0af6b") == [] and problems(nodes="loans/pricing/*") == []
    assert problems(node_id="000000000000") == ["metric 'describe': no step has id '000000000000'"]
    assert problems(nodes="loans/pricng/*") == ["metric 'describe': nodes='loans/pricng/*' matches no step"]
    assert "pick one" in problems(nodes="loans/rate", node_id="03ed37e0af6b")[0]


def test_an_experiment_loads_from_json_with_its_flow_and_input_beside_it(tmp_path):
    from example_projects.experiments.loan_pipeline import applications, build

    applications(300).write_parquet(tmp_path / "applications.parquet")
    exp = (Experiment("x", build(), applications(1))
           .scenario("premium_8pct", params={"loans": {"pricing": {"unsecured": {"premium": 0.08}}}})
           .metric(Divergence(column="approved"), Describe(column="rate", node_id="03ed37e0af6b"))
           .plot(Changes(metric="divergence")))
    exp.def_.flow.entry = "example_projects.experiments.loan_pipeline:build"
    exp.def_.input.path = "applications.parquet"
    loaded = Experiment.load(exp.save(tmp_path / "experiment.json"))

    assert loaded.def_ == exp.def_
    result = loaded.run()
    assert result.manifest.input.row_count == 300
    assert set(result.summaries["describe"]) == {"baseline", "premium_8pct"}
