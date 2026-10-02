from __future__ import annotations

import pytest

from decider.exceptions import DeciderError
from decider.experiments import (Bars, Changes, Divergence, Experiment, ExperimentDef, ValueCounts,
                                 check_experiment_version, compatible, validate)

LIMIT_03 = {"demo": {"approved": {"limit": 0.3}}}


def _def(**kwargs):
    return ExperimentDef(name="x", flow={"entry": "demo:build"}, **kwargs)


def test_json_round_trips_the_definition_with_its_metrics_and_plots(pipeline, frame, tmp_path):
    exp = (Experiment("demo-drift", pipeline, frame, entry="demo:build")
           .scenario("limit_0.3", params=LIMIT_03)
           .scenario("force_ratio", at="demo/ratio", overrides={"ratio": 0.1})
           .metric(Divergence(column="tier", rtol=0.01), ValueCounts(column="approved", nodes="demo/approved"))
           .plot(Changes(metric="divergence"), Bars(metric="value_counts")))
    restored = ExperimentDef.model_validate_json(exp.save(tmp_path / "experiment.json").read_text())

    assert restored == exp.def_
    assert isinstance(restored.metrics[0], Divergence) and restored.metrics[0].rtol == 0.01
    assert isinstance(restored.plots[1], Bars)


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
    with pytest.raises(DeciderError, match="newer"):
        check_experiment_version("3")


def test_the_engine_window_is_a_version_specifier():
    assert compatible(">=1.0,<2.0", "1.4.2")
    assert not compatible(">=1.0,<2.0", "2.0.0")
    assert compatible("", "0.1")
