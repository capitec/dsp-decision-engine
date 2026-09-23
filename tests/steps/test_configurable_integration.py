"""Configurable steps are first-class: they compose with code steps, run, and survive a JSON round trip."""
from __future__ import annotations

import json
import typing as t

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import TypeAdapter

from decider import ConfigurableStep, Engine, ParamRef, Value, engine, flow, step
from decider.config import JsonFileStore
from decider.engine.ir import ParamDecl
from decider.registry import import_path
from decider.steps import StepRef


def _threshold(x: float, threshold: float, above: float, below: float) -> float:
    return above if x > threshold else below


class ThresholdRule(ConfigurableStep):
    type: t.Literal["threshold_rule"] = "threshold_rule"
    input: str
    output: str
    threshold: Value[float]
    above: Value[float] = 1.0
    below: Value[float] = 0.0

    def to_ir(self, ctx):
        return ctx.call(
            self, _threshold, inputs={"x": self.input}, outputs=[self.output],
            values={"threshold": self.threshold, "above": self.above, "below": self.below},
        )


class RuleSet(ConfigurableStep):
    type: t.Literal["rule_set"] = "rule_set"
    rules: list[StepRef]

    def to_ir(self, ctx):
        return ctx.expand(self, flow(*self.rules))


def ratio(income: float, debt: float) -> float:
    return debt / income


@step(output="score")
def total(risk: float, band_lo: float, band_hi: float) -> float:
    return risk * 10 + band_lo + band_hi


DF = pl.DataFrame({"income": [1000.0, 2000.0, 500.0, 800.0], "debt": [200.0, 1500.0, 400.0, 100.0]})


def configs(risk_cut: Value[float] = 0.5):
    risk = ThresholdRule(name="risk", input="ratio", output="risk", threshold=risk_cut)
    bands = RuleSet(name="bands", rules=[
        ThresholdRule(name="lo", input="ratio", output="band_lo", threshold=0.2, above=2.0),
        ThresholdRule(name="hi", input="ratio", output="band_hi",
                      threshold=ParamRef(param="hi_cut", default=0.7), above=ParamRef(param="bonus", default=3.0, shared=True)),
    ])
    return risk, bands


def pipeline(risk, bands):
    return step(ratio) | risk | bands | total


def test_a_mixed_pipeline_runs_the_same_after_a_json_round_trip():
    risk, bands = configs()
    before = pipeline(risk, bands).run(DF)
    reloaded = [ConfigurableStep.load(json.loads(c.model_dump_json())) for c in (risk, bands)]
    assert [type(c) for c in reloaded] == [ThresholdRule, RuleSet]
    assert_frame_equal(pipeline(*reloaded).run(DF), before)
    assert before["score"].to_list() == [0.0, 15.0, 15.0, 0.0]


def test_loads_through_a_step_ref_by_alias_or_import_path():
    risk, _ = configs()
    doc = risk.model_dump(mode="json")
    assert doc["type"] == "threshold_rule"
    by_path = {**doc, "type": import_path(ThresholdRule)}
    for d in (doc, by_path):
        assert TypeAdapter(StepRef).validate_python(d) == risk
    assert ThresholdRule.model_validate_json(risk.model_dump_json()) == risk


def test_load_reads_a_json_file_and_checks_the_tag_against_the_class(tmp_path):
    _, bands = configs()
    path = tmp_path / "bands.json"
    path.write_text(bands.model_dump_json())
    assert RuleSet.load(path) == ConfigurableStep.load(str(path)) == bands
    assert ConfigurableStep.load(bands.model_dump_json()) == bands
    untagged = {k: v for k, v in bands.model_dump(mode="json").items() if k != "type"}
    assert RuleSet.load(untagged) == bands
    with pytest.raises(LookupError, match="not a registered RuleSet"):
        RuleSet.load({"type": "threshold_rule", "name": "x", "input": "a", "output": "b", "threshold": 1.0})


def test_swapping_a_literal_for_a_param_ref_moves_the_value_not_the_answer():
    literal, ref = configs(0.5)[0], configs(ParamRef(param="cut", default=0.5))[0]
    lit_node, ref_node = engine.to_ir(literal), engine.to_ir(ref)
    assert dict(lit_node.consts)["threshold"] == 0.5 and not lit_node.params
    assert ref_node.params == (ParamDecl("cut", float, 0.5, arg="threshold"),) and "threshold" not in dict(ref_node.consts)
    frame = flow(ratio).run(DF)
    assert_frame_equal(literal.run(frame), ref.run(frame))


def test_retuning_a_param_ref_changes_the_output_without_rebuilding_ir():
    risk, bands = configs(ParamRef(param="cut", default=0.5))
    p = pipeline(risk, bands)
    ir = engine.to_ir(p)
    exe = Engine().bind(p)
    at_defaults = exe.run(DF)["score"].to_list()
    retuned = exe.run(DF, params={"risk": {"cut": 0.1}, "bands": {"hi": {"hi_cut": 0.1}}, "shared": {"bonus": 5.0}})
    assert retuned["score"].to_list() != at_defaults
    assert retuned["score"].to_list() == [15.0, 17.0, 17.0, 15.0]
    assert engine.to_ir(p) is ir


def test_parameters_report_the_config_params():
    risk, bands = configs(ParamRef(param="cut", default=0.5))
    assert pipeline(risk, bands).parameters().defaults() == {
        "shared": {"bonus": 3.0}, "risk": {"cut": 0.5}, "bands": {"hi": {"hi_cut": 0.7}},
    }


def test_nested_rules_sit_under_the_owner_and_keep_their_own_origin():
    _, bands = configs()
    node = engine.to_ir(bands)
    assert node.origin.path == "bands" and node.origin.source == import_path(RuleSet)
    assert [(c.origin.path, c.origin.source) for c in node.children()] == [
        ("bands/lo", import_path(ThresholdRule)), ("bands/hi", import_path(ThresholdRule)),
    ]
    dumped = bands.model_dump()
    assert dumped["rules"][1]["threshold"] == {"param": "hi_cut", "default": 0.7, "shared": False}


def test_configs_and_params_from_the_config_store_run_the_same(tmp_path):
    risk, bands = configs(ParamRef(param="cut", default=0.5))
    params = {"risk": {"cut": 0.1}, "shared": {"bonus": 5.0}}
    expected = pipeline(risk, bands).run(DF, params=params)

    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version({
        "steps.risk": risk.model_dump(mode="json"), "steps.bands": bands.model_dump(mode="json"), "params": params,
    })
    fresh = JsonFileStore(basepath=str(tmp_path))
    config = fresh.read(fresh.latest_version()).config
    loaded = pipeline(ConfigurableStep.load(config["steps.risk"]), ConfigurableStep.load(config["steps.bands"]))
    assert_frame_equal(loaded.run(DF, params=config["params"]), expected)
