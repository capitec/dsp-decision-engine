"""Editing a live session: replace or delete a step, keep what's upstream, re-run what's downstream."""
# No `from __future__ import annotations`: the config's pydantic fields resolve against this module.
from collections import Counter

import polars as pl
import pytest

from decider import ConfigurableStep, Value, branch, flow, step
from decider.engine.debug import COMMAND, EVENT_LOG, Delete, Edited, Paused, Replace
from decider.exceptions import WiringError
from decider.testing import MODES, no_recompile

CALLS: Counter = Counter()


def disposable_income(net_income: float, expenses: float) -> float:
    CALLS["disposable_income"] += 1
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def headroom(affordability_ratio: float) -> float:
    return affordability_ratio - 1.0


@step(output="affordability_ratio")
def halved_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment / 2.0


pipeline = flow(disposable_income, affordability_ratio, headroom)
edited = flow(disposable_income, halved_ratio.named("affordability_ratio"), headroom)
FRAME = pl.DataFrame({
    "net_income": [9200.0, 4100.0, 15000.0],
    "expenses":   [3100.0, 1500.0,  6000.0],
    "instalment": [1200.0,  800.0,  2500.0],
})


def test_replacing_a_step_changes_downstream_values_and_runs_nothing_upstream_again():
    s = pipeline.session(FRAME)
    s.resume()
    upstream = s.state.read(s.state.versions("disposable_income")[0])[0]
    calls = CALLS["disposable_income"]
    cp = s.replace("affordability_ratio", halved_ratio)
    assert (cp.origin.path, cp.when) == ("affordability_ratio", "before")
    assert s.events[-2:] == [Edited("replace", "affordability_ratio"), Paused(cp.origin, "before", "edit")]
    s.resume()
    assert CALLS["disposable_income"] == calls
    assert s.state.read(s.state.versions("disposable_income")[0])[0] is upstream
    out = s.output()
    assert out.equals(edited.run(FRAME))
    assert out["headroom"].to_list() != pipeline.run(FRAME)["headroom"].to_list()


@pytest.mark.parametrize("mode", MODES)
def test_replace_mid_session_gives_the_edited_pipelines_output_in_every_mode(mode):
    s = pipeline.session(FRAME, mode=mode)
    s.break_at("headroom")
    s.resume()
    s.replace("affordability_ratio", halved_ratio)
    s.clear_break("headroom")
    assert s.resume() is None
    assert s.output().equals(edited.run(FRAME))


@pytest.mark.parametrize("mode", MODES)
def test_an_edit_downstream_of_the_pause_keeps_the_session_where_it_was(mode):
    s = pipeline.session(FRAME, mode=mode)
    s.break_at("disposable_income")
    s.resume()
    s.set("net_income", 5000.0)
    cp = s.replace("headroom", step(headroom, name="headroom"))
    assert (cp.origin.path, cp.when) == ("disposable_income", "before")
    s.resume()
    assert s.value("disposable_income").to_list() == [1900.0, 3500.0, -1000.0]


def test_replacing_before_the_run_starts_just_swaps_the_pipeline():
    s = pipeline.session(FRAME)
    assert s.replace("affordability_ratio", halved_ratio) is None
    s.resume()
    assert s.output().equals(edited.run(FRAME))


def ratio_of_one(disposable_income: float) -> float:
    return 1.0


def test_the_replacement_takes_the_old_steps_name_and_path():
    s = pipeline.session(FRAME)
    s.replace("affordability_ratio", step(ratio_of_one, output="affordability_ratio"))
    assert "affordability_ratio" in [p for p, _ in s.executable.step.walk()]


@step(output="term_cap")
def cap_at_54(term_cap: float) -> float:
    return min(term_cap, 54.0)


@step(output="term_cap")
def cap_at_48(term_cap: float) -> float:
    return min(term_cap, 48.0)


def term_cap(requested_term: float) -> float:
    return min(requested_term, 60.0)


def instalment_months(term_cap: float) -> float:
    return term_cap * 1.0


waterfall = flow(term_cap, cap_at_54, cap_at_48, instalment_months)
TERMS = pl.DataFrame({"requested_term": [72.0, 50.0, 84.0]})


@pytest.mark.parametrize("mode", MODES)
def test_deleting_a_waterfall_layer_leaves_the_layer_before_it(mode):
    s = waterfall.session(TERMS, mode=mode)
    s.resume()
    assert s.output()["instalment_months"].to_list() == [48.0, 48.0, 48.0]
    s.delete("cap_at_48")
    s.resume()
    assert s.output()["instalment_months"].to_list() == [54.0, 50.0, 54.0]
    assert [v.producer for v in s.state.versions("term_cap@*")] == ["term_cap", "cap_at_54"]


def test_deleting_the_only_producer_of_a_read_name_raises_and_leaves_the_session_alone():
    s = pipeline.session(FRAME)
    s.break_at("headroom")
    cp = s.resume()
    events, executable = list(s.events), s.executable
    with pytest.raises(WiringError, match="nothing produces 'affordability_ratio', which 'headroom' reads"):
        s.delete("affordability_ratio")
    assert s.events == events and s.executable is executable and s.current == cp
    s.resume()
    assert s.output().equals(pipeline.run(FRAME))


def test_deleting_the_last_step_of_a_finished_run_finishes_again():
    s = waterfall.session(TERMS)
    s.resume()
    assert s.delete("instalment_months") is None
    assert s.finished and "instalment_months" not in s.output().columns


def test_a_missing_path_is_a_key_error():
    with pytest.raises(KeyError, match="no step at 'nope'"):
        pipeline.session(FRAME).delete("nope")


nested = flow(disposable_income, flow(affordability_ratio, headroom, name="inner"))


@pytest.mark.parametrize("mode", MODES)
def test_replace_inside_a_nested_named_flow(mode):
    s = nested.session(FRAME, mode=mode)
    s.resume()
    s.replace("inner/affordability_ratio", halved_ratio)
    s.resume()
    expected = flow(disposable_income, flow(halved_ratio.named("affordability_ratio"), headroom, name="inner"))
    assert s.output().equals(expected.run(FRAME))


def is_long(requested_term: float) -> bool:
    return requested_term > 60.0


by_length = flow(term_cap, branch(is_long, cap_at_54, cap_at_48, modifies=["term_cap"], name="by_length"),
                 instalment_months)


@pytest.mark.parametrize("mode", MODES)
def test_replace_a_branch_arm(mode):
    s = by_length.session(TERMS, mode=mode)
    s.resume()
    s.replace("by_length/cap_at_48", step(term_cap, name="cap_at_48", output="term_cap"))
    s.resume()
    assert s.output()["instalment_months"].to_list() == [54.0, 50.0, 54.0]


def scale(x: float, factor: float) -> float:
    return x * factor


class Scaled(ConfigurableStep):
    column: str
    factor: Value[float] = 1.0

    def to_ir(self, ctx):
        return ctx.call(self, scale, inputs={"x": self.column}, values={"factor": self.factor})


def test_replace_a_config_step_with_an_edited_config():
    config = Scaled(name="scaled", column="disposable_income", factor=2.0)
    s = flow(disposable_income, config).session(FRAME)
    s.resume()
    s.replace("scaled", config.model_copy(update={"factor": 3.0}))
    s.resume()
    assert s.output()["scaled"].to_list() == [18300.0, 7800.0, 27000.0]


def less_two(affordability_ratio: float) -> float:
    return affordability_ratio - 2.0


# Fused runs this pipeline as one kernel, whose inner values can't be set.
@pytest.mark.parametrize("mode", ["interpreted", "stepped"])
def test_overrides_upstream_of_the_edit_survive(mode):
    s = pipeline.session(FRAME, mode=mode)
    s.break_at("headroom")
    s.resume()
    s.set("affordability_ratio", 3.0)
    s.resume()
    s.replace("headroom", step(less_two, output="headroom"))
    s.resume()
    assert s.output()["headroom"].to_list() == [1.0, 1.0, 1.0]
    assert [v.producer for v in s.state.versions("affordability_ratio@*")][-1] == "override@headroom"


def test_an_edit_recompiles_only_what_changed():
    edited.session(FRAME, mode="stepped").resume()        # compiles halved_ratio's kernel
    s = pipeline.session(FRAME, mode="stepped")
    s.resume()
    with no_recompile():
        s.replace("affordability_ratio", halved_ratio)
        s.resume()
    assert s.output().equals(edited.run(FRAME))


def test_edit_events_and_commands_round_trip_as_json():
    s = pipeline.session(FRAME)
    s.resume()
    s.apply(Replace("affordability_ratio", halved_ratio))
    s.apply(COMMAND.validate_json('{"kind": "delete", "path": "headroom"}'))
    s.resume()
    assert Edited("delete", "headroom") in s.events
    assert EVENT_LOG.validate_json(EVENT_LOG.dump_json(s.events)) == s.events
    assert COMMAND.validate_json(COMMAND.dump_json(Delete("a/b"))) == Delete("a/b")
    assert "headroom" not in s.output().columns

