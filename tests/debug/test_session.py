"""Debug sessions: breakpoints, stepping, overrides, rewind and pause, interpreted mode."""
import polars as pl
import pytest

from decider import branch, flow, param, step
from decider.engine import Engine
from decider.engine.debug import (NodeFinished, NodeVisited, Overridden, ParamsValidated, Paused, RunFinished,
                                  SetValue)
from decider.engine.run.runners import InterpretedRunner
from decider.steps.trees import TreeConfig


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0)) -> float:
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap


pipeline = flow(disposable_income, affordability_ratio, cap_by_income_band)

FRAME = pl.DataFrame({
    "net_income":     [9200.0, 4100.0, 15000.0],
    "expenses":       [3100.0, 1500.0,  6000.0],
    "instalment":     [1200.0,  800.0,  2500.0],
    "term_cap":       [  60.0,   60.0,    60.0],
    "min_net_salary": [9200.0, 4100.0, 15000.0],
})


def term_cap(requested_term: float) -> float:
    return min(requested_term, 60.0)


@step(output="term_cap")
def cap_private(term_cap: float) -> float:
    return min(term_cap, 54.0)


@step(output="term_cap")
def cap_public(term_cap: float) -> float:
    return min(term_cap, 48.0)


def is_private(sector_code: int) -> bool:
    return sector_code == 1


term = flow(term_cap, branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"), name="term")
TERM_FRAME = pl.DataFrame({"requested_term": [72.0, 50.0, 84.0], "sector_code": [1, 2, 1]})


def _risk_tree() -> TreeConfig:
    return TreeConfig(name="risk_tree", tree={
        "nodes": [{"id": "n0", "data": {"type": "unary", "condition":
                                        {"op": ">", "feature": "requested_term", "threshold": 60}}},
                  {"id": "n1", "data": {"type": "leaf", "result_idx": 0}}],
        "edges": [{"source": "n0", "target": "n1", "data": {"sourceIndex": 0}}],
        "output": {"data": [{"risk_band": 1}], "default": {"risk_band": 0}, "dtypes": [["risk_band", "Int64"]]},
    })


def paths(session, kind):
    return [(e.origin.path, e.reason) if isinstance(e, Paused) else e.origin.path
            for e in session.events if isinstance(e, kind)]


def test_break_inspect_set_resume_changes_the_output():
    s = pipeline.session(FRAME)
    s.break_at("affordability_ratio")
    at = s.resume()
    assert (at.origin.path, at.when) == ("affordability_ratio", "before")
    assert s.value("disposable_income").to_list() == [6100.0, 2600.0, 9000.0]
    s.set("disposable_income", 1200.0)
    assert s.resume() is None
    assert s.output()["affordability_ratio"].to_list() == [1.0, 1.5, 0.48]
    assert isinstance(s.events[-1], RunFinished)


def test_an_untouched_session_outputs_what_run_does():
    s = pipeline.session(FRAME)
    s.resume()
    assert s.output().equals(pipeline.run(FRAME))


def test_output_before_the_end_is_an_error():
    with pytest.raises(RuntimeError, match="hasn't finished"):
        pipeline.session(FRAME).output()


def test_executable_session_runs_the_bound_executable():
    s = Engine().bind(pipeline).session(FRAME, params={"cap_by_income_band": {"cap": 30.0}})
    s.resume()
    assert s.output()["cap_by_income_band"].to_list() == [60.0, 30.0, 60.0]
    (validated,) = [e for e in s.events if isinstance(e, ParamsValidated)]
    assert validated.paths == ("cap_by_income_band",)


def test_a_prefix_breaks_once_on_entering_its_subtree():
    s = term.session(TERM_FRAME)
    s.break_at("term")
    assert s.resume().origin.path == "term"
    assert s.resume() is None


def test_a_prefix_breaks_at_the_first_node_under_it():
    s = term.session(TERM_FRAME)
    s.break_at("term/by_sector")
    assert s.resume().origin.path == "term/by_sector"
    s.clear_break("term/by_sector")
    s.break_at("term/by")                  # a prefix is whole path segments
    assert s.resume() is None


def test_a_predicate_breakpoint_sees_every_checkpoint():
    s = pipeline.session(FRAME)
    s.break_at(lambda cp: cp.when == "after" and cp.origin.path == "disposable_income")
    at = s.resume()
    assert (at.origin.path, at.when) == ("disposable_income", "after")
    assert paths(s, Paused) == [("disposable_income", "breakpoint")]


def test_step_runs_a_branch_as_one_node():
    s = term.session(TERM_FRAME)
    s.break_at("term/by_sector")
    s.resume()
    at = s.step()
    assert (at.origin.path, at.when) == ("term/by_sector", "after")
    assert s.value("term_cap").to_list() == [54.0, 48.0, 54.0]


def test_step_into_enters_the_taken_arms():
    s = term.session(TERM_FRAME)
    s.break_at("term/by_sector")
    s.resume()
    seen = []
    while (at := s.step_into()).origin.path != "term/by_sector":
        seen.append((at.origin.path, at.when))
    assert seen == [
        ("term/by_sector/is_private", "before"), ("term/by_sector/is_private", "after"),
        ("term/by_sector/cap_private", "before"), ("term/by_sector/cap_private", "after"),
        ("term/by_sector/cap_public", "before"), ("term/by_sector/cap_public", "after"),
    ]


def test_step_enters_sequences():
    s = term.session(TERM_FRAME)
    assert s.step().origin.path == "term"
    assert (s.step().origin.path, s.current.when) == ("term/term_cap", "before")
    assert (s.step().origin.path, s.current.when) == ("term/term_cap", "after")


def test_node_finished_carries_summaries_not_full_columns():
    frame = pl.concat([FRAME] * 10)        # 30 rows
    s = pipeline.session(frame)
    s.resume()
    finished = [e for e in s.events if isinstance(e, NodeFinished) and e.origin.path == "disposable_income"]
    summary = finished[0].outputs["disposable_income"]
    assert (summary.dtype, summary.rows, summary.nulls) == ("Float64", 30, 0)
    assert len(summary.preview) < 30
    assert all(len(v.preview) <= 5 for e in s.events if isinstance(e, RunFinished) for v in e.output.values())


def test_pause_stops_at_the_next_checkpoint():
    holder = []

    def slow(disposable_income: float) -> float:
        holder[0].pause()                  # as a UI thread would, mid-run
        return disposable_income

    s = flow(disposable_income, slow, affordability_ratio).session(FRAME)
    holder.append(s)
    at = s.resume()
    assert (at.origin.path, at.when) == ("slow", "after")
    assert paths(s, Paused) == [("slow", "pause")]


def test_set_casts_to_the_declared_dtype():
    s = pipeline.session(FRAME)
    s.set("net_income", 5000)              # an int into a float column
    s.resume()
    assert s.output()["affordability_ratio"].to_list()[0] == pytest.approx(1900 / 1200)


def test_set_rejects_a_value_that_does_not_cast():
    s = pipeline.session(FRAME)
    with pytest.raises(ValueError, match="doesn't cast"):
        s.set("net_income", "lots")
    with pytest.raises(ValueError, match="3 values"):
        s.set("net_income", [1.0, 2.0])


def test_set_takes_one_value_per_row_and_none_as_null():
    s = pipeline.session(FRAME)
    s.set("net_income", [1.0, None, 3.0])
    assert s.value("net_income").to_list() == [1.0, None, 3.0]


def test_set_of_a_value_not_produced_yet_is_an_error():
    with pytest.raises(KeyError, match="disposable_income"):
        pipeline.session(FRAME).set("disposable_income", 1.0)


def test_an_override_is_a_version_produced_by_override_at_the_current_path():
    s = pipeline.session(FRAME)
    s.break_at("affordability_ratio")
    s.resume()
    s.set("disposable_income", 1200.0)
    producers = [v.producer for v in s.state.versions("disposable_income@*")]
    assert producers == ["disposable_income", "override@affordability_ratio"]
    assert s.value("disposable_income@override@affordability_ratio").to_list() == [1200.0] * 3
    (event,) = [e for e in s.events if isinstance(e, Overridden)]
    assert (event.producer, event.value.preview, event.previous.preview) == (
        "override@affordability_ratio", (1200.0,) * 3, (6100.0, 2600.0, 9000.0))


def test_an_override_inside_a_branch_reaches_the_merge():
    s = term.session(TERM_FRAME)
    s.break_at("term/by_sector/cap_public")
    s.resume()
    s.set("term_cap", 10.0)
    s.resume()
    assert s.output()["term_cap"].to_list() == [10.0, 10.0, 10.0]


def test_rewind_reruns_downstream_with_the_current_values():
    s = pipeline.session(FRAME)
    s.resume()
    s.set("disposable_income", 1200.0)     # after the run: nothing downstream has seen it yet
    at = s.rewind("affordability_ratio")
    assert (at.origin.path, at.when) == ("affordability_ratio", "before")
    assert s.value("disposable_income").to_list() == [1200.0] * 3
    s.resume()
    assert s.output()["affordability_ratio"].to_list() == [1.0, 1.5, 0.48]
    assert paths(s, Paused)[-1] == ("affordability_ratio", "rewind")


def test_rewind_to_an_unknown_path_is_an_error_and_changes_nothing():
    s = pipeline.session(FRAME)
    s.break_at("affordability_ratio")
    s.resume()
    with pytest.raises(ValueError, match="nope"):
        s.rewind("nope")
    s.resume()
    assert s.output().equals(pipeline.run(FRAME))


def test_a_row_node_reports_visits_and_a_locator_breakpoint_stops_after_it():
    tree = _risk_tree()
    s = flow(term_cap, tree).session(TERM_FRAME)
    s.break_at("risk_tree#n1")
    at = s.resume()
    assert (at.origin.path, at.when) == ("risk_tree", "after")
    visits = {e.origin.locator: e.rows for e in s.events if isinstance(e, NodeVisited)}
    assert visits == {"n0": 3, "n1": 2}


def test_an_error_is_logged_and_ends_the_run():
    def boom(disposable_income: float) -> float:
        raise ZeroDivisionError("no")

    s = flow(disposable_income, boom).session(FRAME)
    with pytest.raises(ZeroDivisionError):
        s.resume()
    assert s.events[-1].path == "boom"
    with pytest.raises(RuntimeError, match="error"):
        s.resume()


def test_apply_dispatches_a_command():
    s = pipeline.session(FRAME)
    s.apply(SetValue("net_income", 5000.0))
    assert s.value("net_income").to_list() == [5000.0] * 3


class OnlyCalls:
    """A runner that pauses only at calls, as a fused runner pauses only at kernels."""

    def iterate(self, plan, state, params):
        calls = {c.node.origin.path for c in plan.calls}
        return (cp for cp in InterpretedRunner().iterate(plan, state, params) if cp.origin.path in calls)


def test_a_runner_with_fewer_checkpoints_still_breaks_steps_and_finishes():
    exe = Engine().bind(term)
    exe.runner = OnlyCalls()
    s = exe.session(TERM_FRAME)
    s.break_at("term/by_sector")           # the branch itself never checkpoints
    assert s.resume().origin.path == "term/by_sector/is_private"
    assert (s.step().origin.path, s.current.when) == ("term/by_sector/is_private", "after")
    s.set("term_cap", 20.0)
    s.resume()
    assert s.output()["term_cap"].to_list() == [20.0, 20.0, 20.0]
