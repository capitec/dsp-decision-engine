"""Events and commands as JSON: what a transport sends, and summaries instead of values."""
import json

import polars as pl

from decider import flow
from decider.engine.debug import (COMMAND, EVENT, EVENT_LOG, BreakAt, ClearBreak, Pause, ReloadFailed, Resume, Rewind,
                                  SetValue, StepInto, StepOver)


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


FRAME = pl.DataFrame({
    "net_income": [float(i) for i in range(100, 150)],
    "expenses": [10.0] * 50,
    "instalment": [2.0] * 50,
})


def a_session_log():
    s = flow(disposable_income, ratio).session(FRAME, params_validation="lazy")
    s.break_at("ratio")
    s.resume()
    s.set("disposable_income", [0.5] + [1.0] * 49)
    s.step()
    s.resume()
    return s.events


def test_the_event_log_round_trips_as_json():
    events = a_session_log()
    assert {e.kind for e in events} >= {"run_started", "node_started", "node_finished", "paused", "overridden",
                                         "run_finished"}
    assert EVENT_LOG.validate_json(EVENT_LOG.dump_json(events)) == events
    assert [EVENT.validate_json(EVENT.dump_json(e)) for e in events] == events


def test_events_are_tagged_by_kind():
    first = json.loads(EVENT.dump_json(a_session_log()[0]))
    assert first == {"kind": "run_started", "rows": 50}


def test_the_log_carries_summaries_never_full_columns():
    text = EVENT_LOG.dump_json(a_session_log()).decode()
    previews = [s["preview"] for e in json.loads(text)
                for s in [*e.get("outputs", {}).values(), *e.get("output", {}).values()]]
    assert previews and all(len(p) <= 5 for p in previews)
    assert "149.0" not in text               # the last row's net_income


def test_a_failed_reload_round_trips_as_its_own_event():
    event = ReloadFailed("SyntaxError: invalid syntax")
    assert json.loads(EVENT.dump_json(event)) == {"kind": "reload_failed", "error": "SyntaxError: invalid syntax"}
    assert EVENT.validate_json(EVENT.dump_json(event)) == event


def test_commands_round_trip_as_json():
    commands = [BreakAt("term"), ClearBreak("term#n17"), StepOver(), StepInto(), Resume(), Pause(),
                SetValue("x", 1.5), SetValue("x", [1, None]), Rewind("term/cap")]
    assert [COMMAND.validate_json(COMMAND.dump_json(c)) for c in commands] == commands


def test_a_decoded_command_drives_a_session():
    s = flow(disposable_income, ratio).session(FRAME)
    for text in ['{"kind": "break_at", "target": "ratio"}', '{"kind": "resume"}',
                 '{"kind": "set", "name": "disposable_income", "value": 4.0}', '{"kind": "resume"}']:
        s.apply(COMMAND.validate_json(text))
    assert s.output()["ratio"].to_list() == [2.0] * 50
