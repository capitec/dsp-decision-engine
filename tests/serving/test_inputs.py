"""Serving real-world inputs: warm-up with dates and lists, JSON date coercion, code_path precedence, entry points."""
import asyncio
import datetime as dt
import json
import subprocess
import sys

import polars as pl
import pytest
from typing_extensions import TypedDict

from decider import flow, frame_step
from decider.config import JsonFileStore
from decider.exceptions import DeciderError, ParamsError
from decider.serving import RequestHandler
from decider.serving.handler import construct_handler_from_settings
from decider.serving.parse import coerce_record, has_date
from decider.serving.servers.starlette import create_app

from test_serving import _request


def tenure(decision_date: dt.date, opened: list[dt.date], history: list[float], seen: dt.datetime | None,
           extra: dict, bonus: float | None) -> int:
    return (decision_date - min(opened)).days + len(history)


def due_day(decision_date: dt.date, tenure: int) -> int:
    return (decision_date + dt.timedelta(days=tenure)).day


def build():
    return flow(tenure, due_day, name="t")


RECORD = {"decision_date": "2026-01-31", "opened": ["2026-01-01"], "history": [1.0, 2.0],
          "seen": "2026-01-31T09:30:00", "extra": {"k": 1}, "bonus": None}


def _handler(tmp_path, pipeline=build, **kw):
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {}})
    return RequestHandler(store, pipeline, **kw)


def test_stage_warms_date_list_dict_and_optional_inputs_without_a_sample(tmp_path):
    _handler(tmp_path).stage()


def test_raw_json_with_date_strings_is_scored_and_dates_come_back_iso(tmp_path):
    handler = _handler(tmp_path)
    handler.stage()
    handler.activate()
    body = json.dumps(RECORD).encode()
    out = json.loads(asyncio.run(handler.process_fn(body, "application/json", "application/json")).content)
    assert (out["due_day"], out["decision_date"]) == (4, "2026-01-31")
    status, body = _request(create_app(handler), "POST", "/invocations", body)
    assert status == 200 and json.loads(body)["due_day"] == 4


def test_a_json_array_with_date_strings_runs_as_a_frame(tmp_path):
    handler = _handler(tmp_path, lambda: flow(due_day, name="d"))
    handler.stage()
    handler.activate()
    rows = [{"decision_date": "2026-01-31", "tenure": 1}, {"decision_date": "2026-02-01", "tenure": 2}]
    status, body = _request(create_app(handler), "POST", "/invocations", json.dumps(rows).encode())
    assert status == 200
    out = json.loads(body)
    assert [(r["decision_date"], r["due_day"]) for r in out] == [("2026-01-31", 1), ("2026-02-01", 3)]


def test_a_malformed_date_is_a_400_naming_the_input(tmp_path):
    handler = _handler(tmp_path)
    handler.stage()
    handler.activate()
    status, body = _request(create_app(handler), "POST", "/invocations",
                            json.dumps({**RECORD, "decision_date": "31/01/2026"}).encode())
    assert status == 400
    assert "'decision_date'" in json.loads(body)["message"]


class Account(TypedDict):
    opened_date: dt.date
    balance: float


def test_dates_inside_typed_dict_items_are_coerced_and_bare_dicts_are_left_alone():
    # The engine side of nested frame columns is tested elsewhere; this pins the JSON coercion only.
    record = {"accounts": [{"opened_date": "2026-01-01", "balance": 1.0}], "extra": {"d": "2026-01-01"}}
    dates = {n: a for n, a in [("accounts", list[Account]), ("extra", dict)] if has_date(a)}
    out = coerce_record(record, dates)
    assert out["accounts"] == [{"opened_date": dt.date(2026, 1, 1), "balance": 1.0}]
    assert out["extra"] == {"d": "2026-01-01"}


class Dated(TypedDict, total=False):
    opened_date: dt.date


def test_a_typed_dict_declaring_only_its_dates_keeps_its_other_keys():
    record = {"accounts": [{"opened_date": "2026-01-01", "balance": 1.0}], "one": {"opened_date": "2026-01-02", "x": 2}}
    out = coerce_record(record, {"accounts": list[Dated], "one": Dated})
    assert out["accounts"] == [{"opened_date": dt.date(2026, 1, 1), "balance": 1.0}]
    assert out["one"] == {"opened_date": dt.date(2026, 1, 2), "x": 2}


@frame_step(reads={"accounts": list[Dated]}, writes=["oldest_year"])
def oldest_year(df: pl.DataFrame) -> pl.DataFrame:
    years = [min(a["opened_date"] for a in accounts).year for accounts in df["accounts"].to_list()]
    return df.with_columns(oldest_year=pl.Series(years))


def test_a_frame_step_reading_typed_dicts_gets_their_dates_from_json(tmp_path):
    handler = _handler(tmp_path, lambda: flow(oldest_year, name="f"))
    handler.stage()
    handler.activate()
    body = b'{"accounts": [{"opened_date": "2019-05-01", "balance": 1.0}, {"opened_date": "2021-01-01"}]}'
    assert json.loads(asyncio.run(handler.process_fn(body, "application/json", "application/json")).content)[
        "oldest_year"] == 2019


@frame_step(reads=["decision_date"], writes=["seen"])
def seen(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(seen=pl.lit(True))


def test_a_date_is_coerced_when_an_untyped_frame_step_reads_it_before_a_typed_step(tmp_path):
    handler = _handler(tmp_path, lambda: flow(seen, due_day, name="s"))
    handler.stage()
    handler.activate()
    body = b'{"decision_date": "2026-01-31", "tenure": 1}'
    assert json.loads(asyncio.run(handler.process_fn(body, "application/json", "application/json")).content)[
        "due_day"] == 1


def second(history: list[float]) -> float:
    return history[1]


def test_a_warm_up_that_fails_on_synthetic_inputs_suggests_a_sample_request(tmp_path):
    sample = tmp_path / "sample_request.json"
    with pytest.raises(DeciderError, match=r"IndexError.*'history': \[1\.0\].*sample_request\.json"):
        _handler(tmp_path, flow(second, name="g"), sample_request=str(sample)).stage()
    sample.write_text('{"history": [1.0, 2.0]}')
    _handler(tmp_path, flow(second, name="g"), sample_request=str(sample)).stage()


def test_a_failing_sample_request_is_named_in_the_error(tmp_path):
    sample = tmp_path / "sample_request.json"
    sample.write_text('{"history": []}')
    with pytest.raises(DeciderError, match="Fix .*sample_request.json"):
        _handler(tmp_path, flow(second, name="g"), sample_request=str(sample)).stage()


def test_a_sample_request_is_coerced_like_a_request(tmp_path):
    sample = tmp_path / "sample_request.json"
    sample.write_text(json.dumps(RECORD))
    _handler(tmp_path, sample_request=str(sample)).stage()


def test_invalid_params_still_raise_params_error_from_warm_up(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version({"params": {"nope": {"x": 1}}})
    with pytest.raises(ParamsError):
        RequestHandler(store, build).stage()


PIPELINE = '''
from decider import flow
def {name}(x: float) -> float:
    return x
def build():
    return flow({name}, name="p")
'''


def test_code_path_wins_over_an_earlier_path_entry_and_the_pipeline_file_is_reported(
        tmp_path, monkeypatch, capsys):
    from decider.settings import settings

    for name in ("mine", "other"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pipeline.py").write_text(PIPELINE.format(name=name))
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {}})
    mine = str(tmp_path / "mine")
    monkeypatch.setattr(sys, "path", [str(tmp_path / "other"), mine, *sys.path])
    monkeypatch.delitem(sys.modules, "pipeline", raising=False)
    monkeypatch.setattr(settings.api, "code_path", mine)
    monkeypatch.setattr(settings.api, "pipeline", "pipeline:build")
    monkeypatch.setattr(settings.config, "basepath", store.basepath, raising=False)
    handler = construct_handler_from_settings()
    handler.stage()
    handler.activate()
    live = handler.module_fn()
    assert "mine" in live.executable.score({"x": 1.0})
    assert str(tmp_path / "mine" / "pipeline.py") in capsys.readouterr().err
    monkeypatch.delitem(sys.modules, "pipeline")


@pytest.mark.parametrize("handler_path, expected", [
    ("credit_risk.inference:Handler", "Handler"),
    ("credit_risk.missing:Handler", "RequestHandler"),
    ("nope.inference:Handler", "RequestHandler"),
])
def test_a_dotted_handler_in_a_package_is_used_and_a_missing_one_falls_back(
        tmp_path, monkeypatch, handler_path, expected):
    from decider.settings import settings

    package = tmp_path / "credit_risk"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "pipeline.py").write_text(PIPELINE.format(name="mine"))
    (package / "inference.py").write_text("from decider.serving import RequestHandler\n\n"
                                          "class Handler(RequestHandler):\n    pass\n")
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"params": {}})
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("credit_risk", "credit_risk.pipeline", "credit_risk.inference"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(settings.api, "code_path", str(tmp_path))
    monkeypatch.setattr(settings.api, "pipeline", "credit_risk.pipeline:build")
    monkeypatch.setattr(settings.api, "handler", handler_path)
    monkeypatch.setattr(settings.config, "basepath", store.basepath, raising=False)
    assert type(construct_handler_from_settings()).__name__ == expected
    for name in ("credit_risk", "credit_risk.pipeline", "credit_risk.inference"):
        sys.modules.pop(name, None)


def test_python_dash_m_decider_runs_the_cli():
    result = subprocess.run([sys.executable, "-m", "decider", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "build" in result.stdout
