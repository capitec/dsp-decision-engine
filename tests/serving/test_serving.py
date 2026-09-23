"""Serving a pipeline from code with documents from a config store: stage, activate, rollback, HTTP."""
import asyncio
import json
import sys
import threading
import typing as t

import pytest

from decider import ConfigurableStep, Value, flow, param
from decider.config import JsonFileStore, Version
from decider.exceptions import MissingInputError
from decider.serving.handler import RequestHandler
from decider.serving.servers.starlette import create_app


def _scale(x: float, factor: float) -> float:
    return x * factor


class Bonus(ConfigurableStep):
    type: t.Literal["serving_bonus"] = "serving_bonus"
    column: str
    factor: Value[float] = 1.0

    def to_ir(self, ctx):
        return ctx.call(self, _scale, inputs={"x": self.column}, values={"factor": self.factor})


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def capped(bonus: float, cap: float = param(5000.0)) -> float:
    return min(bonus, cap)


def build(bonus: ConfigurableStep):
    return flow(disposable_income, bonus, capped, name="afford").emit("bonus")


RECORD = {"net_income": 9000.0, "expenses": 3000.0}
V0 = {"bonus": {"type": "serving_bonus", "name": "bonus", "column": "disposable_income", "factor": 1.0},
      "params": {"afford": {"capped": {"cap": 5000.0}}}}
V1 = {"bonus": {**V0["bonus"], "factor": 2.0}, "params": {"afford": {"capped": {"cap": 3000.0}}}}
ANSWERS = {"0.0.0": (6000.0, 5000.0), "0.1.0": (12000.0, 3000.0)}


def _answer(out):
    return out["bonus"], out["capped"]


@pytest.fixture
def store(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version(V0)
    store.create_version(V1)
    return store


@pytest.fixture
def handler(store):
    handler = RequestHandler(store, build)
    handler.stage("0.0.0")
    handler.activate()
    return handler


def _score(handler, record=RECORD):
    live = handler.module_fn()
    return live.executable.score(record, live.params)


def test_stage_builds_the_version_without_serving_it(handler):
    assert handler.stage() == Version(0, 1, 0)
    assert (handler.staged, handler.active) == (Version(0, 1, 0), Version(0, 0, 0))
    assert _answer(_score(handler)) == ANSWERS["0.0.0"]


def test_activate_switches_answers_and_rollback_restores_them(handler):
    handler.stage("0.1.0")
    assert handler.activate() == Version(0, 1, 0)
    assert _answer(_score(handler)) == ANSWERS["0.1.0"]
    assert handler.staged is None
    assert handler.rollback() == Version(0, 0, 0)
    assert _answer(_score(handler)) == ANSWERS["0.0.0"]


def test_record_fields_survive_in_the_answer(handler):
    assert _score(handler)["net_income"] == 9000.0


def test_activate_requires_a_prior_stage(store):
    with pytest.raises(RuntimeError, match="nothing staged"):
        RequestHandler(store, build).activate()


def test_rollback_requires_a_previous_activation(handler):
    with pytest.raises(RuntimeError, match="roll back"):
        handler.rollback()


@pytest.mark.parametrize("params", [
    {"afford": {"capped": {"cap": "not a number"}}},
    {"not_a_step": {"x": 1}},
], ids=["invalid value", "unknown namespace"])
def test_a_failing_stage_raises_and_the_active_version_keeps_serving(handler, store, params):
    store.create_version({**V1, "params": params})
    with pytest.raises(ValueError):
        handler.stage()
    assert handler.staged is None
    assert handler.active == Version(0, 0, 0)
    assert _answer(_score(handler)) == ANSWERS["0.0.0"]


def test_a_missing_document_fails_the_stage(tmp_path):
    store = JsonFileStore(basepath=str(tmp_path))
    store.create_version({"params": {}})
    with pytest.raises(KeyError, match="'bonus'"):
        RequestHandler(store, build).stage()


def test_pipeline_may_be_an_import_path_or_a_step(store, tmp_path):
    by_path = RequestHandler(store, f"{__name__}:build")
    by_path.stage()
    bare = JsonFileStore(basepath=str(tmp_path / "bare"))
    bare.create_version({})
    fixed = RequestHandler(bare, flow(disposable_income, name="di"))
    fixed.stage()
    by_path.activate(), fixed.activate()
    assert _answer(_score(by_path)) == ANSWERS["0.1.0"]
    assert _score(fixed)["disposable_income"] == 6000.0


def test_concurrent_requests_during_swaps_each_see_one_version(handler):
    handler.stage("0.1.0")
    handler.activate()
    seen, errors, stop = set(), [], threading.Event()

    def client():
        body = json.dumps(RECORD).encode()
        try:
            while not stop.is_set():
                out = json.loads(asyncio.run(handler.process_fn(body, "application/json", "application/json")).content)
                seen.add(_answer(out))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=client) for _ in range(4)]
    for th in threads:
        th.start()
    for _ in range(20):
        handler.rollback()
        handler.stage("0.1.0")
        handler.activate()
    stop.set()
    for th in threads:
        th.join()
    assert not errors
    assert seen <= set(ANSWERS.values())


# --- over HTTP ------------------------------------------------------------------------------------


def _request(app, method, path, body=b"", content_type="application/json", accept="*/*"):
    # A raw ASGI call: starlette's TestClient needs httpx, which isn't a dependency.
    async def run():
        sent, response = False, {}

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                response["status"] = message["status"]
                response["headers"] = dict(message["headers"])
            elif message["type"] == "http.response.body":
                response["body"] = response.get("body", b"") + message.get("body", b"")

        headers = [(b"content-type", content_type.encode()), (b"accept", accept.encode())]
        scope = {"type": "http", "method": method, "path": path, "headers": headers, "query_string": b"",
                 "root_path": ""}
        await app(scope, receive, send)
        return response["status"], response.get("body", b"")

    return asyncio.run(run())


def test_ping_is_503_until_a_version_is_active(store):
    handler = RequestHandler(store, build)
    app = create_app(handler)
    assert _request(app, "GET", "/ping")[0] == 503
    assert _request(app, "POST", "/invocations", json.dumps(RECORD).encode())[0] == 503
    handler.stage()
    assert _request(app, "GET", "/ping")[0] == 503
    handler.activate()
    assert _request(app, "GET", "/ping")[0] == 200


def test_invocations_scores_a_record_and_follows_activation(handler):
    app = create_app(handler)
    status, body = _request(app, "POST", "/invocations", json.dumps(RECORD).encode())
    assert status == 200
    assert _answer(json.loads(body)) == ANSWERS["0.0.0"]
    handler.stage("0.1.0")
    handler.activate()
    assert _answer(json.loads(_request(app, "POST", "/invocations", json.dumps(RECORD).encode())[1])) == ANSWERS["0.1.0"]


def test_invocations_runs_a_json_array_or_csv_as_a_frame(handler):
    app = create_app(handler)
    rows = [RECORD, {"net_income": 4000.0, "expenses": 1000.0}]
    status, body = _request(app, "POST", "/invocations", json.dumps(rows).encode())
    assert status == 200
    assert [r["capped"] for r in json.loads(body)] == [5000.0, 3000.0]
    status, body = _request(app, "POST", "/invocations", b"net_income,expenses\n9000.0,3000.0\n",
                            content_type="text/csv", accept="text/csv")
    assert status == 200
    header, row = (line.split(",") for line in body.decode().splitlines())
    assert dict(zip(header, row))["capped"] == "5000.0"


def test_a_missing_input_is_a_400(handler):
    status, body = _request(create_app(handler), "POST", "/invocations", json.dumps({"net_income": 1.0}).encode())
    assert status == MissingInputError._STATUS_CODE == 400
    assert "expenses" in json.loads(body)["message"]


def test_an_unsupported_content_type_is_a_415(handler):
    assert _request(create_app(handler), "POST", "/invocations", b"x", content_type="text/plain")[0] == 415


INFERENCE = '''
from decider.serving.format import Response
from decider.serving.handler import RequestHandler

class Handler(RequestHandler):
    def output_fn(self, output, accept):
        return Response(b"%d" % output["capped"], "text/plain")
'''


def test_startup_uses_the_inference_handler_and_activates_the_latest_version(store, tmp_path, monkeypatch):
    from decider.settings import settings

    code = tmp_path / "code"
    code.mkdir()
    (code / "inference.py").write_text(INFERENCE)
    monkeypatch.setattr(settings.api, "code_path", str(code))
    monkeypatch.setattr(settings.api, "pipeline", f"{__name__}:build")
    monkeypatch.setattr(settings.config, "basepath", store.basepath, raising=False)
    monkeypatch.delitem(sys.modules, "inference", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    app = create_app()

    async def lifespan():
        events, sent = asyncio.Queue(), []
        events.put_nowait({"type": "lifespan.startup"})

        async def send(message):
            sent.append(message["type"])
            events.put_nowait({"type": "lifespan.shutdown"})

        await app({"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}, events.get, send)
        return sent

    assert asyncio.run(lifespan()) == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert type(app.state.handler).__name__ == "Handler"
    assert app.state.handler.active == Version(0, 1, 0)
    assert _request(app, "POST", "/invocations", json.dumps(RECORD).encode())[1] == b"3000"
