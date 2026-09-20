"""The serving layer (doc 02 §3.6): SageMaker routes plus the params-play
surface built on `Pipeline.serve()` (doc 03 §6).

`starlette` is not installed in this environment (it is an optional
dependency, per doc 02 §3.6 — see `decider2/serving/app.py`), so every test
here drives the hand-rolled stdlib ASGI fallback directly. If starlette
*is* importable, `app()` returns a real `Starlette` instance instead and
`_client()` below switches to `starlette.testclient.TestClient` with no
other change to these tests — both backends share one `Dispatcher`
(`decider2/serving/dispatch.py`), so the same assertions exercise the same
logic either way.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from decider2.examples.flagship import pipeline as flagship_pipeline
from decider2.serving import app as build_app

try:
    from starlette.testclient import TestClient

    HAVE_STARLETTE = True
except ImportError:
    HAVE_STARLETTE = False


RECORD = {
    "net_income": 4100.0,
    "expenses": 1500.0,
    "instalment": 800.0,
    "term_cap": 60.0,
    "min_net_salary": 4100.0,
}  # min_net_salary (4100) < income_threshold (5000 default) -> cap_by_income_band = min(60, 48) = 48.0


def _drive_asgi(app: Any, method: str, path: str, body: Any) -> tuple[int, Any]:
    """Construct a scope/receive/send triple by hand and call the ASGI app
    directly — no server, no socket, exactly what the task's TESTS section
    asks for when starlette (and therefore its TestClient) is absent."""

    async def _run() -> tuple[int, Any]:
        payload = b"" if body is None else json.dumps(body).encode()
        scope = {"type": "http", "method": method, "path": path, "headers": [], "query_string": b""}
        delivered = False

        async def receive():
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}

        response: dict[str, Any] = {}
        chunks: list[bytes] = []

        async def send(message):
            if message["type"] == "http.response.start":
                response["status"] = message["status"]
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))

        await app(scope, receive, send)
        raw = b"".join(chunks)
        return response["status"], (json.loads(raw) if raw else None)

    return asyncio.run(_run())


@dataclass
class _Resp:
    status_code: int
    _data: Any

    def json(self) -> Any:
        return self._data


class _DirectClient:
    """Same tiny interface `starlette.testclient.TestClient` exposes for
    what these tests need — `.get`/`.post` returning a `.status_code`/
    `.json()` object — so every test body below is backend-agnostic."""

    def __init__(self, app: Any):
        self.app = app

    def get(self, path: str) -> _Resp:
        status, data = _drive_asgi(self.app, "GET", path, None)
        return _Resp(status, data)

    def post(self, path: str, json: Any = None) -> _Resp:
        status, data = _drive_asgi(self.app, "POST", path, json)
        return _Resp(status, data)


def _client(app: Any):
    return TestClient(app) if HAVE_STARLETTE else _DirectClient(app)


@pytest.fixture
def app_obj():
    # Fresh handle per test (doc 02 §3.6 rule 3: the generation pointer
    # lives on the handle) — so tests never see each other's staged params.
    return build_app(flagship_pipeline, mode="live")


@pytest.fixture
def client(app_obj):
    return _client(app_obj)


# --- SageMaker convention ---------------------------------------------------


def test_ping_is_200_when_the_model_is_ready(client):
    resp = client.get("/ping")
    assert resp.status_code == 200


def test_invocations_returns_a_real_decision(client):
    resp = client.post("/invocations", json=RECORD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["cap_by_income_band"] == 48.0
    # additive frame, doc 03 §7 — the record's own fields survive.
    assert body["net_income"] == 4100.0


# --- the parameter-play surface ---------------------------------------------


def test_params_round_trip(client):
    doc = client.get("/params").json()
    assert doc == {"cap_by_income_band": {"cap": 48.0, "income_threshold": 5000.0}}

    resp = client.post("/params", json=doc)
    assert resp.status_code == 200

    doc_again = client.get("/params").json()
    assert doc_again == doc


def test_params_schema_is_pydantics_own_json_schema(client):
    schema = client.get("/params/schema").json()
    cap_field = schema["cap_by_income_band"]["properties"]["cap"]
    assert cap_field["minimum"] == 6
    assert cap_field["maximum"] == 60
    assert cap_field["default"] == 48.0


def test_a_params_change_actually_changes_the_decision(client):
    before = client.post("/invocations", json=RECORD).json()["cap_by_income_band"]
    assert before == 48.0

    resp = client.post("/params", json={"cap_by_income_band": {"cap": 24.0}})
    assert resp.status_code == 200
    assert resp.json()["params"]["cap_by_income_band"]["cap"] == 24.0

    after = client.post("/invocations", json=RECORD).json()["cap_by_income_band"]
    assert after == 24.0


def test_params_preview_shows_both_sides_without_activating(client):
    resp = client.post(
        "/params/preview",
        json={"record": RECORD, "params": {"cap_by_income_band": {"cap": 12.0}}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["current"]["cap_by_income_band"] == 48.0
    assert body["proposed"]["cap_by_income_band"] == 12.0

    # preview must not have staged or activated anything.
    still_current = client.post("/invocations", json=RECORD).json()["cap_by_income_band"]
    assert still_current == 48.0
    assert client.get("/params").json()["cap_by_income_band"]["cap"] == 48.0


def test_a_validation_error_is_400_not_500(client):
    resp = client.post("/params", json={"cap_by_income_band": {"cap": 999.0}})
    assert resp.status_code == 400
    assert "cap" in resp.json()["detail"]

    # the rejected document must never have activated.
    assert client.get("/params").json()["cap_by_income_band"]["cap"] == 48.0


def test_an_unknown_module_is_also_400_not_500(client):
    resp = client.post("/params", json={"not_a_real_module": {"x": 1}})
    assert resp.status_code == 400


def test_rollback_restores_the_previous_answer(client):
    client.post("/params", json={"cap_by_income_band": {"cap": 24.0}})
    assert client.post("/invocations", json=RECORD).json()["cap_by_income_band"] == 24.0

    resp = client.post("/rollback")
    assert resp.status_code == 200

    restored = client.post("/invocations", json=RECORD).json()["cap_by_income_band"]
    assert restored == 48.0


def test_rollback_with_nothing_to_roll_back_to_is_400_not_500(client):
    resp = client.post("/rollback")
    assert resp.status_code == 400


# --- doc 00 §2c --------------------------------------------------------------


def test_health_reports_mode_fingerprint_generations_and_gil(client):
    body = client.get("/health").json()
    assert body["mode"] == "live"
    assert isinstance(body["fingerprint"], str) and body["fingerprint"]
    assert body["generations"] == 1
    assert body["kernels"], "expected at least one kernel in the GIL report"
    # the flagship example declares no @step(nogil=True) steps (doc 00 §2b:
    # off by default), so every kernel here should report holding the GIL.
    assert all(k["holds_gil"] is True for k in body["kernels"])


def test_health_fingerprint_is_unchanged_by_a_params_change(client):
    before = client.get("/health").json()["fingerprint"]
    client.post("/params", json={"cap_by_income_band": {"cap": 24.0}})
    after = client.get("/health").json()["fingerprint"]
    assert before == after
