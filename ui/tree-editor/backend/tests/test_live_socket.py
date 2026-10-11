"""The live socket itself: session renewal and sign-out, messages that fail every time, and clients that stop reading.

The database tests need a migrated Postgres (the e2e stack's) and run only when TREE_EDITOR_DB=1, with DSP_DB__PORT
pointing at it.
"""

import asyncio
import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from dsp_decikit.asgi import app
from dsp_decikit.settings import settings

from .test_live_sync import RULE_A, RULE_B, Socket, flat, name, project  # noqa: F401

db = pytest.mark.skipif(os.environ.get("TREE_EDITOR_DB") != "1", reason="needs the e2e stack's Postgres (TREE_EDITOR_DB=1)")


@pytest.fixture
def live_on(monkeypatch):
    monkeypatch.setattr(settings, "live_collab", True)


def run(coro):
    """Run `coro` on a loop of its own; the app's sockets run on the test client's loop, so neither keeps the
    other's pooled connections."""
    from dsp_decikit.db.session import engine

    async def fresh():
        await engine.dispose(close=False)
        try:
            return await coro
        finally:
            await engine.dispose()

    return asyncio.run(fresh())


async def signed_in_owner():
    """A new user with a session and a flat-rule project they own -> (session token, project id)."""
    from dsp_decikit.auth import sessions
    from dsp_decikit.db.models import Project, ProjectType, User
    from dsp_decikit.db.session import db_session

    async with db_session() as session:
        user = User(email=f"{uuid.uuid4().hex[:10]}@example.com")
        session.add(user)
        await session.flush()
        project = Project(name="live", project_type=ProjectType.FLAT_RULE, project_data=flat(RULE_A, RULE_B),
                          created_by=user.id)
        session.add(project)
        await session.commit()
        user_id, project_id = user.id, project.id
    token, _ = await sessions.create_session(user_id)
    return token, project_id


async def expiring_in(token: str, seconds: int):
    from dsp_decikit.auth import sessions

    info = await sessions.get_session(token)
    info.last_activity = int(time.time()) - settings.auth.session_rolling_duration_s + seconds
    assert await sessions.renew_session(token, info)


async def expiry(token: str):
    from dsp_decikit.auth import sessions

    info = await sessions.get_session(token)
    return None if info is None else info.expires_at


def connect(token: str, project_id):
    return TestClient(app, cookies={settings.auth.cookie.key: token}).websocket_connect(
        f"/api/v1/projects/{project_id}/live")


def receive_until(ws, kind: str) -> dict:
    while True:
        message = json.loads(ws.receive_text())
        if message["type"] == kind:
            return message


# ---- the session ---------------------------------------------------------------------------------------------------

@db
def test_an_edit_sent_near_the_session_expiry_renews_it_and_hands_the_browser_its_new_status_cookie(live_on):
    token, project_id = run(signed_in_owner())
    run(expiring_in(token, 60))
    before = run(expiry(token))

    with connect(token, project_id) as ws:
        ws.send_text(json.dumps({"type": "hello", "since": 0, "client_id": "c1"}))
        version = receive_until(ws, "ready")["version"]
        ws.send_text(json.dumps({"type": "ping"}))
        receive_until(ws, "pong")
        assert run(expiry(token)) == before  # a ping is not activity: an open tab alone must not keep it alive
        ws.send_text(json.dumps({"type": "ops", "id": "m1", "base": version, "ops": [name("a", "Renewed")]}))
        cookie = receive_until(ws, "session")["cookie"]
        receive_until(ws, "applied")

    renewed = run(expiry(token))
    assert renewed >= before + settings.auth.session_rolling_duration_s - 120
    assert cookie.startswith(f"{settings.auth.status_cookie.key}={renewed};") and "Path=/" in cookie


@db
def test_an_edit_far_from_the_session_expiry_leaves_it_alone(live_on):
    token, project_id = run(signed_in_owner())
    before = run(expiry(token))

    with connect(token, project_id) as ws:
        ws.send_text(json.dumps({"type": "hello", "since": 0, "client_id": "c1"}))
        version = receive_until(ws, "ready")["version"]
        ws.send_text(json.dumps({"type": "ops", "id": "m1", "base": version, "ops": [name("a", "Kept")]}))
        receive_until(ws, "applied")

    assert run(expiry(token)) == before


# ---- a client that stops reading -----------------------------------------------------------------------------------

class StalledSocket(Socket):
    """A websocket whose reader stops once `stalled`: a send then never finishes."""

    def __init__(self):
        super().__init__()
        self.stalled, self.closed = False, None

    async def send_text(self, text):
        if self.stalled:
            await asyncio.Event().wait()
        await super().send_text(text)

    async def close(self, code=1000, reason=None):
        self.closed = code


class StalledFirst(set):
    # A set's order is arbitrary; with the stalled client first, a fan-out that sends in turn makes the others wait.
    def __iter__(self):
        return iter(sorted(super().__iter__(), key=lambda client: not isinstance(client.ws, StalledSocket)))


async def until(check, timeout: float = 5):
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)


@db
@pytest.mark.asyncio
async def test_a_client_that_stops_reading_does_not_hold_up_the_others_and_is_dropped(project, monkeypatch):
    from dsp_decikit.services import live_hub
    from dsp_decikit.services.live_hub import Client, Hub
    from dsp_decikit.services.live_sync import apply_message
    project_id, _ = project
    monkeypatch.setattr(live_hub, "SEND_TIMEOUT_S", 1)
    hub = Hub()
    fast, slow = Client(Socket(), "fast"), Client(StalledSocket(), "slow")
    await hub.join(project_id, fast, 1)
    await hub.join(project_id, slow, 1)
    slow.ws.stalled = True
    hub.rooms[project_id].clients = StalledFirst(hub.rooms[project_id].clients)

    await apply_message(project_id, "c3", "m1", 1, [name("a", "One")])
    hub.pump_soon(project_id)
    await until(lambda: fast.ws.versions() == [2], timeout=0.5)  # well inside the stalled send's timeout

    await until(lambda: slow not in hub.rooms[project_id].clients and slow.ws.closed is not None)
    assert slow.ws.closed == 1013
    await apply_message(project_id, "c3", "m2", 2, [name("b", "Two")])
    hub.pump_soon(project_id)
    await until(lambda: fast.ws.versions() == [2, 3], timeout=0.5)


# ---- why a socket closes, without a database -----------------------------------------------------------------------

@pytest.fixture
def auth(monkeypatch, live_on):
    """A signed-in user with write access, as the socket checks them; change `session` or `access` to end either."""
    from dsp_decikit.api.v1.project import live
    from dsp_decikit.auth.models import SessionInfo
    from dsp_decikit.db.models.projects import AccessMode

    now = int(time.time())
    state = {"session": SessionInfo(created_at=now, last_activity=now, user_id=uuid.uuid4()),
             "access": AccessMode.WRITE, "error": None}

    async def get_session(token):
        if state["error"]:
            raise state["error"]
        return state["session"]

    async def permissions(session, project_id, user):
        if state["access"] is None:
            raise PermissionError("no access")
        return False, state["access"], []

    monkeypatch.setattr(live.sessions, "get_session", get_session)
    monkeypatch.setattr(live, "get_effective_project_permissions", permissions)
    return state


def closed(ws) -> WebSocketDisconnect:
    with pytest.raises(WebSocketDisconnect) as e:
        ws.receive_text()  # once: a socket left open would block here for good
    return e.value


def ping(ws):
    ws.send_text(json.dumps({"type": "ping"}))
    assert json.loads(ws.receive_text()) == {"type": "pong"}


@pytest.mark.parametrize("ended, reason", [("session", "sign in"), ("access", "access")])
def test_a_socket_whose_session_or_access_ends_is_closed_saying_which(auth, ended, reason):
    with connect("token", uuid.uuid4()) as ws:
        ping(ws)
        auth[ended] = None  # signed out on another node, or the share removed
        ws.send_text(json.dumps({"type": "ping"}))
        end = closed(ws)

    assert end.code == 1008 and reason in end.reason


def test_a_socket_opened_without_a_session_is_told_to_sign_in_rather_than_refused_unexplained(auth):
    # A refused handshake reaches the page as a bare network error, and it would reconnect for ever.
    auth["session"] = None

    with connect("token", uuid.uuid4()) as ws:
        end = closed(ws)

    assert end.code == 1008 and "sign in" in end.reason


def test_a_message_that_is_not_json_closes_the_socket_so_the_client_counts_it_as_failing(auth):
    with connect("token", uuid.uuid4()) as ws:
        ws.send_text('{"type": "ops", "id": "m1", "base": 1, "ops": [' * 2000)

        assert closed(ws).code == 1007


def test_a_database_outage_closes_the_socket_as_try_again_so_the_client_retries_without_counting(auth):
    with connect("token", uuid.uuid4()) as ws:
        ping(ws)
        auth["error"] = ConnectionRefusedError("the database is restarting")
        ws.send_text(json.dumps({"type": "ping"}))

        assert closed(ws).code == 1013


def test_an_unexpected_server_error_closes_the_socket_as_a_server_failure_so_the_client_counts_it(auth):
    # Left to the server, the socket would drop without a code, as a lost connection does, which is not counted.
    with connect("token", uuid.uuid4()) as ws:
        ping(ws)
        auth["error"] = RuntimeError("a bug")
        ws.send_text(json.dumps({"type": "ping"}))

        assert closed(ws).code == 1011


def test_a_database_outage_when_the_socket_opens_closes_it_as_try_again(auth):
    auth["error"] = ConnectionRefusedError("the database is restarting")

    with connect("token", uuid.uuid4()) as ws:
        assert closed(ws).code == 1013
