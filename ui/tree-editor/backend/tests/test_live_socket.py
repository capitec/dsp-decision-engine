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

from .test_live_sync import RULE_A, RULE_B, flat, name

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
