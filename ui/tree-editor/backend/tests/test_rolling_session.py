import logging
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from dsp_decikit.asgi import app
from dsp_decikit.auth import middleware
from dsp_decikit.auth.models import SessionInfo
from dsp_decikit.settings import settings

TOKEN = "the-raw-session-token-that-must-stay-secret"


@pytest.fixture
def client():
    return TestClient(app, cookies={"session_id": TOKEN}, follow_redirects=False)


def stored(monkeypatch, session):
    async def get_session(token):
        return session if token == TOKEN else None
    monkeypatch.setattr(middleware, "get_session", get_session)


def test_an_expired_session_is_redirected_without_logging_its_token(client, monkeypatch, caplog):
    stored(monkeypatch, SessionInfo(created_at=0, last_activity=0, user_id=uuid.uuid4()))

    with caplog.at_level(logging.DEBUG):
        reply = client.get("/api/no-such-route")

    assert reply.status_code == 302
    assert TOKEN[:8] not in caplog.text

