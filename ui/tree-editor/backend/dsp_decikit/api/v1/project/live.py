"""`WS /api/v1/projects/{id}/live`: live editing of one project.

Client -> server: `hello {since, client_id}` (catch up from version `since`, 0 for a snapshot), then
`ops {id, base, ops, offline?}` (`offline: true` for edits made while disconnected: merged whole or rejected whole),
and `ping` (answered `pong`). Server -> client: `snapshot {version, doc}`, `applied {version, client_id, msg_id, ops}`
for every version in order (the author's ack included), `ready {version}` once caught up, `dropped {msg_id, version,
ops}` and `rejected {msg_id, reason, conflicts}` to the author only, `validation {version, errors}` for the latest
version, and `session {cookie}` when an edit renewed the session (the status cookie, for the page to set).
READ access connects and receives; WRITE sends ops. The session and access are checked again on every message and
every 15 s.
"""

import asyncio
import contextlib
import json
import time
import typing as t
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import Response, WebSocket, WebSocketDisconnect
from sqlalchemy import exc

from dsp_decikit.auth import sessions
from dsp_decikit.auth.authorization import handle_bearer_token_auth
from dsp_decikit.auth.middleware import set_status_cookie
from dsp_decikit.auth.models import SessionInfo
from dsp_decikit.db.models.projects import AccessMode
from dsp_decikit.db.session import db_session
from dsp_decikit.services.folder_permissions import get_effective_project_permissions
from dsp_decikit.services.live_hub import Client, hub
from dsp_decikit.services.live_sync import Rejected, apply_message, check_message, logger, stamp_project
from dsp_decikit.settings import settings
from .routes import router

MAX_MESSAGE_BYTES = 4 * 1024 * 1024
RECHECK_S = 15
POLICY_VIOLATION = 1008
NOT_JSON = 1007
TRY_AGAIN = 1013
SERVER_ERROR = 1011
SIGNED_OUT = "signed out: sign in again to keep editing"
NO_ACCESS = "your access to this project was removed"


async def _user(ws: WebSocket) -> t.Tuple[t.Optional[str], t.Optional[SessionInfo]]:
    """The signed-in user and their session token (None for an API token), or (None, None)."""
    token = ws.cookies.get(settings.auth.cookie.key)
    user = await sessions.get_session(token) if token else None
    if user is None:
        token = None
        scheme, _, bearer = ws.headers.get("authorization", "").partition(" ")
        user = await handle_bearer_token_auth(bearer) if scheme.lower() == "bearer" else None
    if user is None or user.expires_at <= datetime.now(timezone.utc).timestamp():
        return None, None
    return token, user


async def _access(project_id: uuid.UUID, ws: WebSocket) -> t.Tuple[t.Optional[AccessMode], t.Optional[str],
                                                                 t.Optional[SessionInfo]]:
    """The user's access to the project, their session token and the user; the socket is closed, saying why, and
    the access is None when they have none."""
    token, user = await _user(ws)
    mode = None
    if user is not None:
        async with db_session() as session:
            with contextlib.suppress(PermissionError, ValueError):
                mode = (await get_effective_project_permissions(session, project_id, user))[1]
    if mode is None:
        await ws.close(POLICY_VIOLATION, NO_ACCESS if user else SIGNED_OUT)
    return mode, token, user


def _transient(e: Exception) -> bool:
    # A database or network outage: the same message goes through once it is back, so the client resends it
    # without counting it as one the server fails on every time.
    if isinstance(e, (OSError, exc.OperationalError, exc.InterfaceError, exc.TimeoutError)):
        return True
    state = str(getattr(getattr(e, "orig", None), "sqlstate", None) or "")
    return getattr(e, "connection_invalidated", False) or state[:2] in ("08", "53", "57")


def _same_origin(ws: WebSocket) -> bool:
    # Browsers send cookies on cross-site websocket handshakes, so a page elsewhere must not be able to connect.
    origin = ws.headers.get("origin")
    return origin is None or urlparse(origin).netloc == ws.headers.get("host")


@router.websocket("/projects/{project_id}/live")
async def live(ws: WebSocket, project_id: uuid.UUID):
    if not settings.live_collab or not _same_origin(ws):
        await ws.close(POLICY_VIOLATION)
        return
    # Accepted before the session is checked: a refused handshake reaches the page as a bare network error, and it
    # would reconnect for ever instead of telling the user why.
    await ws.accept()
    client = None
    try:
        if (await _access(project_id, ws))[0] is None:
            return
        while True:
            try:
                text = await asyncio.wait_for(ws.receive_text(), RECHECK_S)
            except asyncio.TimeoutError:
                text = ""  # a reader never sends: check its session and access all the same
            started = time.perf_counter()
            if len(text) > MAX_MESSAGE_BYTES:
                await ws.close(1009)
                return
            mode, token, user = await _access(project_id, ws)
            if mode is None:
                return
            if not text:
                continue
            authorised = time.perf_counter()
            try:
                message = json.loads(text)
                kind = message.get("type")
            except (ValueError, AttributeError, RecursionError):
                # Closed rather than answered: the client counts the drop, and gives up on a message after a few.
                await ws.close(NOT_JSON, "not a JSON object")
                return
            writer = mode in (AccessMode.WRITE, AccessMode.ADMIN)
            if kind == "hello":
                since, client_id = message.get("since"), message.get("client_id")
                if (not isinstance(since, int) or since < 0 or not isinstance(client_id, str) or not client_id[:64]
                        or not client_id[:64].isprintable()):
                    await ws.send_text(json.dumps({"type": "error", "reason": "hello needs since and client_id"}))
                    continue
                if client is not None:
                    hub.leave(project_id, client)
                if writer:
                    await stamp_project(project_id)
                client = Client(ws, client_id[:64])
                await hub.join(project_id, client, since)
            elif kind == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
            elif kind == "ops":
                if token and await sessions.renew_if_due(token, user):
                    # Editing live is activity, as a request is; the socket cannot set the status cookie itself.
                    renewed = Response()
                    set_status_cookie(renewed, user.expires_at)
                    await ws.send_text(json.dumps({"type": "session", "cookie": renewed.headers["set-cookie"]}))
                await _ops(ws, project_id, client, message, writer)
                logger.info("live message project=%s auth_ms=%.1f total_ms=%.1f bytes=%d", project_id,
                            (authorised - started) * 1e3, (time.perf_counter() - started) * 1e3, len(text))
            else:
                await ws.send_text(json.dumps({"type": "error", "reason": f"unknown message type {kind!r}"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        # Closed with a code, as uvicorn would drop it without one, like a lost connection: the client counts a
        # server error against the message it sent, and resends as is after a transient one.
        transient = _transient(e)
        if transient:
            logger.warning("live socket project=%s closed on a transient error: %s", project_id, e)
        else:
            logger.exception("live socket project=%s failed", project_id)
        with contextlib.suppress(Exception):
            await ws.close(TRY_AGAIN if transient else SERVER_ERROR)
    finally:
        if client is not None:
            hub.leave(project_id, client)


async def _ops(ws: WebSocket, project_id: uuid.UUID, client: t.Optional[Client], message: dict, writer: bool):
    msg_id = message.get("id")
    try:
        if client is None:
            raise Rejected("say hello first")
        check_message(message)
        if not writer:
            raise Rejected("read-only access: your edits are not saved")
        applied = await apply_message(project_id, client.client_id, msg_id, message["base"], message["ops"],
                                      offline=message.get("offline") is True, executor=hub.validator,
                                      parts=message.get("parts", []))
    except Rejected as e:
        await ws.send_text(json.dumps({"type": "rejected", "msg_id": msg_id, "reason": str(e),
                                       "conflicts": e.conflicts}))
        return
    if applied.dropped:
        await ws.send_text(json.dumps({"type": "dropped", "msg_id": msg_id, "version": applied.version,
                                       "ops": applied.dropped}))
    hub.pump_soon(project_id)
