"""`WS /api/v1/projects/{id}/live`: live editing of one project.

Client -> server: `hello {since, client_id}` (catch up from version `since`, 0 for a snapshot), then
`ops {id, base, ops}`. Server -> client: `snapshot {version, doc}`, `applied {version, client_id, msg_id, ops}` for
every version in order (the author's ack included), `ready {version}` once caught up, `dropped {msg_id, version, ops}`
and `rejected {msg_id, reason}` to the author only, `validation {version, errors}` for the latest version.
READ access connects and receives; WRITE sends ops. The session and access are checked again on every message and
every 15 s.
"""

import asyncio
import json
import time
import typing as t
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from dsp_decikit.auth.authorization import handle_bearer_token_auth, handle_session_auth
from dsp_decikit.auth.models import SessionInfo
from dsp_decikit.cache import get_cache
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


async def _user(ws: WebSocket) -> t.Optional[SessionInfo]:
    try:
        user = await handle_session_auth(ws.cookies.get(settings.auth.cookie.key), get_cache())
    except HTTPException:
        return None
    scheme, _, token = ws.headers.get("authorization", "").partition(" ")
    user = user or (await handle_bearer_token_auth(token) if scheme.lower() == "bearer" else None)
    if user is None or user.expires_at <= datetime.now(timezone.utc).timestamp():
        return None
    return user


async def _access(project_id: uuid.UUID, ws: WebSocket) -> t.Optional[AccessMode]:
    user = await _user(ws)
    if user is None:
        return None
    async with db_session() as session:
        try:
            return (await get_effective_project_permissions(session, project_id, user))[1]
        except (PermissionError, ValueError):
            return None


def _same_origin(ws: WebSocket) -> bool:
    # Browsers send cookies on cross-site websocket handshakes, so a page elsewhere must not be able to connect.
    origin = ws.headers.get("origin")
    return origin is None or urlparse(origin).netloc == ws.headers.get("host")


@router.websocket("/projects/{project_id}/live")
async def live(ws: WebSocket, project_id: uuid.UUID):
    if not settings.live_collab or not _same_origin(ws) or await _access(project_id, ws) is None:
        await ws.close(POLICY_VIOLATION)
        return
    await ws.accept()
    client = None
    try:
        while True:
            try:
                text = await asyncio.wait_for(ws.receive_text(), RECHECK_S)
            except asyncio.TimeoutError:
                text = ""  # a reader never sends: check its session and access all the same
            started = time.perf_counter()
            if len(text) > MAX_MESSAGE_BYTES:
                await ws.close(1009)
                return
            mode = await _access(project_id, ws)
            if mode is None:
                await ws.close(POLICY_VIOLATION, "session ended or access removed")
                return
            if not text:
                continue
            authorised = time.perf_counter()
            try:
                message = json.loads(text)
                kind = message.get("type")
            except (ValueError, AttributeError):
                await ws.send_text(json.dumps({"type": "error", "reason": "not a JSON object"}))
                continue
            writer = mode in (AccessMode.WRITE, AccessMode.ADMIN)
            if kind == "hello":
                since, client_id = message.get("since"), message.get("client_id")
                if not isinstance(since, int) or since < 0 or not isinstance(client_id, str) or not client_id[:64]:
                    await ws.send_text(json.dumps({"type": "error", "reason": "hello needs since and client_id"}))
                    continue
                if client is not None:
                    hub.leave(project_id, client)
                if writer:
                    await stamp_project(project_id)
                client = Client(ws, client_id[:64])
                await hub.join(project_id, client, since)
            elif kind == "ops":
                await _ops(ws, project_id, client, message, writer)
                logger.info("live message project=%s auth_ms=%.1f total_ms=%.1f bytes=%d", project_id,
                            (authorised - started) * 1e3, (time.perf_counter() - started) * 1e3, len(text))
            else:
                await ws.send_text(json.dumps({"type": "error", "reason": f"unknown message type {kind!r}"}))
    except WebSocketDisconnect:
        pass
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
        applied = await apply_message(project_id, client.client_id, msg_id, message["base"], message["ops"])
    except Rejected as e:
        await ws.send_text(json.dumps({"type": "rejected", "msg_id": msg_id, "reason": str(e)}))
        return
    if applied.dropped:
        await ws.send_text(json.dumps({"type": "dropped", "msg_id": msg_id, "version": applied.version,
                                       "ops": applied.dropped}))
    hub.pump_soon(project_id)
