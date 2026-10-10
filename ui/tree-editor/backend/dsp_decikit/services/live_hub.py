"""Fan-out of project versions to the live sockets of this process.

Every process LISTENs on the `project_ops` channel; a notification makes it send the new `project_ops` rows of that
project, in version order, to each of its sockets on the project. After a reconnect of the listener it refetches
for every project it serves, so a notification missed while it was down only delays delivery.
"""

import asyncio
import contextlib
import json
import logging
import multiprocessing
import time
import typing as t
import uuid
from concurrent.futures import ProcessPoolExecutor

from fastapi import WebSocket
from sqlalchemy import func, select

from dsp_decikit.db.models.projects import Project, ProjectOp
from dsp_decikit.db.session import db_session, engine
from dsp_decikit.services.documents import validation_errors
from dsp_decikit.services.live_sync import CHANNEL

logger = logging.getLogger("uvicorn.error").getChild("live")


class Client:
    def __init__(self, ws: WebSocket, client_id: str):
        self.ws, self.client_id, self.last = ws, client_id, 0


class Room:
    def __init__(self):
        self.clients: t.Set[Client] = set()
        self.lock = asyncio.Lock()
        self.dirty = False
        self.pumping: t.Optional[asyncio.Task] = None
        self.validating: t.Optional[asyncio.Task] = None
        self.validation: t.Optional[str] = None
        self.validated = 0


class Hub:
    def __init__(self):
        self.rooms: t.Dict[uuid.UUID, Room] = {}
        self.listener: t.Optional[asyncio.Task] = None
        self.validator: t.Optional[ProcessPoolExecutor] = None

    async def join(self, project_id: uuid.UUID, client: Client, since: int) -> None:
        """Bring `client` up to date from version `since` (a snapshot when the log does not reach back), then
        send it `ready` and keep it in the project's fan-out."""
        room = self.rooms.setdefault(project_id, Room())
        async with room.lock:
            async with db_session() as session:
                version, doc = (await session.execute(select(Project.version, Project.project_data)
                                                      .where(Project.id == project_id))).one()
                logged = await session.scalar(select(func.count()).select_from(ProjectOp).where(
                    ProjectOp.project_id == project_id, ProjectOp.version > since))
            if since == 0 or since > version or logged != version - since:
                await client.ws.send_text(json.dumps({"type": "snapshot", "version": version, "doc": doc}))
                client.last = version
            else:
                client.last = since
            room.clients.add(client)
            await self._send_rows(project_id, room)
            await client.ws.send_text(json.dumps({"type": "ready", "version": client.last}))
            if room.validation and room.validated == client.last:
                await client.ws.send_text(room.validation)
        self._validate_soon(project_id, room)

    def leave(self, project_id: uuid.UUID, client: Client) -> None:
        # The room stays: a client joining right now is added to this room object once it is caught up.
        # ponytail: one small Room per project opened on this process; evict idle rooms under their lock if it grows
        room = self.rooms.get(project_id)
        if room is not None:
            room.clients.discard(client)

    def pump_soon(self, project_id: uuid.UUID) -> None:
        """Send this project's new versions to its sockets here (coalesced: one fetch serves many calls)."""
        room = self.rooms.get(project_id)
        if room is None:
            return
        room.dirty = True
        if room.pumping is None or room.pumping.done():
            room.pumping = asyncio.create_task(self._pump(project_id, room))

    async def _pump(self, project_id: uuid.UUID, room: Room) -> None:
        while room.dirty:
            room.dirty = False
            try:
                async with room.lock:
                    await self._send_rows(project_id, room)
            except Exception as e:
                # No other notification may come for this version, so the pump itself tries again.
                logger.warning("live fan-out project=%s failed, retrying: %s", project_id, e)
                room.dirty = True
                await asyncio.sleep(1)
        self._validate_soon(project_id, room)

    async def _send_rows(self, project_id: uuid.UUID, room: Room) -> None:
        if not room.clients:
            return
        async with db_session() as session:
            rows = (await session.execute(select(ProjectOp).where(
                ProjectOp.project_id == project_id, ProjectOp.version > min(c.last for c in room.clients))
                .order_by(ProjectOp.version))).scalars().all()
        for row in rows:
            text = json.dumps({"type": "applied", "version": row.version, "client_id": row.client_id,
                               "msg_id": row.msg_id, "ops": row.ops})
            for client in list(room.clients):
                if client.last < row.version:
                    # ponytail: one slow socket holds up the project's fan-out; add a per-client queue if it shows
                    if await _send(client, text):
                        client.last = row.version
                    else:
                        room.clients.discard(client)

    def _validate_soon(self, project_id: uuid.UUID, room: Room) -> None:
        if room.clients and (room.validating is None or room.validating.done()):
            room.validating = asyncio.create_task(self._validate(project_id, room))

    async def _validate(self, project_id: uuid.UUID, room: Room) -> None:
        # Validating a large document takes a few hundred ms, so it runs off the write path, once per version that
        # is still the latest when the previous run ends, in a process of its own: in a thread it holds the GIL
        # and every write on this node waits behind it.
        while room.clients:
            async with db_session() as session:
                version, doc = (await session.execute(select(Project.version, Project.project_data)
                                                      .where(Project.id == project_id))).one()
            if version == room.validated:
                return
            started = time.perf_counter()
            errors = await asyncio.get_running_loop().run_in_executor(self.validator, validation_errors, doc)
            logger.info("live validation project=%s version=%d errors=%d ms=%.1f", project_id, version, len(errors),
                        (time.perf_counter() - started) * 1e3)
            room.validation = json.dumps({"type": "validation", "version": version, "errors": errors})
            room.validated = version
            for client in list(room.clients):
                await _send(client, room.validation)

    def start(self) -> None:
        self.validator = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        self.listener = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self.validator:
            self.validator.shutdown(cancel_futures=True)
        if self.listener:
            self.listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.listener

    def _notified(self, _conn, _pid, _channel, payload: str) -> None:
        with contextlib.suppress(ValueError, KeyError):
            self.pump_soon(uuid.UUID(json.loads(payload)["p"]))

    async def _listen(self) -> None:
        while True:
            conn = None
            try:
                conn = await engine.connect()
                raw = (await conn.get_raw_connection()).driver_connection
                await raw.add_listener(CHANNEL, self._notified)
                logger.info("live listener connected")
                for project_id in list(self.rooms):
                    self.pump_soon(project_id)
                while True:
                    # A dead connection is noticed here, not by asyncpg, when Postgres goes away without a goodbye.
                    await asyncio.wait_for(raw.execute("SELECT 1"), timeout=5)
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("live listener lost: %s", e)
            finally:
                if conn is not None:
                    # A LISTENing connection must never go back to the pool.
                    with contextlib.suppress(Exception):
                        await conn.invalidate()
                    with contextlib.suppress(Exception):
                        await conn.close()
            await asyncio.sleep(1)


async def _send(client: Client, text: str) -> bool:
    try:
        await client.ws.send_text(text)
        return True
    except Exception:
        return False


hub = Hub()
