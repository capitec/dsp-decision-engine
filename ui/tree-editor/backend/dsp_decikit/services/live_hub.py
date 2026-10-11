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
from concurrent.futures.process import BrokenProcessPool

from fastapi import WebSocket
from sqlalchemy import func, select

from dsp_decikit.db.models.projects import Project, ProjectOp
from dsp_decikit.db.session import db_session, engine
from dsp_decikit.services.documents import validation_errors
from dsp_decikit.services.live_sync import CHANNEL

logger = logging.getLogger("uvicorn.error").getChild("live")
VALIDATE_TIMEOUT_S = 30
# A client that cannot take a message in this long is dropped; it catches up when it reconnects.
SEND_TIMEOUT_S = 5
TRY_AGAIN = 1013


class Client:
    def __init__(self, ws: WebSocket, client_id: str):
        self.ws, self.client_id, self.last, self.alive = ws, client_id, 0, True


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
        self.validating = asyncio.Lock()

    async def join(self, project_id: uuid.UUID, client: Client, since: int) -> None:
        """Bring `client` up to date from version `since` (a snapshot when the log does not reach back), then
        send it `ready` and keep it in the project's fan-out."""
        room = self.rooms.setdefault(project_id, Room())
        async with room.lock:
            async with db_session() as session:
                version, doc = (await session.execute(select(Project.version, Project.project_data)
                                                      .where(Project.id == project_id))).one()
                # Up to `version` only: a write committed since then may have trimmed a version this client needs.
                logged = await session.scalar(select(func.count()).select_from(ProjectOp).where(
                    ProjectOp.project_id == project_id, ProjectOp.version.between(since + 1, version)))
            if since == 0 or since > version or logged != version - since:
                await _send(client, json.dumps({"type": "snapshot", "version": version, "doc": doc}))
                client.last = version
            else:
                client.last = since
            room.clients.add(client)
            await self._send_rows(project_id, room)
            await _send(client, json.dumps({"type": "ready", "version": client.last}))
            if room.validation and room.validated == client.last:
                await _send(client, room.validation)
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
            # Not the snapshot column: a whole document every 100 versions that the fan-out never sends.
            rows = (await session.execute(select(ProjectOp.version, ProjectOp.client_id, ProjectOp.msg_id, ProjectOp.ops)
                .where(ProjectOp.project_id == project_id, ProjectOp.version > min(c.last for c in room.clients))
                .order_by(ProjectOp.version))).all()
        for row in rows:
            text = json.dumps({"type": "applied", "version": row.version, "client_id": row.client_id,
                               "msg_id": row.msg_id, "ops": row.ops})
            behind = [client for client in room.clients if client.last < row.version]
            # All at once, so a client that stopped reading holds up the others for one SEND_TIMEOUT_S at most.
            for client, sent in zip(behind, await asyncio.gather(*(_send(client, text) for client in behind))):
                if sent:
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
            try:
                errors = await self.validate(doc)
            except (asyncio.TimeoutError, BrokenProcessPool):
                # The process hung or was killed (out of memory, say); a new one validates the next version.
                logger.warning("live validator failed on project=%s version=%d", project_id, version)
                return
            logger.info("live validation project=%s version=%d errors=%d ms=%.1f", project_id, version, len(errors),
                        (time.perf_counter() - started) * 1e3)
            room.validation = json.dumps({"type": "validation", "version": version, "errors": errors})
            room.validated = version
            for client in list(room.clients):
                await _send(client, room.validation)

    async def validate(self, doc: dict) -> list:
        """The validation errors of `doc`, from the validator process. Raises asyncio.TimeoutError when it does not
        answer within VALIDATE_TIMEOUT_S, or BrokenProcessPool when it died; either way a new process takes over."""
        # One at a time, so the timeout counts this document's validation, not a queue behind others.
        async with self.validating:
            pool = self.validator
            try:
                return await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(pool, validation_errors, doc), VALIDATE_TIMEOUT_S)
            except (asyncio.TimeoutError, BrokenProcessPool):
                if pool is not None and self.validator is pool:
                    self.validator = _validator()
                    _kill(pool)
                raise

    def start(self) -> None:
        self.validator = _validator()
        self.listener = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        # Killed, not shut down: shutdown() waits for a hung validation on the event loop, so its timeout never fires.
        pool, self.validator = self.validator, None
        if pool:
            _kill(pool)
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


def _validator() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))


def _kill(pool: ProcessPoolExecutor) -> None:
    # shutdown() leaves a hung worker running. ponytail: private attribute; Python 3.14 has kill_workers()
    for process in list((pool._processes or {}).values()):
        process.kill()
    pool.shutdown(wait=False, cancel_futures=True)


async def _send(client: Client, text: str) -> bool:
    if not client.alive:
        return False
    try:
        await asyncio.wait_for(client.ws.send_text(text), SEND_TIMEOUT_S)
        return True
    except Exception as e:
        client.alive = False
        if isinstance(e, asyncio.TimeoutError):
            # In a task of its own: closing waits on the stalled reader too.
            task = asyncio.create_task(_close(client.ws))
            _closing.add(task)
            task.add_done_callback(_closing.discard)
        return False


_closing: t.Set[asyncio.Task] = set()


async def _close(ws: WebSocket) -> None:
    with contextlib.suppress(Exception):
        await ws.close(TRY_AGAIN)


hub = Hub()
