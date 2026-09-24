from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Callable

from pydantic_core import to_json

from decider.engine.debug import COMMAND, EVENT, Pause, Session
from decider.engine.debug.hot import ModuleWatcher
from decider.exceptions import wrap_import_errors

with wrap_import_errors("starlette"):
    from starlette.applications import Starlette
    from starlette.concurrency import run_in_threadpool
    from starlette.routing import WebSocketRoute
    from starlette.websockets import WebSocket, WebSocketDisconnect


def session_app(factory: Callable[[], Session], watch: str | None = None, interval: float = 0.5) -> Starlette:
    """A starlette app serving a debug `Session` over a websocket at `/`, one fresh session per connection.

    The client sends commands as JSON (`{"kind": "break_at", "target": "ratio"}`,
    `{"kind": "resume"}`, `{"kind": "set", "name": "x", "value": 1.0}`, ...).
    After each one the server sends every new event as its own JSON message.
    Events carry summaries only; a full value is sent only when asked for with
    `{"kind": "value", "spec": "name"}` or `"name@path"`, and
    `{"kind": "structure"}` answers `{"kind": "structure", "steps": [...]}`
    (see `Session.structure`). `pause` takes effect at once, even while a
    `resume` is running. A message that can't be run gets
    `{"kind": "rejected", "message": ...}` and the connection stays open.

    `watch="module:attr"` follows the pipeline's source files with a
    `ModuleWatcher`, checked every `interval` seconds between commands: a
    saved edit reloads the session (`edited` events, then `paused`), and one
    that doesn't import or wire sends `reload_failed` and keeps the old one.
    The factory should look the pipeline up when called, so a connection
    opened after an edit starts from the edited code.

    Example::

        from decider.serving.session_ws import session_app
        app = session_app(lambda: pipeline.session(frame, params))
        # uvicorn.run(app, port=8000)  ->  ws://localhost:8000/

        # reload on every save under credit/
        target = "credit.pipeline:pipeline"
        app = session_app(lambda: pkgutil.resolve_name(target).session(frame), watch=target)
    """
    async def endpoint(ws: WebSocket) -> None:
        await ws.accept()
        session = await run_in_threadpool(factory)
        watcher = None if watch is None else await run_in_threadpool(ModuleWatcher, watch)
        queue: asyncio.Queue = asyncio.Queue()
        sent = 0
        pending = False

        def reload() -> None:
            nonlocal pending
            pending = False
            # Logged as ReloadFailed; it answers nothing the client sent, so it isn't rejected.
            with contextlib.suppress(Exception):
                session._reload_from(watcher.poll)

        async def tick() -> None:
            # Through the queue, so a reload never runs while a command is using the session.
            nonlocal pending
            while True:
                await asyncio.sleep(interval)
                if not pending:
                    pending = True
                    queue.put_nowait(reload)

        async def push() -> None:
            nonlocal sent
            new, sent = session.events[sent:], len(session.events)
            for event in new:
                await ws.send_text(EVENT.dump_json(event).decode())

        async def read() -> None:
            # Runs alongside the command loop so a pause reaches a resume that is still running.
            try:
                while True:
                    text = await ws.receive_text()
                    try:
                        msg = json.loads(text)
                        kind = msg.get("kind") if isinstance(msg, dict) else None
                        if kind == "value":
                            spec = str(msg["spec"])
                            queue.put_nowait(lambda: _value(session, spec))
                        elif kind == "structure":
                            queue.put_nowait(lambda: {"kind": "structure", "steps": session.structure()})
                        elif isinstance(cmd := COMMAND.validate_python(msg), Pause):
                            session.pause()
                        else:
                            queue.put_nowait(cmd)
                    except (ValueError, KeyError) as e:
                        await _reject(ws, e)
            except WebSocketDisconnect:
                session.pause()
                queue.put_nowait(None)

        tasks = [asyncio.create_task(read())] + ([] if watcher is None else [asyncio.create_task(tick())])
        try:
            with contextlib.suppress(WebSocketDisconnect):
                await push()
                # ponytail: commands and requests run one at a time per connection; a value asked for
                # mid-resume waits for it, since reading state while the worker writes it isn't safe.
                while (item := await queue.get()) is not None:
                    error = reply = None
                    try:
                        # A request is a function returning its reply (or nothing); a command is data.
                        if callable(item):
                            reply = await run_in_threadpool(item)
                        else:
                            await run_in_threadpool(session.apply, item)
                    except Exception as e:
                        error = e
                    await push()
                    if reply is not None:
                        await ws.send_text(to_json(reply).decode())
                    if error is not None:
                        await _reject(ws, error)
        finally:
            for task in tasks:
                task.cancel()

    return Starlette(routes=[WebSocketRoute("/", endpoint)])


def _value(session: Session, spec: str) -> dict:
    series = session.value(spec)
    return {"kind": "value", "spec": spec, "dtype": str(series.dtype), "values": series.to_list()}


async def _reject(ws: WebSocket, error: Exception) -> None:
    await ws.send_text(json.dumps({"kind": "rejected", "message": f"{type(error).__name__}: {error}"}))
