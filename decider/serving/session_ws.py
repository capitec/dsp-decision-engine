from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Callable

from pydantic_core import to_json

from decider.engine.debug import COMMAND, EVENT, Pause, Session
from decider.exceptions import wrap_import_errors

with wrap_import_errors("starlette"):
    from starlette.applications import Starlette
    from starlette.concurrency import run_in_threadpool
    from starlette.routing import WebSocketRoute
    from starlette.websockets import WebSocket, WebSocketDisconnect


def session_app(factory: Callable[[], Session]) -> Starlette:
    """A starlette app serving a debug `Session` over a websocket at `/`, one fresh session per connection.

    The client sends commands as JSON (`{"kind": "break_at", "target": "ratio"}`,
    `{"kind": "resume"}`, `{"kind": "set", "name": "x", "value": 1.0}`, ...).
    After each one the server sends every new event as its own JSON message.
    Events carry summaries only; a full value is sent only when asked for with
    `{"kind": "value", "spec": "name"}` or `"name@path"`. `pause` takes effect
    at once, even while a `resume` is running. A message that can't be run
    gets `{"kind": "rejected", "message": ...}` and the connection stays open.

    Example::

        from decider.serving.session_ws import session_app
        app = session_app(lambda: pipeline.session(frame, params))
        # uvicorn.run(app, port=8000)  ->  ws://localhost:8000/
    """
    async def endpoint(ws: WebSocket) -> None:
        await ws.accept()
        session = await run_in_threadpool(factory)
        queue: asyncio.Queue = asyncio.Queue()
        sent = 0

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
                        if isinstance(msg, dict) and msg.get("kind") == "value":
                            queue.put_nowait(str(msg["spec"]))
                        elif isinstance(cmd := COMMAND.validate_python(msg), Pause):
                            session.pause()
                        else:
                            queue.put_nowait(cmd)
                    except (ValueError, KeyError) as e:
                        await _reject(ws, e)
            except WebSocketDisconnect:
                session.pause()
                queue.put_nowait(None)

        reader = asyncio.create_task(read())
        try:
            with contextlib.suppress(WebSocketDisconnect):
                await push()
                # ponytail: commands and value requests run one at a time per connection; a value asked for
                # mid-resume waits for it, since reading state while the worker writes it isn't safe.
                while (item := await queue.get()) is not None:
                    error = None
                    try:
                        if isinstance(item, str):
                            series = session.value(item)
                            await ws.send_text(to_json({"kind": "value", "spec": item, "dtype": str(series.dtype),
                                                        "values": series.to_list()}).decode())
                        else:
                            await run_in_threadpool(session.apply, item)
                    except Exception as e:
                        error = e
                    await push()
                    if error is not None:
                        await _reject(ws, error)
        finally:
            reader.cancel()

    return Starlette(routes=[WebSocketRoute("/", endpoint)])


async def _reject(ws: WebSocket, error: Exception) -> None:
    await ws.send_text(json.dumps({"kind": "rejected", "message": f"{type(error).__name__}: {error}"}))
