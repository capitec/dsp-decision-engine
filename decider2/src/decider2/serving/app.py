"""`app(pipeline, *, mode="sealed"|"live")` — doc 02 §3.6.

Returns an ASGI app either way: a real `starlette.applications.Starlette`
when starlette is importable, or a small hand-rolled ASGI callable (pure
stdlib, no third-party ASGI framework) when it is not. **starlette stays an
optional dependency** — nothing in this package fails to import without it,
and nothing in `graph/`, `compile/`, `params.py` or `runtime/` imports this
package at all (doc 02 §3.6 rule 1; there is no lint enforcing that here,
but the import graph already keeps it true — `runtime/serve.py`'s own
docstring states the same discipline).

Both branches share one `Dispatcher` (`dispatch.py`) and one route table
(`ROUTES`), so the two backends cannot silently drift apart on what an
endpoint does — only on how a request/response crosses the wire.

Running the returned app over an actual socket (with or without starlette)
is `serving/server.py`'s job, not this module's — `uvicorn` is checked for
the same way starlette is here, and a tiny stdlib `http.server` bridge
covers the case where neither is installed (the `decider2 serve` CLI is the
only thing that needs a bridge; the ASGI app itself is oblivious to which
one is running it).
"""
from __future__ import annotations

import importlib.util
import json
from typing import Any

from decider2.serving.dispatch import ROUTES, BadRequest, Dispatcher

__all__ = ["app"]


def _starlette_available() -> bool:
    return importlib.util.find_spec("starlette") is not None


def app(pipeline: Any, *, mode: str = "sealed", warm: bool = True) -> Any:
    """Build the ASGI app for `pipeline`. `mode` is doc 08 §4.1's deployment
    mode, forwarded straight to `pipeline.serve()` — reported back verbatim
    on `GET /health`.

    `warm=True` (the default) calls `handle.warm()` here, before this
    function ever returns an app for a server to bind and accept
    connections on — doc 05 §8's "no compilation after warm-up" only holds
    if warm-up finishes before the first request can arrive, and `GET
    /ping` (`serving/dispatch.py`) also refuses to answer 200 until
    `handle.is_warm`, as a second, independent guard against a request
    racing it. Pass `warm=False` only for a caller that wants to call
    `handle.warm()` itself with a specific `shared=` (a table's own rows,
    doc 08 §3.4) before serving traffic.
    """
    handle = pipeline.serve(mode=mode)
    if warm:
        handle.warm()
    dispatcher = Dispatcher(handle)
    if _starlette_available():
        return _starlette_app(dispatcher)
    return _stdlib_asgi_app(dispatcher)


# ---------------------------------------------------------------------------
# starlette backend
# ---------------------------------------------------------------------------


def _starlette_app(dispatcher: Dispatcher) -> Any:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    def _make_endpoint(handler):
        async def endpoint(request: Request) -> JSONResponse:
            body: Any = None
            if request.method == "POST":
                raw = await request.body()
                if raw:
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        return JSONResponse({"detail": f"invalid JSON body: {exc}"}, status_code=400)
            try:
                status, payload = handler(dispatcher, body)
            except BadRequest as exc:
                return JSONResponse({"detail": str(exc), "errors": exc.errors}, status_code=400)
            except Exception as exc:  # noqa: BLE001 — never leak a raw traceback over HTTP
                return JSONResponse(
                    {"detail": "internal error", "type": type(exc).__name__}, status_code=500
                )
            return JSONResponse(payload, status_code=status)

        return endpoint

    routes = [
        Route(path, _make_endpoint(handler), methods=[method])
        for (method, path), handler in ROUTES.items()
    ]
    return Starlette(routes=routes)


# ---------------------------------------------------------------------------
# stdlib fallback backend — a real ASGI app, just not built with a framework
# ---------------------------------------------------------------------------


async def _send_json(send, status: int, payload: Any) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"application/json")],
    })
    await send({"type": "http.response.body", "body": body})


def _stdlib_asgi_app(dispatcher: Dispatcher):
    async def asgi_app(scope, receive, send):
        if scope["type"] != "http":
            return
        method = scope["method"]
        path = scope["path"]

        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        raw = b"".join(chunks)

        body: Any = None
        if raw:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError as exc:
                await _send_json(send, 400, {"detail": f"invalid JSON body: {exc}"})
                return

        handler = ROUTES.get((method, path))
        if handler is None:
            await _send_json(send, 404, {"detail": f"no route for {method} {path}"})
            return

        try:
            status, payload = handler(dispatcher, body)
        except BadRequest as exc:
            await _send_json(send, 400, {"detail": str(exc), "errors": exc.errors})
            return
        except Exception as exc:  # noqa: BLE001 — never leak a raw traceback over HTTP
            await _send_json(send, 500, {"detail": "internal error", "type": type(exc).__name__})
            return

        await _send_json(send, status, payload)

    return asgi_app
