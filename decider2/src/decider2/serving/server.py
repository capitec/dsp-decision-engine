"""Running the ASGI app `serving.app.app()` returns, over a real socket.

Prefers `uvicorn` when it is importable; falls back to a small synchronous
bridge over the stdlib `http.server`, so `decider2 serve` works with zero
extra dependencies installed (the task's own requirement — serving must be
optional, and this repo's venv has neither `starlette` nor `uvicorn`
installed as this was written). Either way the ASGI app itself
(`serving.app.app`) is unchanged: this module only supplies the transport,
which is exactly the seam doc 02 §3.6 asks for — swap this file for a
`gunicorn`/`hypercorn` invocation and nothing else in `serving/` has to
change.

The stdlib bridge is single-process, threaded, and meant for exactly what
the task asks for — "run each of them as an endpoint and play with
parameters" — not for production load; `GET /health`'s GIL report is what
tells you whether *the pipeline* is safe under real concurrency (doc 00
§2c), independent of which of these two runs it.
"""
from __future__ import annotations

import asyncio
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["run"]


def _uvicorn_available() -> bool:
    return importlib.util.find_spec("uvicorn") is not None


def run(asgi_app: Any, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    if _uvicorn_available():
        import uvicorn

        uvicorn.run(asgi_app, host=host, port=port, log_level="info")
        return
    _run_stdlib(asgi_app, host=host, port=port)


async def _call_asgi(
    asgi_app: Any, method: str, path: str, body: bytes
) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    """Drive one request through an ASGI app with no ASGI server involved —
    the same technique `tests/test_serving.py` uses to drive it without
    starlette's `TestClient`, factored here because the stdlib runner needs
    it for real rather than just for a test."""
    path, _, query_string = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "path": path, "raw_path": path.encode(),
        "query_string": query_string.encode(), "headers": [],
    }
    response: dict[str, Any] = {}
    body_chunks: list[bytes] = []
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            body_chunks.append(message.get("body", b""))

    await asgi_app(scope, receive, send)
    return response.get("status", 500), response.get("headers", []), b"".join(body_chunks)


def _run_stdlib(asgi_app: Any, *, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _dispatch(self) -> None:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b""
            status, headers, payload = asyncio.run(
                _call_asgi(asgi_app, self.command, self.path, body)
            )
            self.send_response(status)
            for k, v in headers:
                name = k.decode() if isinstance(k, bytes) else k
                value = v.decode() if isinstance(v, bytes) else v
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = _dispatch
        do_POST = _dispatch

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter than the default
            pass

    print(
        f"decider2 serving on http://{host}:{port} — using the stdlib http.server "
        "fallback (install starlette + uvicorn for a production-grade server; "
        "see decider2/serving/app.py)."
    )
    ThreadingHTTPServer((host, port), Handler).serve_forever()
