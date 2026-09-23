import typing as t
from contextlib import asynccontextmanager
from decider.exceptions import DeciderError, wrap_import_errors

with wrap_import_errors("starlette"):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

from decider.serving.handler import RequestHandler, construct_handler_from_settings
from .core import error_response, parse_content_headers, ready, _INITIALIZING


async def decider_error_handler(request: Request, exc: DeciderError) -> Response:
    status_code, body, media_type = error_response(exc)
    return Response(content=body, status_code=status_code, media_type=media_type)


async def predict(request: Request) -> Response:
    handler = request.app.state.handler
    if not ready(handler):
        return Response(content=_INITIALIZING, status_code=503, media_type="application/json")
    content_type, accept = parse_content_headers(request.headers)
    result = await handler.process_fn(await request.body(), accept, content_type)
    return Response(content=result.content, media_type=result.media_type)


async def ping(request: Request) -> Response:
    if not ready(request.app.state.handler):
        return Response(content=_INITIALIZING, status_code=503, media_type="application/json")
    return Response(status_code=200)


@asynccontextmanager
async def lifespan(app: "Starlette"):
    if app.state.handler is None:
        handler = construct_handler_from_settings()
        await handler.init_fn()
        app.state.handler = handler
    yield
    await app.state.handler.shutdown_fn()


def create_app(handler: t.Optional[RequestHandler] = None) -> "Starlette":
    """The SageMaker app: `POST /invocations` and `GET /ping` (200 once a version is active).

    Without a `handler`, startup builds one from settings and activates the
    store's latest version.

    Example::

        uvicorn.run("decider.serving.servers.starlette:create_app", factory=True)
    """
    app = Starlette(
        routes=[
            Route("/invocations", predict, methods=["POST"]),
            Route("/ping", ping, methods=["GET"]),
        ],
        lifespan=lifespan,
        exception_handlers={DeciderError: decider_error_handler},
    )
    app.state.handler = handler
    return app
