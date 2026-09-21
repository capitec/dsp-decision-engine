import typing as t
from contextlib import asynccontextmanager
from decider.exceptions import DeciderError, wrap_import_errors

with wrap_import_errors("starlette"):
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

from decider.serving.handler import RequestHandler, construct_handler_from_settings
from .core import error_response, parse_content_headers, _INITIALIZING


handler: t.Optional[RequestHandler] = None


async def decider_error_handler(request: Request, exc: DeciderError) -> Response:
    status_code, body, media_type = error_response(exc)
    return Response(content=body, status_code=status_code, media_type=media_type)


async def predict(request: Request) -> Response:
    if handler is None:
        return Response(content=_INITIALIZING, status_code=503, media_type="application/json")
    content_type, accept = parse_content_headers(request.headers)
    result = await handler.process_fn(await request.body(), accept, content_type)
    return Response(content=result.content, media_type=result.media_type)


async def ping(_request: Request) -> Response:
    if handler is None:
        return Response(content=_INITIALIZING, status_code=503, media_type="application/json")
    return Response(status_code=200)


@asynccontextmanager
async def lifespan(app: "Starlette"):
    from decider.initialization import initialize_decider
    global handler
    initialize_decider()
    _handler = construct_handler_from_settings()
    await _handler.init_fn()
    handler = _handler
    yield
    if handler is not None:
        await handler.shutdown_fn()


def create_app() -> "Starlette":
    return Starlette(
        routes=[
            Route("/invocations", predict, methods=["POST"]),
            Route("/ping", ping, methods=["GET"]),
        ],
        lifespan=lifespan,
        exception_handlers={DeciderError: decider_error_handler},
    )


app: t.Optional["Starlette"] = None


def get_app() -> "Starlette":
    global app
    if app is None:
        app = create_app()
    return app
