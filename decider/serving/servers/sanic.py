import typing as t
from decider.exceptions import DeciderError, wrap_import_errors

with wrap_import_errors("sanic"):
    from sanic import Sanic
    from sanic.request import Request
    from sanic.response import HTTPResponse, raw

from decider.serving.handler import RequestHandler, construct_handler_from_settings
from .core import error_response, parse_content_headers, _INITIALIZING


handler: t.Optional[RequestHandler] = None


async def predict(request: Request) -> HTTPResponse:
    if handler is None:
        return raw(_INITIALIZING, status=503, content_type="application/json")
    content_type, accept = parse_content_headers(request.headers)
    result = handler.process_fn(request.body, accept, content_type)
    return raw(result.content, status=200, content_type=result.media_type)


async def ping(_request: Request) -> HTTPResponse:
    if handler is None:
        return raw(_INITIALIZING, status=503, content_type="application/json")
    return raw(b"", status=200)


def create_app(name: str = "decider") -> Sanic:
    app = Sanic(name)
    app.add_route(predict, "/predict", methods=["POST"])
    app.add_route(ping, "/ping", methods=["GET"])

    @app.exception(DeciderError)
    async def decider_error_handler(_request: Request, exc: DeciderError) -> HTTPResponse:
        status_code, body, media_type = error_response(exc)
        return raw(body, status=status_code, content_type=media_type)

    @app.before_server_start
    async def startup(_app, _loop) -> None:
        global handler
        _handler = construct_handler_from_settings()
        await _handler.init_fn()
        handler = _handler

    @app.after_server_stop
    async def shutdown(_app, _loop) -> None:
        if handler is not None:
            await handler.shutdown_fn()

    return app


app = create_app()
