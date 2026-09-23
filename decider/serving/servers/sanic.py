import typing as t
from decider.exceptions import DeciderError, wrap_import_errors

with wrap_import_errors("sanic"):
    from sanic import Sanic
    from sanic.request import Request
    from sanic.response import HTTPResponse, raw

from decider.serving.handler import RequestHandler, construct_handler_from_settings
from .core import error_response, parse_content_headers, ready, _INITIALIZING


async def predict(request: Request) -> HTTPResponse:
    handler = request.app.ctx.handler
    if not ready(handler):
        return raw(_INITIALIZING, status=503, content_type="application/json")
    content_type, accept = parse_content_headers(request.headers)
    result = await handler.process_fn(request.body, accept, content_type)
    return raw(result.content, status=200, content_type=result.media_type)


async def ping(request: Request) -> HTTPResponse:
    if not ready(request.app.ctx.handler):
        return raw(_INITIALIZING, status=503, content_type="application/json")
    return raw(b"", status=200)


def create_app(name: str = "decider", handler: t.Optional[RequestHandler] = None) -> Sanic:
    """The SageMaker app on sanic: `POST /invocations` and `GET /ping` (200 once a version is active).

    Call it inside each worker process, e.g. through sanic's `AppLoader(factory=create_app)`.
    """
    app = Sanic(name)
    app.ctx.handler = handler
    app.add_route(predict, "/invocations", methods=["POST"])
    app.add_route(ping, "/ping", methods=["GET"])

    @app.exception(DeciderError)
    async def decider_error_handler(_request: Request, exc: DeciderError) -> HTTPResponse:
        status_code, body, media_type = error_response(exc)
        return raw(body, status=status_code, content_type=media_type)

    @app.before_server_start
    async def startup(app_) -> None:
        if app_.ctx.handler is None:
            handler_ = construct_handler_from_settings()
            await handler_.init_fn()
            app_.ctx.handler = handler_

    @app.after_server_stop
    async def shutdown(app_) -> None:
        if app_.ctx.handler is not None:
            await app_.ctx.handler.shutdown_fn()

    return app
