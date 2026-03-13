from starlette.applications import Starlette
from starlette.responses import Response, JSONResponse
from starlette.routing import Route
from starlette.requests import Request
from starlette import status
from starlette.exceptions import HTTPException
from ped.exceptions import PEDError


def ping(request: Request):
    return Response(status_code=status.HTTP_200_OK)


def startup():
    from ped import initialize_ped
    initialize_ped()

def shutdown():
    pass

routes = [
    Route("/ping", ping, methods=["GET"]),
    Route("/invocations", predict, methods=["POST"]),
]


async def not_found(request: Request, exc: HTTPException):
    return JSONResponse(content={"message": exc.detail}, status_code=exc.status_code)


async def server_error(request: Request, exc: HTTPException):
    return JSONResponse(
        content={"message": getattr(exc, "detail", str(exc))},
        status_code=getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR),
    )


def ped_error_handler(request: Request, exc: PEDError):
    return JSONResponse(
        content=exc.get_response_body().dict(),
        status_code=exc.get_status_code(),
    )

exception_handlers = {
    PEDError: ped_error_handler,
    404: not_found,
    500: server_error,
}

app = Starlette(
    routes=routes,
    on_startup=[startup],
    on_shutdown=[],
    exception_handlers=exception_handlers,
)
