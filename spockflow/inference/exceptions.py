import typing

try:
    from starlette import status
except ImportError:
    class StatusCodeWrapper:
        def __getattr__(self, attr: str) -> int:
            try:
                return int(attr.split("_")[1])
            except ValueError as e:
                return -1
    status = StatusCodeWrapper()


if typing.TYPE_CHECKING:
    from starlette.responses import JSONResponse
    from starlette.requests import Request

class APIException(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        detail: str,
        status_code: int = None
    ) -> None:
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code

    def additional_content(self) -> dict:
        return {}

    def get_response(self):
        from starlette.responses import JSONResponse
        return JSONResponse(
            {"message": self.error_message, **self.additional_content()},
            status_code=self.status_code
        )
    
    @staticmethod
    def handle(request: "Request", exc: "APIException") -> "JSONResponse":
        return exc.get_response()

class InvalidInputError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST


class UnsupportedEncoding(APIException):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def __init__(
        self,
        error_message: str,
        status_code: int = None,
        supported_encodings: typing.Optional[typing.List[str]] = None
    ) -> None:
        self.supported_encodings = supported_encodings
        super().__init__(error_message, status_code)

    def additional_content(self) -> dict:
        if self.supported_encodings is None: return {}
        return {"supported_encodings": self.supported_encodings}

    @classmethod
    def from_content_type(cls, encoding: str, supported_encodings: typing.Optional[typing.List[str]] = None):
        if supported_encodings is not None:
            additional_info = f" Expected one of {supported_encodings}."
        return cls(
            error_message=f"Unsupported content-type {encoding}."+additional_info,
            supported_encodings=supported_encodings
        )

class UnsupportedAcceptTypeError(APIException):
    status_code = status.HTTP_406_NOT_ACCEPTABLE

    def __init__(
        self,
        error_message: str,
        status_code: int = None,
        accepted_types: typing.Optional[typing.List[str]] = None
    ) -> None:
        self.accepted_types = accepted_types
        super().__init__(error_message, status_code)

    def additional_content(self) -> dict:
        if self.accepted_types is None: return {}
        return {"accepted_types": self.accepted_types}

    @classmethod
    def from_accept_type(cls, accept_type: str, accepted_types: typing.Optional[typing.List[str]] = None):
        if accepted_types is not None:
            additional_info = f" Expected one of {accepted_types}."
        return cls(
            error_message=f"Unsupported accept type {accept_type}."+additional_info,
            accepted_types=accepted_types
        )