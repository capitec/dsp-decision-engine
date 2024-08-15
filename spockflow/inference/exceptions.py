import typing
from pydantic import ValidationError
from spockflow.exceptions import SpockFlowException
from contextlib import contextmanager

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

    class PydanticErrorDict(typing.TypedDict):
        type: str
        loc: typing.List[str]
        msg: str
        input: typing.Any


class APIException(SpockFlowException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str, status_code: int = None) -> None:
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code

    def additional_content(self) -> dict:
        return {}

    def get_response(self):
        from starlette.responses import JSONResponse

        return JSONResponse(
            {"message": self.detail, **self.additional_content()},
            status_code=self.status_code,
        )

    @staticmethod
    async def handle(request: "Request", exc: "APIException") -> "JSONResponse":
        return exc.get_response()


class InvalidInputError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST


class UnsupportedEncoding(APIException):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def __init__(
        self,
        error_message: str,
        status_code: int = None,
        supported_encodings: typing.Optional[typing.List[str]] = None,
    ) -> None:
        self.supported_encodings = supported_encodings
        super().__init__(error_message, status_code)

    def additional_content(self) -> dict:
        if self.supported_encodings is None:
            return {}
        return {"supported_encodings": self.supported_encodings}

    @classmethod
    def from_content_type(
        cls,
        encoding: str,
        supported_encodings: typing.Optional[typing.List[str]] = None,
    ):
        if supported_encodings is not None:
            additional_info = f" Expected one of {supported_encodings}."
        return cls(
            error_message=f"Unsupported content-type {encoding}." + additional_info,
            supported_encodings=supported_encodings,
        )


class UnsupportedAcceptTypeError(APIException):
    status_code = status.HTTP_406_NOT_ACCEPTABLE

    def __init__(
        self,
        error_message: str,
        status_code: int = None,
        accepted_types: typing.Optional[typing.List[str]] = None,
    ) -> None:
        self.accepted_types = accepted_types
        super().__init__(error_message, status_code)

    def additional_content(self) -> dict:
        if self.accepted_types is None:
            return {}
        return {"accepted_types": self.accepted_types}

    @classmethod
    def from_accept_type(
        cls, accept_type: str, accepted_types: typing.Optional[typing.List[str]] = None
    ):
        if accepted_types is not None:
            additional_info = f" Expected one of {accepted_types}."
        return cls(
            error_message=f"Unsupported accept type {accept_type}." + additional_info,
            accepted_types=accepted_types,
        )


class PydanticFormatError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST

    def __init__(
        self,
        title: str,
        pydantic_errors: typing.List["PydanticErrorDict"],
        status_code: int = None,
    ) -> None:
        self.pydantic_errors = pydantic_errors
        super().__init__(
            f'Could not create model "{title}" from input data.', status_code
        )

    @classmethod
    def from_pydantic(cls, err: "ValidationError", status_code: int = None):
        return cls(
            title=err.title,
            pydantic_errors=err.errors(include_url=False),
            status_code=status_code,
        )

    def additional_content(self) -> dict:
        return {"errors": self.pydantic_errors}


@contextmanager
def reraise_common_input_exceptions(err_callback=None, status_code: int = None):
    try:
        yield
    except ValidationError as e:
        if err_callback is not None:
            err_callback(e)
        raise PydanticFormatError.from_pydantic(e, status_code) from e
    except (ValueError, TypeError, AssertionError) as e:
        if err_callback is not None:
            err_callback(e)
        raise InvalidInputError(str(e), status_code) from e
