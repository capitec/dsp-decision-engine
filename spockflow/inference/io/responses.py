# Might be a good idea to wrap these but can always be wrapped if the backend needs to change
import typing
from dataclasses import dataclass

try:
    from starlette import responses
except ImportError:
    responses = None


@dataclass
class Response:
    content: typing.Any = None
    status_code: int = 200
    headers: typing.Optional[typing.Mapping[str, str]] = None
    media_type: typing.Optional[str] = None

    @property
    def _api_response_cls(self):
        return responses.Response

    def to_api(self):
        # TODO there must be a better way of doing this
        assert (
            responses is not None
        ), "Requires starlette to be installed. Please install with pip install SpockFlow[webapp]"
        return self._api_response_cls(
            content=self.content,
            status_code=self.status_code,
            headers=self.headers,
            media_type=self.media_type,
        )


class JSONResponse(Response):
    @property
    def _api_response_cls(self):
        return responses.JSONResponse


class PlainTextResponse(Response):
    @property
    def _api_response_cls(self):
        return responses.PlainTextResponse


@dataclass
class CSVResponse(Response):
    media_type: str = "text/csv"
