import typing as t
from pydantic import Field
from .core import BaseSource
from ._ext import register_source

class RequestSource(BaseSource):
    type: t.Literal['request'] = "request"
    base_key: str
    defaults: dict[str, t.Any] = Field(default_factory=dict)

    def requires_refresh(self, **kwargs) -> bool:
        return True

    def get(self, key: str, request: t.Dict[str, t.Any], **kwargs) -> t.Any:
        params = request.get(self.base_key, {})
        if key in params:
            return params[key]
        # Cant refactor to return params.get(key, self.defaults[key])
        # because if the key is missing from defaults and is in the request then that will fail.
        return self.defaults[key]

register_source(RequestSource)
