import typing as t
from pydantic import Field

from ..types import VersionedValue
from .core import BaseSource
from ._ext import register_source

class RequestSource(BaseSource):
    type: t.Literal['request'] = "request"
    base_key: str
    version_key: t.Optional[str] = None
    defaults: dict[str, t.Any] = Field(default_factory=dict)
    # Doesn't make sense to cache this one as all the information is in the request.
    cache_kwargs: t.Optional[t.Dict[str, t.Any]] = None

    def requires_refresh(
        self, 
        curr_version: t.Any, 
        request: t.Dict[str, t.Any], 
        **kwargs
    ) -> bool:
        if self.version_key is None: return True
        # This helps the model cache not need to reload the model if it doesnt have to.
        current_version = request.get(self.base_key, {}).get(self.version_key)
        if current_version is None: return True
        return current_version != curr_version

    def get(self, key: str, request: t.Dict[str, t.Any], **kwargs) -> VersionedValue:
        params = request.get(self.base_key, {})
        curr_version = params.get(self.version_key) if self.version_key else None
        if key in params:
            return VersionedValue(version=curr_version, value=params[key])
        # Cant refactor to return params.get(key, self.defaults[key])
        # because if the key is missing from defaults and is in the request then that will fail.
        return VersionedValue(version=curr_version, value=self.defaults[key])

register_source(RequestSource)
