import typing as t
from pydantic import PrivateAttr

from ..types import VersionedValue
from .core import BaseParameterCache
from ._ext import register_cache_provider


class DefaultParameterCache(BaseParameterCache):
    type: t.Literal["default"] = "default"
    _cache: t.Dict[str, t.Any] = PrivateAttr(default_factory=dict)
    _curr_version: t.Any = PrivateAttr(default=None)

    def requires_refresh(
        self, 
        requested_version: t.Any = None, 
        **kwargs
    ) -> bool:
        if self._curr_version is None:
            return True
        if requested_version is not None:
            return requested_version != self._curr_version
        return self.parameter_source.root.requires_refresh(
            requested_version=requested_version,
            curr_version=self._curr_version, 
            **kwargs
        )
    
    def get(
        self, 
        key: str, 
        requested_version: t.Any = None, 
        **kwargs
    ) -> t.Any:
        if not self.requires_refresh(requested_version=requested_version) and key in self._cache:
            return self._cache[key]
        value, cache_key = self.parameter_source.root.get(key, requested_version=requested_version)
        if cache_key != self._curr_version:
            self._curr_version = cache_key
            self._cache = {}
        self._cache[key] = value
        return VersionedValue(version=self._curr_version, value=value)

register_cache_provider(DefaultParameterCache)