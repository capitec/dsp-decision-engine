import typing as t
from pydantic import BaseModel
from abc import ABC, abstractmethod
from ..sources import ParameterSource


_UNSET_ = object()

class VersionedValue(t.NamedTuple):
    version: t.Any
    value: t.Any


class BaseCache(ABC):
    @abstractmethod
    def requires_refresh(self, requested_version: t.Any = None, **kwargs) -> bool:
        ...

    @abstractmethod
    def get(self, key: str, requested_version: t.Any = None) -> t.Any:
        ...

class DefaultParameterSourceCache(BaseCache):
    def __init__(self, parameter_source: ParameterSource):
        self.parameter_source = parameter_source
        self.cache = {}
        self.curr_version = _UNSET_

    def requires_refresh(self, requested_version: t.Any = _UNSET_, **kwargs) -> bool:
        if requested_version is not _UNSET_ and requested_version != self.curr_version:
            return True
        if self.curr_version is _UNSET_:
            return True
        return self.parameter_source.requires_refresh(self.curr_version, **kwargs)
    
    def get(self, key: str, requested_version: t.Any = _UNSET_, **kwargs) -> t.Any:
        if not self.requires_refresh(requested_version=requested_version) and key in self.cache:
            return self.cache[key]
        value, cache_key = self.parameter_source.get(key, requested_version=requested_version)
        if cache_key != self.curr_version:
            self.curr_version = cache_key
            self.cache = {}
        self.cache[key] = value
        return value
