import typing as t
from pydantic import BaseModel
from abc import ABC, abstractmethod
from ..sources import ParameterSource
from ..types import VersionedValue

class BaseParameterCache(BaseModel,ABC):
    parameter_source: ParameterSource

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    @abstractmethod
    def requires_refresh(
        self, 
        requested_version: t.Any = None, 
        **kwargs
    ) -> bool:
        ...

    @abstractmethod
    def get(
        self, 
        key: str, 
        requested_version: t.Any = None,
        **kwargs
    ) -> VersionedValue:
        ...
