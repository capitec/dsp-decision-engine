import typing as t
from pydantic import BaseModel
from abc import ABC, abstractmethod


class VersionedValue(t.NamedTuple):
    version: t.Any
    value: t.Any


class BaseSource(BaseModel, ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    @abstractmethod
    def requires_refresh(self, curr_version:t.Any, **kwargs) -> bool:
        ...

    @abstractmethod
    def get(self, key: str, requested_version: t.Any = None, **kwargs) -> t.Any:
        ...
