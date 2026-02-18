import typing as t
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from ped.param.types import VersionedValue

class BaseSource(BaseModel, ABC):
    cache_kwargs: t.Optional[t.Dict[str, t.Any]] = Field(default_factory=lambda: {"type": "default"})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    @abstractmethod
    def requires_refresh(self, 
        curr_version: t.Any, 
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

