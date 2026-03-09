import typing as t
from dataclasses import dataclass
from pydantic import Field
from abc import ABC, abstractmethod

from ped._ext import TypeDiscriminatedBaseModule
from ped.types import TInputType
from ped.param.types import TVersionType, TParamValue


class VersionedSource(ABC):
    # The source should have a way to determine the version to return values of
    def get(
        self, 
        key: str,
        inputs: TInputType,
        args
    ) -> TParamValue:
        ...


@dataclass
class DictVersionedSource(VersionedSource):
    values: dict[str, TParamValue]
    version: TVersionType = "unknown"

    def get(
        self, 
        key: str,
        inputs: TInputType,
        args
    ) -> TParamValue:
        return self.values[key]

TVersionedSource = t.TypeVar("TVersionedSource", bound=VersionedSource)


class BaseSource(TypeDiscriminatedBaseModule, ABC, t.Generic[TVersionedSource]):
    cache_kwargs: t.Optional[t.Dict[str, t.Any]] = Field(default_factory=lambda: {"type": "default"})

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     # Skip validation for Pydantic's generic parameterizations
    #     if hasattr(cls, '__pydantic_generic_metadata__'):
    #         return
    #     if 'type' not in cls.__annotations__:
    #         raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    @abstractmethod
    async def get_version(self, 
        inputs: t.Mapping[str, t.Any],
    ) -> t.Optional[TVersionType]:
        ...

    @abstractmethod
    async def get_versioned_source(
        self, 
        version: TVersionType,
        inputs: t.Mapping[str, t.Any],
    ) -> TVersionedSource:
        ...

