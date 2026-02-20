import typing as t
from .core import BaseSource, DictVersionedSource
from ._ext import register_source
from ped.types import TInputType

class StaticSource(BaseSource[DictVersionedSource]):
    type: t.Literal['static'] = "static"
    values: dict[str, t.Any]
    # We dont want to return none here as it will then always rebuild for static which is very wasteful.
    version: t.Any = "unknown" 

    async def get_version(self, **kwargs) -> t.Optional[t.Any]:
        return self.version
    
    async def get_versioned_source(
        self, 
        inputs: TInputType,
        version: t.Any,
    ) -> DictVersionedSource:
        # Not sure if we should add an assert here for safety
        return DictVersionedSource(version=self.version, values=self.values)


register_source(StaticSource)
