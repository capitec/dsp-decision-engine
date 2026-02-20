import typing as t
from pydantic import Field

from ped.types import TInputType
from ..types import TVersionType
from .core import BaseSource, DictVersionedSource
from ._ext import register_source

class InputsSource(BaseSource[DictVersionedSource]):
    """
    This module gets the parameters from the inputs per request. Providing a version allows the system to not have to rebuild the models for each request but enables caching.
    """
    type: t.Literal['inputs'] = "inputs"
    base_key: str
    version_key: t.Optional[str] = None
    defaults: dict[str, t.Any] = Field(default_factory=dict)
    # Doesn't make sense to cache this one as all the information is in the request.
    cache_kwargs: t.Optional[t.Dict[str, t.Any]] = None

    async def get_version(
        self, 
        inputs: TInputType, 
    ) -> t.Optional[TVersionType]:
        if self.version_key is None: return None # None tells the model to do a refresh regardless
        # This helps the model cache not need to reload the model if it doesnt have to.
        # Note for none it is important to ensure that the get doesnt return different parameter versions when the requested version is none
        current_version = inputs.get(self.base_key, {}).get(self.version_key)
        return current_version
    
    async def get_versioned_source(
        self, 
        inputs: TInputType,
        version: TVersionType,
    ) -> DictVersionedSource:
        # Not sure if we should add an assert here for safety
        parameters = inputs.get(self.base_key, {})
        version_from_inputs = parameters.get(self.version_key) if self.version_key else None
        return DictVersionedSource(
            values=self.defaults | inputs.get(self.base_key, {}),
            version=version_from_inputs
        )


register_source(InputsSource)
