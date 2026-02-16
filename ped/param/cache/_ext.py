"""
This module enables config sources to be extended by external packages without creating hard dependencies on those packages. It does this by maintaining a global union type of all registered sources, which can be extended by calling the `register_source` function with a new source type. The `ParameterSource` model is then rebuilt to include the new source type in its union.
"""
from .core import BaseCache
from ped._ext import create_extendable_model

CacheProvider, register_cache_provider = create_extendable_model(
    BaseCache, 
    discriminator_field="type", 
    model_name="ParameterCache"
)
