"""
This module enables cache implementations to be extended by external packages without creating hard dependencies on those packages. It does this by maintaining a global union type of all registered cache types, which can be extended by calling the `register_cache` function with a new cache type. The `CacheProvider` model is then rebuilt to include the new cache type in its union.
"""
from .core import BaseCache
from ped._ext import create_extendable_model

CacheProvider, register_cache = create_extendable_model(
    BaseCache, 
    discriminator_field="type",
    model_name="CacheProvider"
)


__all__ = ['CacheProvider', 'register_cache']