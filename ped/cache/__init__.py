from ._ext import register_cache, CacheProvider
from .core import BaseCache, LockingMethod, LockManager
# Preload some of the caches that aren't expensive to import
from . import (
    lru,
    none
)