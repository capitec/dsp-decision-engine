import typing as t
from collections import OrderedDict
from .core import BaseCache
from ._ext import register_cache


class LRUCache(BaseCache[t.Any, t.Any]):
    """
    Least Recently Used cache implementation with configurable locking behavior.
    
    This cache maintains entries in order of access, evicting the least recently used
    items when the cache reaches its maximum size.
    """
    
    type: t.Literal['lru']
    max_size: int = 1
    
    def __init__(self, **data):
        super().__init__(**data)
        self._cache: OrderedDict[t.Any, t.Any] = OrderedDict()

    def has(self, source_versions: t.Any) -> bool:
        """Check if the cache contains an entry for the given source versions."""
        return source_versions in self._cache

    def get(self, source_versions: t.Any) -> t.Any:
        """
        Retrieve a cached graph for the given source versions.
        
        Accessing an item moves it to the end (most recently used).
        """
        if source_versions not in self._cache:
            raise KeyError(f"Source versions {source_versions} not found in cache")
        
        # Move to end (most recently used)
        graph = self._cache[source_versions]
        self._cache.move_to_end(source_versions)
        return graph

    async def _put(self, source_versions: t.Any, graph: t.Any) -> None:
        """
        Internal method to store a graph in the cache.
        
        Updates existing entries or adds new ones, maintaining LRU order.
        Evicts least recently used items when max_size is reached.
        """
        # Update or add the entry
        if source_versions in self._cache:
            # Move existing entry to end
            self._cache.move_to_end(source_versions)
        else:
            # Add new entry
            if len(self._cache) >= self.max_size:
                # Remove least recently used (first item)
                self._cache.popitem(last=False)
        
        self._cache[source_versions] = graph


    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def evict(self, source_versions: t.Any) -> bool:
        """
        Manually evict a specific entry from the cache.
        
        Returns True if the entry was found and removed, False otherwise.
        """
        if source_versions in self._cache:
            del self._cache[source_versions]
            return True
        return False

    def size(self) -> int:
        """Get the current number of entries in the cache."""
        return len(self._cache)

    def keys(self) -> t.List[t.Any]:
        """Get all cached source versions, ordered from least to most recently used."""
        return list(self._cache.keys())


# Register the LRUCache with the extendable model system
register_cache(LRUCache)