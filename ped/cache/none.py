import typing as t
from .core import BaseCache
from ._ext import register_cache


class NoCache(BaseCache[t.Any, t.Any]):
    """
    A cache implementation that doesn't cache anything.
    
    This is useful for disabling caching entirely while maintaining
    the same interface as other cache implementations.
    """
    
    type: t.Literal['none']

    def has(self, source_versions: t.Any) -> bool:
        """Always returns False since nothing is cached."""
        return False

    def get(self, source_versions: t.Any) -> t.Any:
        """Always raises KeyError since nothing is cached."""
        raise KeyError(f"Source versions {source_versions} not found in cache (NoCache)")

    async def _put(self, source_versions: t.Any, graph: t.Any) -> None:
        """Does nothing since this cache doesn't store anything."""
        pass

    def clear(self) -> None:
        """Does nothing since there's nothing to clear."""
        pass


# Register the NoCache with the extendable model system
register_cache(NoCache)