import typing as t
from functools import lru_cache
from .keys import CacheKeys


class CacheProtocol(t.Protocol):
    async def put(self, k: str, v, expire_seconds: int | None = None): ...
    async def delete(self, k: str): ...
    async def get(self, k: str, default=None): ...


@lru_cache(maxsize=1)
def get_cache() -> CacheProtocol:
    from dsp_decikit.settings import settings

    return settings.cache.create()
