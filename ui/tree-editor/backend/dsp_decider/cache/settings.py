from pydantic import BaseModel, Field
import typing as t
from typing import Annotated
from datetime import datetime, timezone, timedelta


class InMemoryCacheWrapper:
    _sentinel = object()

    def __init__(self, cache):
        self.cache = cache

    async def put(self, k: str, v, expire_seconds: int | None = None):
        self.cache[k] = v

    async def delete(self, k: str):
        try:
            del self.cache[k]
        except KeyError:
            pass  # Key doesn't exist, which is fine for delete

    async def get(self, k: str, default=_sentinel):
        if k in self.cache:
            return self.cache[k]
        if default is self._sentinel:
            raise KeyError(k)
        return default


class DiskCacheWrapper:
    _sentinel = object()

    def __init__(self, cache):
        self.cache = cache

    async def put(self, k: str, v, expire_seconds: int | None = None):
        if expire_seconds:
            self.cache.set(k, v, expire=expire_seconds)
        else:
            self.cache[k] = v

    async def delete(self, k: str):
        try:
            del self.cache[k]
        except KeyError:
            pass  # Key doesn't exist, which is fine for delete

    async def get(self, k: str, default=_sentinel):
        if k in self.cache:
            return self.cache[k]
        if default is self._sentinel:
            raise KeyError(k)
        return default


class RedisCacheWrapper:
    _sentinel = object()

    def __init__(self, redis_client):
        self.redis = redis_client

    async def put(self, k: str, v, expire_seconds: int | None = None):
        import json

        value = json.dumps(v)
        if expire_seconds:
            await self.redis.setex(k, expire_seconds, value)
        else:
            await self.redis.set(k, value)

    async def delete(self, k: str):
        await self.redis.delete(k)

    async def get(self, k: str, default=_sentinel):
        import json

        value = await self.redis.get(k)
        if value is not None:
            return json.loads(value)
        if default is self._sentinel:
            raise KeyError(k)
        return default


class InMemoryProvider(BaseModel):
    provider: t.Literal["in-memory"] = "in-memory"
    max_size: int = 0
    ttl: float = 0

    def create(self):
        if self.ttl <= 0 and self.max_size <= 0:
            return InMemoryCacheWrapper(dict())
        if self.ttl <= 0:
            from cachetools import LRUCache

            return InMemoryCacheWrapper(LRUCache(maxsize=self.max_size))
        from cachetools import TTLCache

        return InMemoryCacheWrapper(TTLCache(maxsize=self.max_size, ttl=self.ttl))


class DiskCacheProvider(BaseModel):
    provider: t.Literal["diskcache"] = "diskcache"
    directory: str = "/tmp/diskcache"
    size_limit: int = 1024 * 1024 * 100  # 100MB

    def create(self):
        import diskcache

        cache = diskcache.Cache(self.directory, size_limit=self.size_limit)
        return DiskCacheWrapper(cache)


class RedisProvider(BaseModel):
    provider: t.Literal["redis"] = "redis"
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None

    def create(self):
        import redis.asyncio as redis

        client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True,
        )
        return RedisCacheWrapper(client)


class PersistedInMemoryCacheWrapper:
    """Cache that combines any base cache provider with PostgreSQL persistence"""

    _sentinel = object()

    def __init__(self, base_cache_provider):
        self.base_cache = base_cache_provider
        self._cleanup_threshold = 1000  # Clean up expired entries every N operations
        self._operation_count = 0

    async def _cleanup_expired_entries(self):
        """Remove expired entries from PostgreSQL"""
        from dsp_decider.db.session import db_session
        from dsp_decider.db.models.cache import CacheEntry
        from sqlalchemy import select, delete

        async with db_session() as session:
            # Delete expired entries
            now = datetime.now(timezone.utc)
            delete_stmt = delete(CacheEntry).where(
                CacheEntry.expires_at.isnot(None), CacheEntry.expires_at < now
            )
            await session.execute(delete_stmt)
            await session.commit()

    async def _maybe_cleanup(self):
        """Periodically clean up expired entries"""
        self._operation_count += 1
        if self._operation_count >= self._cleanup_threshold:
            self._operation_count = 0
            await self._cleanup_expired_entries()

    async def put(self, k: str, v, expire_seconds: int | None = None):
        """Store in both base cache and PostgreSQL"""
        from dsp_decider.db.session import db_session
        from dsp_decider.db.models.cache import CacheEntry
        from sqlalchemy.dialects.postgresql import insert

        # Store in base cache layer
        await self.base_cache.put(k, v, expire_seconds)

        # Calculate expiry time
        expires_at = None
        if expire_seconds:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expire_seconds)

        now = datetime.now(timezone.utc)

        # Store in PostgreSQL using upsert (INSERT ... ON CONFLICT DO UPDATE)
        async with db_session() as session:
            stmt = insert(CacheEntry).values(
                key=k, value=v, created_at=now, expires_at=expires_at
            )
            # On conflict (duplicate key), update the existing entry
            stmt = stmt.on_conflict_do_update(
                index_elements=["key"],
                set_={
                    "value": stmt.excluded.value,
                    "created_at": stmt.excluded.created_at,
                    "expires_at": stmt.excluded.expires_at,
                },
            )
            await session.execute(stmt)
            await session.commit()

        # Periodic cleanup
        await self._maybe_cleanup()

    async def delete(self, k: str):
        """Delete from both base cache and PostgreSQL"""
        from dsp_decider.db.session import db_session
        from dsp_decider.db.models.cache import CacheEntry
        from sqlalchemy import delete as sql_delete

        # Delete from base cache layer (ignore if not found)
        try:
            await self.base_cache.delete(k)
        except KeyError:
            pass  # Key doesn't exist in base cache, which is fine

        # Delete from PostgreSQL
        async with db_session() as session:
            delete_stmt = sql_delete(CacheEntry).where(CacheEntry.key == k)
            await session.execute(delete_stmt)
            await session.commit()

    async def get(self, k: str, default=_sentinel):
        """Get from base cache first, fallback to PostgreSQL"""
        # Try base cache first
        try:
            return await self.base_cache.get(k)
        except KeyError:
            pass  # Not in base cache, check database

        # Fallback to PostgreSQL
        from dsp_decider.db.session import db_session
        from dsp_decider.db.models.cache import CacheEntry
        from sqlalchemy import select

        async with db_session() as session:
            stmt = select(CacheEntry).where(CacheEntry.key == k)
            result = await session.execute(stmt)
            cache_entry = result.scalar_one_or_none()

            if cache_entry:
                # Check if expired
                if cache_entry.expires_at and cache_entry.expires_at < datetime.now(
                    timezone.utc
                ):
                    # Entry is expired, delete it
                    await session.delete(cache_entry)
                    await session.commit()
                else:
                    # Store in base cache for faster future access
                    value = cache_entry.value
                    await self.base_cache.put(
                        k, value, None
                    )  # Store without expiry in base cache
                    return value

        # Not found in either cache
        if default is self._sentinel:
            raise KeyError(k)
        return default


class PersistedInMemoryProvider(BaseModel):
    """Configurable base cache with PostgreSQL persistence"""

    provider: t.Literal["persisted-cache"] = "persisted-cache"
    cache: t.Union[InMemoryProvider, DiskCacheProvider] = Field(
        discriminator="provider"
    )

    def create(self):
        # Create the base cache provider
        base_cache = self.cache.create()
        return PersistedInMemoryCacheWrapper(base_cache)


TCacheProviders = Annotated[
    t.Union[
        InMemoryProvider, DiskCacheProvider, RedisProvider, PersistedInMemoryProvider
    ],
    Field(discriminator="provider"),
]
