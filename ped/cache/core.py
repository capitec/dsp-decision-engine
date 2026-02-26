import typing as t
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import Field, PrivateAttr
from ped._ext import TypeDiscriminatedBaseModule
from contextlib import asynccontextmanager


class LockingMethod(str, Enum):
    """
    Defines when cache operations should acquire locks to prevent race conditions.
    
    LOCK_ON_BUILD: Acquires lock during entire cache get operation including build.
        - Tradeoff: Highest safety, prevents duplicate builds but may cause longer waits
        - Use when: Build operations are expensive and duplicate builds should be avoided
    
    LOCK_ON_SAVE: Acquires lock only during the store_graph operation.
        - Tradeoff: Allows parallel builds but prevents write conflicts
        - Use when: Build operations are fast but write conflicts need prevention
    
    NO_LOCK: No locking, allows full parallelism.
        - Tradeoff: Fastest execution but may result in duplicate builds and race conditions
        - Use when: Build operations are very fast and occasional duplicate work is acceptable
    """
    LOCK_ON_BUILD = "lock_on_build"
    LOCK_ON_SAVE = "lock_on_save"
    NO_LOCK = "no_lock"


TGraph = t.TypeVar("TGraph")
TSourceVersions = t.TypeVar("TSourceVersions")


class LockManager(t.Generic[TSourceVersions]):
    """Manages per-source-version locks with automatic cleanup."""
    
    def __init__(self):
        self._locks: t.Dict[TSourceVersions, asyncio.Lock] = {}
        self._global_lock: asyncio.Lock = asyncio.Lock()
        self._lock_ref_counts: t.Dict[TSourceVersions, int] = {}

    async def acquire_lock(self, source_versions: TSourceVersions) -> asyncio.Lock:
        """Get or create a lock for the given source versions."""
        async with self._global_lock:
            if source_versions not in self._locks:
                self._locks[source_versions] = asyncio.Lock()
                self._lock_ref_counts[source_versions] = 0
            self._lock_ref_counts[source_versions] += 1
            return self._locks[source_versions]

    async def release_lock(self, source_versions: TSourceVersions):
        """Release and potentially cleanup lock for the given source versions."""
        async with self._global_lock:
            if source_versions in self._lock_ref_counts:
                self._lock_ref_counts[source_versions] -= 1
                if self._lock_ref_counts[source_versions] <= 0:
                    # Clean up the lock when no one is waiting
                    self._locks.pop(source_versions, None)
                    self._lock_ref_counts.pop(source_versions, None)


class BaseCache(TypeDiscriminatedBaseModule, ABC, t.Generic[TGraph, TSourceVersions]):
    """Base class for all cache implementations."""
    
    locking_method: LockingMethod = Field(default=LockingMethod.LOCK_ON_BUILD)
    
    # Lock manager for coordinating access
    _lock_manager: LockManager[TSourceVersions] = PrivateAttr(default_factory=LockManager)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip validation for Pydantic's generic parameterizations
        if hasattr(cls, '__pydantic_generic_metadata__'):
            return
        if 'type' not in cls.__annotations__:
            raise TypeError(f"{cls.__name__} must define a 'type' class variable")

    @abstractmethod
    def has(self, source_versions: TSourceVersions) -> bool:
        """Check if the cache contains an entry for the given source versions."""
        ...

    @abstractmethod
    def get(self, source_versions: TSourceVersions) -> TGraph:
        """Retrieve a cached graph for the given source versions."""
        ...

    @abstractmethod
    async def _put(self, source_versions: TSourceVersions, graph: TGraph) -> None:
        """Internal method to store a graph in the cache. Implemented by subclasses."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached entries."""
        ...

    async def put(self, source_versions: TSourceVersions, graph: TGraph) -> None:
        """
        Store a graph in the cache for the given source versions.
        
        This method handles locking based on the locking_method configuration.
        """
        if self.locking_method == LockingMethod.LOCK_ON_SAVE:
            lock = await self._lock_manager.acquire_lock(source_versions)
            async with lock:
                try:
                    await self._put(source_versions, graph)
                finally:
                    await self._lock_manager.release_lock(source_versions)
        else:
            # For NO_LOCK and LOCK_ON_BUILD, store directly
            # LOCK_ON_BUILD already has the lock acquired in the context
            await self._put(source_versions, graph)

    @asynccontextmanager
    async def lock(self, source_versions: TSourceVersions):
        """
        Context manager for acquiring locks based on the locking method.
        
        For LOCK_ON_BUILD: acquires the lock for the entire context
        For LOCK_ON_SAVE and NO_LOCK: does nothing (lock is handled in put)
        """
        if self.locking_method == LockingMethod.LOCK_ON_BUILD:
            lock = await self._lock_manager.acquire_lock(source_versions)
            async with lock:
                try:
                    yield
                finally:
                    await self._lock_manager.release_lock(source_versions)
        else:
            yield