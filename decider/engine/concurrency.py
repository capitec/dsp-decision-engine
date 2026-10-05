from __future__ import annotations

import threading
from typing import Any, Callable

_MISSING = object()


class Cache(dict):
    """A dict that `once` fills: lock-free reads, one lock for the builds."""

    __slots__ = ("lock",)

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.RLock()


def once(cache: Cache, key: Any, build: Callable[[], Any]) -> Any:
    """The value under `key`, built by `build()` the first time and never again, however many threads ask.

    A hit takes no lock. `build` runs under the cache's lock, so it may take
    time; the value is published with a single insert, fully made.
    """
    value = cache.get(key, _MISSING)
    if value is _MISSING:
        with cache.lock:
            value = cache.get(key, _MISSING)
            if value is _MISSING:
                value = cache[key] = build()
    return value


# Reusable scratch lives per thread, so concurrent calls never share a buffer.
_local = threading.local()


def checkout(pool: str, key: Any, make: Callable[[Any], Any]) -> Any:
    # Popped while in use, so a re-entrant call on the same thread gets its own.
    return _local.__dict__.setdefault(pool, {}).pop(key, None) or make(key)


def checkin(pool: str, key: Any, item: Any) -> None:
    _local.__dict__.setdefault(pool, {})[key] = item
