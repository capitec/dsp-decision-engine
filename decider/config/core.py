import asyncio
import typing as t
from abc import abstractmethod

from pydantic import PrivateAttr
from decider._ext import TypeDiscriminatedBaseModule
from .versioned import VersionedConfig, Version, VersionPart, with_versioned_config


class CoreConfigManager(TypeDiscriminatedBaseModule):
    """Base pydantic model for versioned config managers.

    Runtime state (_current, _lock) is stored in PrivateAttr so pydantic
    doesn't include it in serialisation. Subclasses implement the four
    storage primitives; the public API is fully implemented here.

    _dirty state lives on VersionedConfig.is_dirty; _lock guards all reads
    and writes of _current.
    """

    _current: t.Optional[VersionedConfig] = PrivateAttr(default=None)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    # ------------------------------------------------------------------
    # Abstract storage primitives
    # ------------------------------------------------------------------

    @abstractmethod
    async def _load_version(self, version: str) -> VersionedConfig: ...

    @abstractmethod
    async def _write_version(self, versioned_config: VersionedConfig) -> None: ...

    @abstractmethod
    async def _version_exists(self, version: str) -> bool: ...

    @abstractmethod
    async def _latest_version(self) -> t.Optional[Version]: ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self) -> VersionedConfig:
        if self._current is None:
            async with self._lock:
                # Check it hasn't updated while we were waiting for the lock
                if self._current is None:
                    latest = await self._latest_version()
                    if latest is None:
                        self._current = VersionedConfig(version=Version(0, 0, -1), config={})
                    else:
                        self._current = await self._load_version(latest)
        return self._current
    

    async def current_version_context(self) -> t.ContextManager[VersionedConfig]:
        """Context manager to get the current versioned config. Use this in any code that needs access to the config."""
        config = await self.get()
        return with_versioned_config(config)


    async def create_version(self, bump: VersionPart = VersionPart.MINOR, force: bool = False) -> VersionedConfig:
        async with self._lock:
            if self._current is None:
                self._current = VersionedConfig(version=Version(0,0,0), config={})
                return self._current
            if not force:
                latest_version = await self._latest_version()
                if latest_version is not None and latest_version != self._current.version:
                    raise ValueError(
                        f"Current version {self._current.version!r} should exactly match latest version {latest_version!r} before creating a new version."
                        + (
                            "You are behind the latest version please run check_for_updates to fetch the latest version, or pass force=True to ignore this check."
                            if latest_version > self._current.version else 
                            "You are ahead of the latest version, which likely means you have un-pushed changes. Please push them before creating a new version, or pass force=True to ignore this check."
                          )
                    )
            self._current = VersionedConfig(
                version=self._current.version.bump(bump), 
                config=self._current.config.copy()
            )
            return self._current

    async def save_version(self, overwrite: bool = False) -> None:
        async with self._lock:
            if self._current is None:
                raise RuntimeError("No current version to save. Call create_version() first.")

            exists = await self._version_exists(self._current.version)
            if exists and not overwrite:
                raise FileExistsError(
                    f"Version {self._current.version!r} already exists. "
                    "Pass overwrite=True to overwrite."
                )

            await self._write_version(self._current)

    async def check_for_updates(self) -> t.Tuple[t.Optional[Version], bool]:
        latest = await self._latest_version()
        if latest is None:
            return None, False
        async with self._lock:
            current = self._current
        if current is None:
            return latest, True
        has_update = latest > current.version
        return latest, has_update

    async def pull_version(
        self,
        version: t.Optional[str] = None,
        force: bool = False,
    ) -> VersionedConfig:
        async with self._lock:

            target = version or await self._latest_version()
            if target is None:
                raise RuntimeError("No versions available in the store.")
            if not force and self._current is not None:
                if self._current.version >= target.version:
                    raise ValueError(
                        f"Current version {self._current.version!r} is newer or equal to target version {target!r}. "
                        "Pass force=True to ignore this check."
                    )
            
            self._current = await self._load_version(target)
            return self._current

    async def subscribe_version_updates(self, force: bool = True) -> None:
        """Poll for new versions and auto-pull. Safe to run as a background task."""
        from decider.settings import settings, SETTINGS_DEFAULT_CONFIG_POLL_DURATION_S

        while True:
            try:
                poll_seconds: int = getattr(
                    settings, "config_poll_duration_s", SETTINGS_DEFAULT_CONFIG_POLL_DURATION_S
                )
                await asyncio.sleep(poll_seconds)
                _, has_update = await self.check_for_updates()
                if has_update:
                    await self.pull_version(force=force)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
