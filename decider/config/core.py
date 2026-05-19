import asyncio
import typing as t
from abc import abstractmethod
from dataclasses import dataclass, field

from pydantic import PrivateAttr
from decider._ext import TypeDiscriminatedBaseModule


@dataclass
class VersionedConfig:
    version: str
    config: t.Dict[str, t.Any] = field(default_factory=dict)


def _parse_version(version: str) -> t.Tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version string: {version!r}. Expected MAJOR.MINOR.PATCH")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _bump_version(version: str, bump: t.Literal["major", "minor", "patch"] = "minor") -> str:
    major, minor, patch = _parse_version(version)
    if bump == "major":
        return f"{major + 1}.0.0"
    elif bump == "minor":
        return f"{major}.{minor + 1}.0"
    else:
        return f"{major}.{minor}.{patch + 1}"


class ConfigManager(TypeDiscriminatedBaseModule):
    """Base pydantic model for versioned config managers.

    Runtime state (_current, _dirty) is stored in PrivateAttr so pydantic
    doesn't include it in serialisation. Subclasses implement the four
    storage primitives; the public API is fully implemented here.
    """

    _current: t.Optional[VersionedConfig] = PrivateAttr(default=None)
    _dirty: bool = PrivateAttr(default=False)

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
    async def _latest_version(self) -> t.Optional[str]: ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self) -> VersionedConfig:
        if self._current is None:
            latest = await self._latest_version()
            if latest is None:
                self._current = VersionedConfig(version="0.1.0")
            else:
                self._current = await self._load_version(latest)
        return self._current

    def create_version(self, bump: t.Literal["major", "minor", "patch"] = "minor") -> VersionedConfig:
        if self._current is None:
            new_version = "0.1.0"
            new_config: t.Dict[str, t.Any] = {}
        else:
            new_version = _bump_version(self._current.version, bump)
            new_config = dict(self._current.config)

        self._current = VersionedConfig(version=new_version, config=new_config)
        self._dirty = True
        return self._current

    async def save_version(self, override: bool = False) -> None:
        if self._current is None:
            raise RuntimeError("No current version to save. Call create_version() first.")

        exists = await self._version_exists(self._current.version)
        if exists and not override:
            raise FileExistsError(
                f"Version {self._current.version!r} already exists. "
                "Pass override=True to overwrite."
            )

        await self._write_version(self._current)
        self._dirty = False

    async def check_for_updates(self) -> t.Tuple[t.Optional[str], bool]:
        latest = await self._latest_version()
        if latest is None:
            return None, False
        if self._current is None:
            return latest, True
        has_update = _parse_version(latest) > _parse_version(self._current.version)
        return latest, has_update

    async def pull_version(
        self,
        version: t.Optional[str] = None,
        force: bool = False,
    ) -> VersionedConfig:
        if self._dirty and not force:
            raise RuntimeError(
                "There are unsaved changes to the current version. "
                "Call save_version() first, or pass force=True to discard them."
            )

        target = version or await self._latest_version()
        if target is None:
            raise RuntimeError("No versions available in the store.")

        self._current = await self._load_version(target)
        self._dirty = False
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
