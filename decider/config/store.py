from __future__ import annotations

import copy
import typing as t
from abc import abstractmethod

from pydantic import ConfigDict, PrivateAttr

from decider.registry import BaseRegistryModule

from .versions import Version, VersionedConfig, VersionPart


class ConfigStore(BaseRegistryModule, root=True):
    """Versioned params and `ConfigurableStep` documents; subclass it for a new backend.

    A store holds immutable versions. It keeps one of them loaded as `get()`;
    nothing changes that except an explicit call (`get_latest`, `pull_version`,
    `create_version`), so a caller decides when to look for a newer version.
    Backends implement `versions`, `read` and `_write`.

    Example::

        store = JsonFileStore(basepath="configs")
        store.create_version({"params": {"cut": 0.7}, "tree": tree.model_dump(mode="json")})
        latest, has_update = store.check_for_updates()
        if has_update:
            store.pull_version()
        doc = store.get().config["tree"]
        tree = ConfigurableStep.resolve(doc["type"]).model_validate(doc)
    """

    # ponytail: no lock; add a threading.Lock if several threads pull or create on one store.
    model_config = ConfigDict(extra="forbid")

    _current: VersionedConfig | None = PrivateAttr(default=None)

    @abstractmethod
    def versions(self) -> list[Version]:
        """Every stored version, oldest first."""

    @abstractmethod
    def read(self, version: str | Version) -> VersionedConfig:
        """One stored version, without loading it as `get()`."""

    @abstractmethod
    def _write(self, versioned: VersionedConfig) -> None: ...

    def latest_version(self) -> Version | None:
        versions = self.versions()
        return versions[-1] if versions else None

    def get(self) -> VersionedConfig:
        """The loaded version."""
        if self._current is None:
            raise RuntimeError("no config version loaded; call get_latest() or pull_version() first.")
        return self._current

    def get_latest(self) -> VersionedConfig:
        """Load the newest stored version if it is newer than the loaded one, and return the loaded one."""
        latest = self.latest_version()
        if latest is None:
            raise RuntimeError("the config store has no versions.")
        if self._current is None or self._current.version < latest:
            self._current = self.read(latest)
        return self._current

    def check_for_updates(self) -> tuple[Version | None, bool]:
        """`(latest stored version, whether it is newer than the loaded one)`."""
        latest = self.latest_version()
        if latest is None:
            return None, False
        return latest, self._current is None or latest > self._current.version

    def pull_version(self, version: str | Version | None = None, force: bool = False) -> VersionedConfig:
        """Load `version` (default: the latest). Refuses to go back or reload unless `force`, e.g. for a rollback."""
        target = Version.parse(version) if version is not None else self.latest_version()
        if target is None:
            raise RuntimeError("the config store has no versions.")
        if not force and self._current is not None and self._current.version >= target:
            raise ValueError(
                f"loaded version {self._current.version} is newer than or equal to {target}; "
                "pass force=True to load it anyway."
            )
        self._current = self.read(target)
        return self._current

    def create_version(
        self, config: dict[str, t.Any], bump: VersionPart = VersionPart.MINOR, force: bool = False
    ) -> VersionedConfig:
        """Write `config` as a new version one `bump` past the latest, and load it.

        The first version is `0.0.0`. Unless `force`, the loaded version must be
        the latest, so an edit made against an old version can't silently
        replace a newer one.
        """
        latest = self.latest_version()
        loaded = self._current.version if self._current is not None else None
        if not force and latest != loaded:
            raise ValueError(
                f"loaded version {loaded} is not the latest ({latest}); "
                "pull the latest first, or pass force=True."
            )
        version = latest.bump(bump) if latest is not None else Version(0, 0, 0)
        # Deep copy so later edits to the caller's dicts can't change a stored, loaded version.
        versioned = VersionedConfig(version, copy.deepcopy(config))
        self._write(versioned)
        self._current = versioned
        return versioned
