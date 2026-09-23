from __future__ import annotations

import copy
import typing as t
from abc import abstractmethod

from pydantic import ConfigDict

from decider.registry import BaseRegistryModule

from .versions import Version, VersionedConfig, VersionPart


class ConfigStore(BaseRegistryModule, root=True):
    """Versioned params and `ConfigurableStep` documents; subclass it for a new backend.

    A store holds immutable versions and never loads one by itself: a caller
    reads the version it wants. Backends implement `versions`, `read` and `_write`.

    Example::

        store = JsonFileStore(basepath="configs")
        store.create_version({"params": {"cut": 0.7}, "tree": tree.model_dump(mode="json")})
        doc = store.read(store.latest_version()).config["tree"]
        tree = ConfigurableStep.load(doc)
    """

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def versions(self) -> list[Version]:
        """Every stored version, oldest first.

        Example::

            store.versions()   # [Version(0, 0, 0), Version(0, 1, 0)]
        """

    @abstractmethod
    def read(self, version: str | Version) -> VersionedConfig:
        """One stored version.

        Example::

            store.read("0.1.0").config["params"]
        """

    @abstractmethod
    def _write(self, versioned: VersionedConfig) -> None: ...

    def latest_version(self) -> Version | None:
        """The newest stored version, or `None` for an empty store.

        Example::

            store.read(store.latest_version())
        """
        versions = self.versions()
        return versions[-1] if versions else None

    def create_version(self, config: dict[str, t.Any], bump: VersionPart = VersionPart.MINOR) -> VersionedConfig:
        """Write `config` as a new version one `bump` past the latest; the first version is `0.0.0`.

        Example::

            store.create_version({"params": {"cut": 0.8}}, bump=VersionPart.PATCH)   # 0.1.0 -> 0.1.1
        """
        latest = self.latest_version()
        version = latest.bump(bump) if latest is not None else Version(0, 0, 0)
        # Deep copy so later edits to the caller's dicts can't change a stored version.
        versioned = VersionedConfig(version, copy.deepcopy(config))
        self._write(versioned)
        return versioned
