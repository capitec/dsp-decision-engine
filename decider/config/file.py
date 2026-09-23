from __future__ import annotations

import json
import os
import tempfile
import typing as t
from pathlib import Path

from .store import ConfigStore
from .versions import Version, VersionedConfig


class JsonFileStore(ConfigStore):
    """A config store in a directory: `{basepath}/{version}/{key}.json`.

    A dotted key is a subdirectory, so `"credit.params"` is
    `{basepath}/1.2.0/credit/params.json`.

    Example::

        store = JsonFileStore(basepath="configs")
        store.create_version({"params": {"cut": 0.7}})   # configs/0.0.0/params.json
    """

    type: t.Literal["file:json"] = "file:json"
    basepath: str = "configs"

    def versions(self) -> list[Version]:
        root = Path(self.basepath)
        if not root.is_dir():
            return []
        found = []
        for entry in root.iterdir():
            try:
                version = Version.parse(entry.name)
            except ValueError:
                continue  # not a version directory, e.g. an unfinished write
            if entry.is_dir():
                found.append(version)
        return sorted(found)

    def read(self, version: str | Version) -> VersionedConfig:
        version = Version.parse(version)
        root = Path(self.basepath, str(version))
        if not root.is_dir():
            raise FileNotFoundError(f"config version {version} not found in {self.basepath!r}")
        config = {
            ".".join(path.relative_to(root).with_suffix("").parts): json.loads(path.read_bytes())
            for path in sorted(root.rglob("*.json"))
        }
        return VersionedConfig(version, config)

    def _write(self, versioned: VersionedConfig) -> None:
        final = Path(self.basepath, str(versioned.version))
        if final.exists():
            raise FileExistsError(f"config version {versioned.version} already exists in {self.basepath!r}")
        os.makedirs(self.basepath, exist_ok=True)
        # Written aside and renamed into place, so a failed write never shows up as a version.
        staging = Path(tempfile.mkdtemp(prefix=".", dir=self.basepath))
        for key, value in versioned.config.items():
            path = staging.joinpath(*key.split(".")).with_suffix(".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2))
        os.rename(staging, final)
