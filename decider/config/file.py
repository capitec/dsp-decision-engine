import os
import typing as t
from abc import abstractmethod

from pydantic import Field
from .core import CoreConfigManager, VersionedConfig
from .versioned import Version


class BaseFileConfigManager(CoreConfigManager):
    """Shared logic for all file-backed config managers.

    Files are stored as:
        {basepath}/{version}/{key}.{ext}

    All keys within a version directory are merged into VersionedConfig.config.

    Example:
        configs/1.0.0/tree.json
        configs/1.0.0/scorecard.json
        → VersionedConfig(version="1.0.0", config={"tree": {...}, "scorecard": {...}})
    """

    basepath: str = Field(default="configs")

    # Subclasses set this to the file extension (without leading dot).
    _file_ext: t.ClassVar[str]

    def _version_dir(self, version) -> str:
        return os.path.join(self.basepath, str(version))

    def _file_path(self, version, key: str) -> str:
        return os.path.join(self._version_dir(version), f"{key}.{self._file_ext}")

    def _list_versions(self) -> t.List[str]:
        if not os.path.isdir(self.basepath):
            return []
        valid = []
        for entry in os.listdir(self.basepath):
            if not os.path.isdir(os.path.join(self.basepath, entry)):
                continue
            try:
                parts = entry.split(".")
                if len(parts) == 3:
                    int(parts[0]); int(parts[1]); int(parts[2])
                    valid.append(entry)
            except (ValueError, AttributeError):
                pass
        return sorted(valid, key=lambda v: tuple(int(x) for x in v.split(".")))

    # ------------------------------------------------------------------
    # Format-specific primitives (override in subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def _deserialize(self, _raw: bytes) -> t.Any: ...

    @abstractmethod
    def _serialize(self, _data: t.Any) -> bytes: ...

    # ------------------------------------------------------------------
    # ConfigManager storage primitives
    # ------------------------------------------------------------------

    async def _load_version(self, version: str) -> VersionedConfig:
        version_dir = self._version_dir(version)
        if not os.path.isdir(version_dir):
            raise FileNotFoundError(f"Version directory not found: {version_dir!r}")

        ext = f".{self._file_ext}"
        config: t.Dict[str, t.Any] = {}
        for filename in sorted(os.listdir(version_dir)):
            if filename.endswith(ext):
                key = filename[: -len(ext)]
                with open(os.path.join(version_dir, filename), "rb") as f:
                    config[key] = self._deserialize(f.read())

        return VersionedConfig(version=version, config=config)

    async def _write_version(self, versioned_config: VersionedConfig) -> None:
        for key, value in versioned_config.config.items():
            path = self._file_path(versioned_config.version, key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(self._serialize(value))

    async def _version_exists(self, version: str) -> bool:
        return os.path.isdir(self._version_dir(version))

    async def _latest_version(self) -> t.Optional[Version]:
        versions = self._list_versions()
        if not versions:
            return None
        v = versions[-1]
        parts = v.split(".")
        return Version(int(parts[0]), int(parts[1]), int(parts[2]))


class JsonFileConfigManager(BaseFileConfigManager):
    type: t.Literal["file:json"]
    _file_ext: t.ClassVar[str] = "json"

    def _deserialize(self, raw: bytes) -> t.Any:
        import json
        return json.loads(raw)

    def _serialize(self, data: t.Any) -> bytes:
        import json
        return json.dumps(data, indent=2).encode()


class YamlFileConfigManager(BaseFileConfigManager):
    type: t.Literal["file:yaml"]
    _file_ext: t.ClassVar[str] = "yaml"

    def _deserialize(self, raw: bytes) -> t.Any:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to use yaml format. "
                "Install it with: pip install pyyaml"
            )
        return yaml.safe_load(raw)

    def _serialize(self, data: t.Any) -> bytes:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to use yaml format. "
                "Install it with: pip install pyyaml"
            )
        return yaml.dump(data, default_flow_style=False, allow_unicode=True).encode()


class TomlFileConfigManager(BaseFileConfigManager):
    type: t.Literal["file:toml"]
    _file_ext: t.ClassVar[str] = "toml"

    def _deserialize(self, raw: bytes) -> t.Any:
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                raise ImportError(
                    "A TOML library is required to use toml format. "
                    "Install tomli with: pip install tomli  (Python < 3.11)"
                )
        return tomllib.loads(raw.decode())

    def _serialize(self, data: t.Any) -> bytes:
        try:
            import tomli_w  # type: ignore
        except ImportError:
            raise ImportError(
                "tomli_w is required to write toml files. "
                "Install it with: pip install tomli-w"
            )
        return tomli_w.dumps(data).encode()
