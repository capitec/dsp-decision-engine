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

    Subdirectories are supported and produce dotted keys:
        {basepath}/{version}/{group}/{key}.{ext}  →  config key "group.key"

    Nesting is arbitrary depth — each path segment becomes a dot-separated
    component of the key.

    Examples:
        configs/1.0.0/tree.json                  → key "tree"
        configs/1.0.0/scorecard.json             → key "scorecard"
        configs/1.0.0/01_init_afs/bureau.json    → key "01_init_afs.bureau"
        configs/1.0.0/01_init_afs/main.json      → key "01_init_afs.main"
        → VersionedConfig(version="1.0.0", config={
              "tree": {...},
              "scorecard": {...},
              "01_init_afs.bureau": {...},
              "01_init_afs.main": {...},
          })

    When writing, dotted keys are converted back to subdirectory paths.
    """

    basepath: str = Field(default="configs")

    # Subclasses set this to the file extension (without leading dot).
    _file_ext: t.ClassVar[str]

    def _version_dir(self, version) -> str:
        return os.path.join(self.basepath, str(version))

    def _file_path(self, version, key: str) -> str:
        # Convert dotted key to filesystem path:
        #   "01_init_afs.bureau_source"  →  "{version_dir}/01_init_afs/bureau_source.json"
        parts = key.split(".")
        return os.path.join(self._version_dir(version), *parts[:-1], f"{parts[-1]}.{self._file_ext}")

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

    def _collect_keys(self, directory: str, prefix: str = "") -> t.Dict[str, str]:
        """Recursively collect {dotted_key: filepath} for all matching files under directory."""
        ext = f".{self._file_ext}"
        result: t.Dict[str, str] = {}
        for entry in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                sub_prefix = f"{prefix}{entry}." if prefix else f"{entry}."
                result.update(self._collect_keys(full_path, sub_prefix))
            elif entry.endswith(ext):
                key = f"{prefix}{entry[:-len(ext)]}"
                result[key] = full_path
        return result

    async def _load_version(self, version: str) -> VersionedConfig:
        version_dir = self._version_dir(version)
        if not os.path.isdir(version_dir):
            raise FileNotFoundError(f"Version directory not found: {version_dir!r}")

        config: t.Dict[str, t.Any] = {}
        for key, filepath in self._collect_keys(version_dir).items():
            with open(filepath, "rb") as f:
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
