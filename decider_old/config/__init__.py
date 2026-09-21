from .core import CoreConfigManager, VersionedConfig
from .file import JsonFileConfigManager, YamlFileConfigManager, TomlFileConfigManager
from ._ext import ConfigManager, register_config_manager


def _register_builtins() -> None:
    register_config_manager(JsonFileConfigManager)
    register_config_manager(YamlFileConfigManager)
    register_config_manager(TomlFileConfigManager)


_register_builtins()

__all__ = [
    "CoreConfigManager",
    "VersionedConfig",
    "JsonFileConfigManager",
    "YamlFileConfigManager",
    "TomlFileConfigManager",
    "ConfigManager",
    "register_config_manager",
]
