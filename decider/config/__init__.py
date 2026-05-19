from .core import ConfigManager, VersionedConfig
from .file import JsonFileConfigManager, YamlFileConfigManager, TomlFileConfigManager
from ._ext import ConfigManagerModel, register_config_manager


def _register_builtins() -> None:
    register_config_manager(JsonFileConfigManager)
    register_config_manager(YamlFileConfigManager)
    register_config_manager(TomlFileConfigManager)


_register_builtins()

__all__ = [
    "ConfigManager",
    "VersionedConfig",
    "JsonFileConfigManager",
    "YamlFileConfigManager",
    "TomlFileConfigManager",
    "ConfigManagerModel",
    "register_config_manager",
]
