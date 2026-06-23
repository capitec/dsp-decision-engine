"""
Stub file for static type checkers.
ConfigManagerModel is generated dynamically; this gives pyright/mypy a
concrete type.
"""
import typing as t
from .core import CoreConfigManager
from decider._ext import TExtendableModel

ConfigManager = TExtendableModel[CoreConfigManager]


def register_config_manager(provider_class: t.Type[CoreConfigManager]) -> None: ...


__all__: list[str]
