"""
Stub file for static type checkers.
ConfigManagerModel is generated dynamically; this gives pyright/mypy a
concrete type.
"""
import typing as t
from .core import ConfigManager
from decider._ext import TExtendableModel

ConfigManagerModel = TExtendableModel[ConfigManager]


def register_config_manager(provider_class: t.Type[ConfigManager]) -> None: ...


__all__: list[str]
