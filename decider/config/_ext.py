"""
Enables external packages to register custom ConfigManager implementations
without creating hard dependencies. Follows the same _ext.py pattern as
decider/modules/_ext.py — a pydantic discriminated union keyed on `type`.
"""
import typing as t
from .core import ConfigManager
from decider._ext import create_extendable_model


ConfigManagerModel, register_config_manager = create_extendable_model(
    ConfigManager,
    discriminator_field="type",
    model_name="ConfigManagerModel",
)

__all__ = ["ConfigManagerModel", "register_config_manager"]
