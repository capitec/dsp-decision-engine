"""
Enables external packages to register custom ConfigManager implementations
without creating hard dependencies. Follows the same _ext.py pattern as
decider/modules/_ext.py — a pydantic discriminated union keyed on `type`.
"""
import typing as t
from .core import CoreConfigManager
from decider._ext import create_extendable_model


ConfigManager, register_config_manager = create_extendable_model(
    CoreConfigManager,
    discriminator_field="type",
    model_name="ConfigManagerModel",
)

__all__ = ["ConfigManager", "register_config_manager"]
