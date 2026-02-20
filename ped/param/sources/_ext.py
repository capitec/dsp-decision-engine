"""
This module enables config sources to be extended by external packages without creating hard dependencies on those packages. It does this by maintaining a global union type of all registered sources, which can be extended by calling the `register_source` function with a new source type. The `ParameterSource` model is then rebuilt to include the new source type in its union.
"""
from .core import BaseSource
from ped._ext import create_extendable_model

ParameterSource, register_source = create_extendable_model(
    BaseSource, 
    discriminator_field="type",
    model_name="ParameterSource"
)


__all__ = ['ParameterSource', 'register_source']