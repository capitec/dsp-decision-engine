"""
This module enables graph builders to be extended by external packages without creating hard dependencies on those packages. It does this by maintaining a global union type of all registered builders, which can be extended by calling the `register_builder` function with a new builder type. The `GraphBuilder` model is then rebuilt to include the new builder type in its union.
"""
from .builder import BaseBuilder
from ped._ext import create_extendable_model

GraphBuilder, register_builder = create_extendable_model(
    BaseBuilder, 
    discriminator_field="type",
    model_name="GraphBuilder"
)


__all__ = ['GraphBuilder', 'register_builder']