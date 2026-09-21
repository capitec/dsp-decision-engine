"""
This module enables config sources to be extended by external packages without creating hard dependencies on those packages. It does this by maintaining a global union type of all registered sources, which can be extended by calling the `register_module` function with a new source type. The `GraphModule` model is then rebuilt to include the new source type in its union.
"""
import typing as t
from .core import BaseModule
from decider._ext import create_extendable_model


GraphModule, register_graph_module = create_extendable_model(
    BaseModule, 
    discriminator_field="type",
    model_name="GraphModule"
)

__all__ = ['GraphModule', 'register_graph_module', ]