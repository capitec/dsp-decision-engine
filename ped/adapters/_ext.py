"""
This module enables config sources to be extended by external packages without creating hard dependencies on those packages. It does this by maintaining a global union type of all registered sources, which can be extended by calling the `register_module` function with a new source type. The `GraphModule` model is then rebuilt to include the new source type in its union.
"""
import typing as t
from .core import BaseAdapter
from ped._ext import create_extendable_model


GraphAdapter, register_graph_adaptor = create_extendable_model(
    BaseAdapter, 
    discriminator_field="type",
    model_name="GraphAdapter"
)

__all__ = ['GraphAdapter', 'register_graph_adaptor', ]