"""
This module enables config sources to be extended by external packages without creating hard dependencies on those packages. It does this by maintaining a global union type of all registered sources, which can be extended by calling the `register_module` function with a new source type. The `GraphModule` model is then rebuilt to include the new source type in its union.
"""
import typing as t
from pydantic import RootModel
from .core import BaseModule
from ped._ext import create_extendable_model

GraphModule, register_graph_module = create_extendable_model(
    BaseModule, 
    discriminator_field="type",
    model_name="GraphModule"
)

class GraphModuleDict(RootModel):
    root: t.Dict[str, GraphModule] # pyright: ignore[reportInvalidTypeForm]


__all__ = ['GraphModule', 'register_graph_module', 'GraphModuleDict']