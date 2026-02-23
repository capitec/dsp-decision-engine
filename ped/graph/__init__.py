from ._ext import GraphBuilder, register_builder
from .graph import BaseGraph

def register_default_builders():
    """
    Registers the default graph builders included with PED. This function is called automatically when the module is imported, but can be called again to ensure that the default builders are registered if the module was imported before the builders were defined.
    """
    from . import (
        hamilton
    )
    register_builder(hamilton.HamiltonBuilder)

register_default_builders()