from ._ext import register_graph_module, GraphModule
from .expression import Node

def _load_core_modules():
    from .credit import register_credit_modules
    register_credit_modules()

_load_core_modules()

__all__: list[str] = [
    "register_graph_module",
    "GraphModule",
    "Node",
]