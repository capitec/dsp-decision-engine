from ._ext import register_graph_module, GraphModule
from .constructed import ConstructedGraphModules
from .core import PEDNode

def _load_core_modules():
    from .primitives import register_primitive_modules
    from .credit import register_credit_modules

    register_primitive_modules()
    register_credit_modules()

_load_core_modules()

__all__: list[str] = [
    "register_graph_module",
    "GraphModule",
    "PEDNode",
    "ConstructedGraphModules",
]