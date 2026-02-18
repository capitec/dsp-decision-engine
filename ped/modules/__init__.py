from ._ext import register_graph_module, GraphModule, ConstructedGraphModules
from .core import PEDNode

def _load_core_modules():
    from .decision_table import DecisionTableModule
    from .extract_from_struct import ExtractFromStructModule
    register_graph_module(DecisionTableModule)
    register_graph_module(ExtractFromStructModule)

_load_core_modules()

__all__: list[str] = [
    "register_graph_module",
    "GraphModule",
    "PEDNode",
    "ConstructedGraphModules",
]