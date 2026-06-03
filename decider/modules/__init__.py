from ._ext import register_graph_module, GraphModule
from .expression import Node

def _load_core_modules():
    from .primitives.sequential import SequentialModule
    from .primitives.join import JoinModule, FrameRef, FrameModule
    from .credit import register_credit_modules

    for cls in (SequentialModule, JoinModule, FrameRef):
        register_graph_module(cls)

    register_credit_modules()

_load_core_modules()

__all__: list[str] = [
    "register_graph_module",
    "GraphModule",
    "Node",
]