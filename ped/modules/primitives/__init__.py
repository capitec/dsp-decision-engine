from .chain import ChainModule
from .external import ExternalModule
from .hamilton import HamiltonModule
from .namespaced import NamespacedModule
from .stream import StreamablePandasModule

def register_primitive_modules():
    from .._ext import register_graph_module
    register_graph_module(ChainModule)
    register_graph_module(ExternalModule)
    register_graph_module(HamiltonModule)
    register_graph_module(NamespacedModule)
    register_graph_module(StreamablePandasModule)


register_primitive_modules()

__all__ = ["ChainModule", "ExternalModule", "HamiltonModule", "NamespacedModule", "StreamablePandasModule"]
