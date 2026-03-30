from decider.modules._ext import register_graph_module
from .mapper import MapperModule, ModuleOutputSelector, ModuleInputSelector

register_graph_module(MapperModule)

__all__ = [
    "MapperModule",
    "ModuleOutputSelector",
    "ModuleInputSelector",
]
