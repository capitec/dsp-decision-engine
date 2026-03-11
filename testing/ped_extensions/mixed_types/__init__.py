from ped.modules.functional import generate_from_module
from . import impl
from ped.modules import register_graph_module

MixedTypesModule = generate_from_module("mixed_types", impl)
register_graph_module(MixedTypesModule)