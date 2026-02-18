from ._ext import register_graph_module, GraphModule, ConstructedGraphModules


def load_core_modules():
    from .decision_table import DecisionTableModule
    from .extract_from_struct import ExtractFromStructModule