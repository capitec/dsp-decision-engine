from .module import PrioritizedTreeModule, TreeModule

def register_tree_modules():
    from .._ext import register_graph_module
    register_graph_module(PrioritizedTreeModule)
    register_graph_module(TreeModule)
