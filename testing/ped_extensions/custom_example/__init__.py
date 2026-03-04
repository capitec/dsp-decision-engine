def register_modules():
    from ped.modules import register_graph_module
    from .eg3.module import EG3Module
    from .eg1.func_example import ModWithConfig, ModWithoutConfig
    from .eg2.module import EG2Module

    register_graph_module(EG3Module)
    register_graph_module(ModWithConfig)
    register_graph_module(ModWithoutConfig)
    register_graph_module(EG2Module)

register_modules()