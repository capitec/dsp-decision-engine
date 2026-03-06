from ._ext import register_graph_adaptor, GraphAdapter

def _load_adapters():
    from .types import TypeAdapter
    register_graph_adaptor(TypeAdapter)

def get_default_adapters() -> list[GraphAdapter]:
    return [GraphAdapter(type="type_adapter")]

_load_adapters()

__all__ = ['GraphAdapter', 'register_graph_adaptor', 'get_default_adapters']