from ._ext import register_graph_adapter, GraphAdapter

def _load_adapters():
    from .types import TypeAdapter
    register_graph_adapter(TypeAdapter)

def get_default_adapters() -> list[GraphAdapter]:
    return [GraphAdapter(type="type_adapter")]

_load_adapters()

__all__ = ['GraphAdapter', 'register_graph_adapter', 'get_default_adapters']