"""
This file is needed because the static type checkers dont like the 
fact that ParameterSource is generated from a function. So we define a
stub file that defines the type of ParameterSource and the register_graph_module function, 
which can be used by static type checkers to understand the types of these objects.
"""
import typing as t
from .core import BaseAdapter
from ped._ext import TExtendableModel

GraphAdapter = TExtendableModel[BaseAdapter]


def register_graph_adapter(provider_class: t.Type[BaseAdapter]) -> None: ...

__all__: list[str]