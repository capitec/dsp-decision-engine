"""
This file is needed because the static type checkers dont like the 
fact that ParameterSource is generated from a function. So we define a
stub file that defines the type of ParameterSource and the register_source function, 
which can be used by static type checkers to understand the types of these objects.
"""
import typing as t
from .core import BaseSource
from ped._ext import TExtendableModel

ParameterSource = TExtendableModel[BaseSource]

def register_source(provider_class: t.Type[BaseSource]) -> None: ...

__all__: list[str]