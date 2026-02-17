"""
This file is needed because the static type checkers dont like the 
fact that CacheProvider is generated from a function. So we define a
stub file that defines the type of CacheProvider and the register_cache_provider function, 
which can be used by static type checkers to understand the types of these objects.
"""
import typing as t
from .core import BaseParameterCache
from ped._ext import TExtendableModel

ParameterCache = TExtendableModel[BaseParameterCache]

def register_cache_provider(provider_class: t.Type[BaseParameterCache]) -> None: ...
__all__: list[str]