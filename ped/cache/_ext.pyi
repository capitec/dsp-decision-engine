from typing import Type, Callable
from ped._ext import TExtendableModel
from .core import BaseCache

CacheProvider: TExtendableModel[BaseCache]
register_cache: Callable[[Type[BaseCache]], None]

__all__: list[str]