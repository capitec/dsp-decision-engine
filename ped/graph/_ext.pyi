import typing as t
from .builder import BaseBuilder
from ped._ext import TExtendableModel

GraphBuilder = TExtendableModel[BaseBuilder]


def register_builder(provider_class: t.Type[BaseBuilder]) -> None: ...

__all__: list[str]