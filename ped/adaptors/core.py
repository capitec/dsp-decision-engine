import typing as t
from abc import ABC, abstractmethod
from ..modules.core import PEDNode
from ped._ext import TypeDiscriminatedBaseModule


class BaseAdapter(TypeDiscriminatedBaseModule, ABC):
    @abstractmethod
    def adapt(
        self, 
        inputs: t.List[PEDNode],
    ) -> t.List[PEDNode]:
        ...
