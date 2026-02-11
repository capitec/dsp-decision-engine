import typing as t
from pydantic import BaseModel
from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from hamilton import node


class DeciderExpandableModule(ABC):
    @abstractmethod
    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, "node.Node"]:
        pass

class ConfigurableDeciderExpandableModule(BaseModel, DeciderExpandableModule):
    @abstractmethod
    def expand_nodes(self, config: t.Dict[str, t.Any]) -> t.Dict[str, "node.Node"]:
        pass