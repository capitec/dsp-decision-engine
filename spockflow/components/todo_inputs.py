import typing
from dataclasses import dataclass
from hamilton import node
import pandas as pd

from spockflow.nodes import VariableNodeExpander


@dataclass
class InputNode(VariableNodeExpander):
    return_type: typing.Any
    doc: typing.Optional[str] = None

    def generate_nodes(
        self, config: dict, var_name: str = None
    ) -> typing.List[node.Node]:
        # TODO potentially add some sort of input data verification here.
        return []


T = typing.TypeVar("T")


def input_value(return_type: T = pd.Series, doc: typing.Optional[str] = None) -> T:
    return InputNode(return_type, doc)
