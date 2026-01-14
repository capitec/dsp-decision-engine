import typing as t
from hamilton import node
from decider.dag.core import DeciderExpandableModule
from . import funcs

class EG4Module(DeciderExpandableModule):
    def expand_nodes(self) -> t.Dict[str, node.Node]:
        return {
            "converted": node.Node.from_fn(funcs.converted)
        }