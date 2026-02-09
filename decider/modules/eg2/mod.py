import typing as t
from hamilton import node
from decider.dag.expanders.base import DeciderExpandableModule
from . import funcs

class EG2Module(DeciderExpandableModule):
    def expand_nodes(self) -> t.Dict[str, node.Node]:
        return {
            "a": node.Node.from_fn(funcs.a),
            "b": node.Node.from_fn(funcs.b),
            "c": node.Node.from_fn(funcs.c),
        }
    def compile(self):
        from decider.dag.compile import CompiledModulePlaceholder
        # TODO do this from the nodes and maybe a wrapper or something?
        return CompiledModulePlaceholder(funcs)
