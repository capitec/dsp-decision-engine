import typing as t
from hamilton import node
from dataclasses import dataclass
from decider.dag.expanders.base import DeciderExpandableModule
from . import funcs

@dataclass
class EG3Module(DeciderExpandableModule):
    config_key: str
    def expand_nodes(self) -> t.Dict[str, node.Node]:
        return {
            "config": ConfigManager(key=self.config_key).to_nodes(),
            "a": node.Node.from_fn(funcs.a),
            "b": node.Node.from_fn(funcs.b),
            "c": node.Node.from_fn(funcs.c),
        }
    def compile(self, config=None):
        from decider.dag.compile import CompiledModulePlaceholder
        # TODO do this from the nodes and maybe a wrapper or something?
        return CompiledModulePlaceholder(funcs) #<-  TODO have it compile with config
