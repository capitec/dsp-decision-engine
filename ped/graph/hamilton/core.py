import typing as t
from types import ModuleType
from dataclasses import dataclass, field
from hamilton.lifecycle.base import BasePostGraphConstruct
from decider.dag.expanders.base import DeciderExpandableModule

if t.TYPE_CHECKING:
    from hamilton import graph

@dataclass
class DeciderAdaptorHook(BasePostGraphConstruct):
    modules: t.List[DeciderExpandableModule] = field(default_factory=list)

    def add_module(self, mod: DeciderExpandableModule):
        self.modules.append(mod)

    def post_graph_construct(
        self,
        *,
        graph: "graph.FunctionGraph",
        modules: t.List[ModuleType],
        config: t.Dict[str, t.Any],
    ):
        extra_nodes = {}
        for m in self.modules:
            extra_nodes.update(m.expand_nodes(config))
        graph.nodes = graph.with_nodes(extra_nodes).nodes
