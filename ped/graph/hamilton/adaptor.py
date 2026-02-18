import typing as t
from types import ModuleType
from dataclasses import dataclass, field
from hamilton.lifecycle.base import BasePostGraphConstruct
from ped.modules import ConstructedGraphModules, GraphModule

if t.TYPE_CHECKING:
    from hamilton import graph
    from hamilton.node import Node

@dataclass
class DeciderAdaptorHook(BasePostGraphConstruct):
    modules: ConstructedGraphModules = field(default_factory=ConstructedGraphModules)

    def add_module(self, key: str, mod: GraphModule):
        self.modules.root[key] = mod

    # It would actually be nicer to add this to GraphModule as its more flexible and can be overridden for needs of a module.
    # however that breaks the isolation of hamilton being independent from the 
    # up to the point of the specific graph builder making it harder to swap out builders
    # if later we choose to create a custom node type we can move this over.
    @staticmethod
    def get_namespaced_nodes(
        namespace: str,
        module: GraphModule,
        config: t.Dict[str, t.Any],
    ) -> t.Dict[str, "Node"]:
        nodes = module.root.expand_nodes(config)

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
