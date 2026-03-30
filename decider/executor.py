# For now we only have 1
import typing as t
import polars as pl
from dataclasses import dataclass


if t.TYPE_CHECKING:
    from modules.core import Node

@dataclass(slots=True)
class Graph:
    nodes: t.Dict[str, "Node"]  # mapping of node_id → Node
    cached_plan: t.Dict[str, pl.Expr] = None

class Executor:
    def compile(self, modules: t.List["BaseModule"]) -> Graph:
        graph_nodes = {}
        for module in modules:
            module_nodes = module.get_nodes()
            for node in module_nodes:
                if node.node_id in graph_nodes:
                    raise ValueError(f"Duplicate node_id {node.node_id} found in module {module.name}")
                graph_nodes[node.node_id] = node
        return Graph(nodes=graph_nodes)
    
    def execute(
        self, 
        graph: Graph,
        # **inputs: pl.LazyFrame,
        # The above would be preferable but for now 
        inputs: t.Dict[str, pl.Expr],
        outputs: t.List[str] = None
    ) -> pl.LazyFrame: