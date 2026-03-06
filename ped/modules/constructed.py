import typing as t
from pydantic import Field, BaseModel, model_validator
from ped.graph import BaseGraph
from ped.graph.builder import BaseBuilder
from ped.modules._ext import GraphModule
from ped.types import TInputType, TOutputType
from ped.adapters import GraphAdapter, get_default_adapters

if t.TYPE_CHECKING:
    from .core import PEDNode


class ConstructedGraphModules(BaseModel):
    modules: t.List[GraphModule] = Field(default_factory=list) # pyright: ignore[reportInvalidTypeForm]
    adapters: t.List[GraphAdapter] = Field(default_factory=get_default_adapters)

    @model_validator(mode='after')
    def validate_unique_names(self):
        names = [m.root.name for m in self.modules]
        seen = set()
        duplicates = {n for n in names if n in seen or seen.add(n)}
        if duplicates:
            raise ValueError(f"Module names must be unique. Duplicates found: {duplicates}")
        return self


    def namespaced_nodes(self) -> t.List["PEDNode"]:
        """Return all nodes from all modules with namespaced names."""
        nodes = []
        for module in self.modules:
            module_nodes = module.root.module_namespaced_nodes()
            nodes.extend(module_nodes)
        for adapter in self.adapters:
            nodes = adapter.root.adapt(nodes)
        return nodes


    def build_graph(
        self,
        output_nodes: t.List[str],
        builder: "BaseBuilder"=None, # pyright: ignore[reportUnusedFunction]
    ) -> "BaseGraph":
        if builder is None:
            from ..graph import GraphBuilder
            builder = GraphBuilder(type="hamilton").root
        return builder.build_graph(
            self,
            output_nodes=output_nodes,
        )

    def execute(
        self,
        inputs: TInputType,
        output_nodes:t.List[str],
        builder: "BaseBuilder"=None, # pyright: ignore[reportUnusedFunction]
        **kwargs
    ) -> TOutputType:
        graph = self.build_graph(output_nodes=output_nodes, builder=builder)
        return graph.execute(inputs=inputs, **kwargs)