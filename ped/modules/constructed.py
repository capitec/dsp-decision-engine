import typing as t
from pydantic import Field, RootModel, model_validator
from ped.graph import BaseGraph
from ped.graph.builder import BaseBuilder
from ped.modules._ext import GraphModule
from ped.types import TInputType, TOutputType


class ConstructedGraphModules(RootModel):
    root: t.List[GraphModule] = Field(default_factory=list) # pyright: ignore[reportInvalidTypeForm]

    @model_validator(mode='after')
    def validate_unique_names(self):
        names = [m.root.name for m in self.root]
        seen = set()
        duplicates = {n for n in names if n in seen or seen.add(n)}
        if duplicates:
            raise ValueError(f"Module names must be unique. Duplicates found: {duplicates}")
        return self


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
        output_nodes:t.List[str]=[],
        builder: "BaseBuilder"=None, # pyright: ignore[reportUnusedFunction]
        **kwargs
    ) -> TOutputType:
        graph = self.build_graph(output_nodes=output_nodes, builder=builder)
        return graph.execute(inputs=inputs, **kwargs)