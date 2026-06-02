import typing as t
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

import polars as pl

from decider._ext import TypeDiscriminatedBaseModule
from decider.types import TInputType, TOutputType
from decider.graphutil import topological_sort

if t.TYPE_CHECKING:
    from decider.modules.expression import Node, CompiledExpressions
    from decider.modules.core import BaseModule


# ── FrameNode ─────────────────────────────────────────────────────────────────

@dataclass
class FrameNode:
    """A named frame-level computation with declared dependencies.

    Stores a bound execute method rather than the module itself, so
    CompiledFrameGraph can call it directly without re-entering executor.execute.
    """
    name: str
    callable: t.Callable[[TInputType, "Executor"], TOutputType]
    depends_on: t.List[str]

    def get_dependencies(self) -> t.List[str]:
        return self.depends_on


@dataclass
class CompiledFrameGraph:
    """Sorted list of FrameNodes ready to execute in dependency order."""
    nodes: t.List[FrameNode]

    def execute(
        self,
        inputs: TInputType,
        executor: "Executor",
    ) -> TOutputType:
        frames = {**inputs}
        for node in self.nodes:
            result = node.callable(frames, executor)
            frames[node.name] = result
        return frames


# ── Executor ABC ──────────────────────────────────────────────────────────────

class Executor(TypeDiscriminatedBaseModule, ABC):
    debug: bool = False
    collect: bool = True

    @abstractmethod
    def compile_expression_graph(self, nodes: t.List["Node"]) -> "CompiledExpressions":
        """Topologically sort expression nodes into a CompiledExpressions artifact."""
        ...

    def compile_frame_graph(self, frame_nodes: t.List[FrameNode]) -> CompiledFrameGraph:
        """Topologically sort frame nodes into an executable graph."""
        return CompiledFrameGraph(nodes=topological_sort(frame_nodes))

    def execute(self, module: "BaseModule", inputs: TInputType) -> TOutputType:
        graph = module.compile(self)
        result = graph.execute(inputs, self)
        if self.debug:
            if self.collect:
                result = {k: v.collect() if isinstance(v, pl.LazyFrame) else v for k, v in result.items()}
            return result
        res = result[graph.nodes[-1].name]  # Return only the final output frame
        if self.collect and isinstance(res, pl.LazyFrame):
            res = res.collect()
        return res


# ── SimpleExecutor ────────────────────────────────────────────────────────────

class SimpleExecutor(Executor):
    type: t.Literal["simple"]

    def compile_expression_graph(self, nodes: t.List["Node"]) -> "CompiledExpressions":
        from decider.modules.expression import CompiledExpressions, ExternalInputNode

        sorted_nodes = topological_sort(nodes)

        expressions: OrderedDict[str, pl.Expr] = OrderedDict()
        for node in sorted_nodes:
            try:
                expr = node.callable(**node.get_input_expressions())
            except Exception as e:
                missing = [
                    f"'{k}' (column '{v.input_name}')"
                    for k, v in node.input_map.items()
                    if isinstance(v, ExternalInputNode)
                ]
                hint = (
                    f"\nHint: node '{node.name}' expected columns: "
                    + ", ".join(missing)
                ) if missing else ""
                raise ValueError(
                    f"Error building expression for '{node.name}': {e}{hint}"
                ) from e
            expressions[node.name] = expr

        return CompiledExpressions(expressions=expressions)

