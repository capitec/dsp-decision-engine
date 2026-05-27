import typing as t
import inspect
import polars as pl
from pydantic import PrivateAttr
from dataclasses import dataclass, field
from collections import OrderedDict
from abc import abstractmethod

from decider.types import TInputType, TOutputType
from .core import BaseModule, BaseExecuteModule

if t.TYPE_CHECKING:
    from decider.executor import Executor


# ── Input ref types ───────────────────────────────────────────────────────────

@dataclass(slots=True)
class StaticValueNode:
    value: t.Any

    def get_expr(self) -> t.Any:
        return pl.lit(self.value)

    def get_frame_value(self, _frames: t.Dict[str, t.Any]) -> t.Any:
        return self.value


@dataclass(slots=True)
class ExternalInputNode:
    input_name: str

    def get_expr(self) -> pl.Expr:
        return pl.col(self.input_name)

    def get_frame_value(self, frames: t.Dict[str, t.Any]) -> t.Any:
        if self.input_name not in frames:
            raise ValueError(
                f"Frame '{self.input_name}' not found. Available: {list(frames.keys())}"
            )
        return frames[self.input_name]


# ── Node ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Node:
    name: str
    callable: t.Callable

    input_map: t.Dict[str, t.Union["Node", StaticValueNode, ExternalInputNode]] = field(
        default_factory=dict
    )

    def get_expr(self) -> pl.Expr:
        return pl.col(self.name)

    def get_input_expressions(self) -> t.Dict[str, t.Any]:
        return {k: ref.get_expr() for k, ref in self.input_map.items()}

    def get_frame_kwargs(self, frames: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        return {k: ref.get_frame_value(frames) for k, ref in self.input_map.items()}

    @property
    def frame_dependencies(self) -> t.List[str]:
        return [
            ref.input_name for ref in self.input_map.values()
            if isinstance(ref, ExternalInputNode)
        ]

    def get_dependencies(self) -> t.List[str]:
        return [ref.name for ref in self.input_map.values() if isinstance(ref, Node)]

    @classmethod
    def from_callable(
        cls,
        func: t.Callable,
        name: t.Optional[str] = None,
        input_map: t.Optional[t.Dict[str, str]] = None,
        static_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> "Node":
        name = name or func.__name__
        params = inspect.signature(func).parameters
        static_kwargs = static_kwargs or {}

        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        named_params = {
            k for k, p in params.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        required = {
            k for k in named_params
            if params[k].default is inspect.Parameter.empty and k not in static_kwargs
        }

        input_map = input_map or {}
        resolved = {k: input_map.get(k, k) for k in required}

        extra = {k for k in input_map if k not in named_params and k not in static_kwargs}
        if extra:
            if not has_var_keyword:
                raise ValueError(
                    f"Parameters {sorted(extra)} are in input_map but '{func.__name__}' "
                    "has no matching parameter and no **kwargs."
                )
            for k in extra:
                resolved[k] = input_map[k]

        return cls(
            name=name,
            callable=func,
            input_map={
                k: ExternalInputNode(input_name=resolved[k]) for k in resolved
            } | {
                k: StaticValueNode(value=v) for k, v in static_kwargs.items()
            },
        )


# ── CompiledExpressions ───────────────────────────────────────────────────────

@dataclass(slots=True)
class CompiledExpressions:
    expressions: OrderedDict  # OrderedDict[str, pl.Expr]
    input_frame: str = "input"
    drop_inputs: bool = False

    def execute(self, inputs: TInputType) -> TOutputType:
        frame = inputs[self.input_frame]
        if isinstance(frame, pl.DataFrame):
            frame = frame.lazy()
        for name, expr in self.expressions.items():
            frame = frame.with_columns(expr.alias(name))
        if self.drop_inputs:
            return frame.select(list(self.expressions.keys()))
        return frame


# ── ExpressionModule ──────────────────────────────────────────────────────────

class ExpressionModule(BaseExecuteModule):
    name: str
    _compiled_expressions: t.Optional[CompiledExpressions] = PrivateAttr(None)

    @abstractmethod
    def expand_nodes(self) -> t.Dict[str, Node]:
        ...

    def compile_expressions(self, executor: t.Optional["Executor"] = None) -> CompiledExpressions:
        if self._compiled_expressions is None:
            from decider.settings import get_default_executor
            executor = executor or get_default_executor()
            nodes = list(self.expand_nodes().values())
            self._compiled_expressions = executor.compile_expression_graph(nodes)
        return self._compiled_expressions

    def execute(self, inputs: TInputType, _executor: "Executor") -> TOutputType:
        if self._compiled_expressions is None:
            raise RuntimeError(
                f"Module '{self.name}' has not been compiled. "
                "Call .compile_expressions() first, or use mod(inputs) which compiles automatically."
            )
        return self._compiled_expressions.execute(inputs)

    def __call__(self, inputs: TInputType, executor: t.Optional["Executor"] = None) -> TOutputType:
        from decider.settings import get_default_executor
        executor = executor or get_default_executor()
        self.compile_expressions(executor)
        return executor.execute(self, inputs)

    def __and__(self, other: "ExpressionModule") -> "ExpressionModule":
        from decider.modules.primitives.union import UnionExpressionModule
        left_modules = self.modules if isinstance(self, UnionExpressionModule) else [self]  # type: ignore[attr-defined]
        right_modules = other.modules if isinstance(other, UnionExpressionModule) else [other]  # type: ignore[attr-defined]
        return UnionExpressionModule(
            name=f"{self.name}__{other.name}",
            modules=left_modules + right_modules,
        )

    def __or__(self, other: "BaseModule") -> "SequentialModule":
        from decider.modules.primitives.sequential import SequentialModule
        return SequentialModule(name=self.name, steps=[self, other])
