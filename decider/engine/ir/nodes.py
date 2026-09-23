from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, Literal

from decider.engine.ir.decls import Input, Output, ParamDecl
from decider.engine.ir.origin import Origin


class IRNode(ABC):
    """A node of the IR: one of `CallNode`, `SequenceNode`, `BranchNode`, `LoopNode`.

    Every node carries its `origin`; `children()` lists the nodes it runs.
    """

    __slots__ = ()
    origin: Origin

    @abstractmethod
    def children(self) -> tuple[IRNode, ...]: ...


@dataclass(frozen=True, slots=True, eq=False)
class CallNode(IRNode):
    """Calls one function.

    - `scalar`: `fn(*inputs, **params)`, per row. `inputs[i]` feeds the i-th
      argument of `fn` that is not a param, so a relabelled input still
      reaches the right argument.
    - `row`: `fn(row, params)`, both tuples in declared order; returns a tuple
      of outputs. `reference(row, params, visit)` is its plain-Python twin.
    - `frame`: `fn(df) -> df`. `inputs`/`outputs` of `None` mean unknown
      lineage: names read after it are checked at run time.

    Example::

        CallNode(origin, "scalar", ratio, (Input("income", float),), (Output("ratio", float),), ())
    """

    origin: Origin
    kind: Literal["scalar", "row", "frame"]
    fn: Callable
    inputs: tuple[Input, ...] | None
    outputs: tuple[Output, ...] | None
    params: tuple[ParamDecl, ...]
    reference: Callable | None = None
    nogil: bool = False

    def children(self) -> tuple[IRNode, ...]:
        return ()


@dataclass(frozen=True, slots=True, eq=False)
class SequenceNode(IRNode):
    """Runs its children in order. `emits`/`drops` are `name` or `name@path` specs."""

    origin: Origin
    children_: tuple[IRNode, ...]
    emits: tuple[str, ...] = ()
    drops: tuple[str, ...] = ()

    def children(self) -> tuple[IRNode, ...]:
        return self.children_


@dataclass(frozen=True, slots=True, eq=False)
class BranchNode(IRNode):
    """Runs the arm `condition` picks per row: a bool picks arm 0 (true) or 1, an int picks by index."""

    origin: Origin
    condition: CallNode
    arms: tuple[IRNode, ...]
    modifies: tuple[str, ...]

    def children(self) -> tuple[IRNode, ...]:
        return (self.condition, *self.arms)


@dataclass(frozen=True, slots=True, eq=False)
class LoopNode(IRNode):
    """Runs `body` while `condition` holds, at most `max_iterations` times, carrying `carries`."""

    origin: Origin
    condition: CallNode
    body: IRNode
    carries: tuple[str, ...]
    max_iterations: int

    def children(self) -> tuple[IRNode, ...]:
        return (self.condition, self.body)


def iter_nodes(node: IRNode) -> Iterator[IRNode]:
    """Every node of an IR tree, parents before children.

    Example::

        paths = [n.origin.path for n in iter_nodes(ir)]
    """
    yield node
    for child in node.children():
        yield from iter_nodes(child)
