from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal

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

    - `scalar`: `fn(**{i.arg: value for i in inputs}, **dict(consts), **params)`,
      per row. Inputs go by argument name (`Input.arg`), so a relabelled input
      still reaches its argument, keyword-only arguments included.
    - `row`: `fn(row, params, consts)`, all three tuples in declared order
      (`consts` holds the values only); returns a tuple of outputs.
      `reference(row, params, consts, visit)` is its plain-Python twin.
    - `frame`: `fn(df) -> df`. `inputs`/`outputs` of `None` mean unknown
      lineage: names read after it are checked at run time.

    `consts` are named literal arguments, such as a config's inline
    `Value[T]`. Like params they arrive as arguments at run time, never baked
    into compiled code, so changing one never recompiles.

    Example::

        CallNode(origin, "scalar", ratio, (Input("income", float),), (Output("ratio", float),), ())
        CallNode(origin, "row", kernel, inputs, outputs, (), consts=(("threshold", 0.5),))
    """

    origin: Origin
    kind: Literal["scalar", "row", "frame"]
    fn: Callable
    inputs: tuple[Input, ...] | None
    outputs: tuple[Output, ...] | None
    params: tuple[ParamDecl, ...]
    reference: Callable | None = None
    nogil: bool = False
    consts: tuple[tuple[str, Any], ...] = ()

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
