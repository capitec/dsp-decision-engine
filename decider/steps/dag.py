from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from decider.engine.ir.nodes import IRNode, SequenceNode
from decider.engine.wiring.interface import interface
from decider.exceptions import WiringError
from decider.steps.base import Step, as_step
from decider.steps.sequential import SequentialStep

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


@dataclass(frozen=True, slots=True, eq=False)
class DagStep(SequentialStep):
    """Members run in dependency order; each name has one writer. Build one with `dag(...)`."""

    __module__ = "decider.steps"

    def to_ir(self, ctx: IRContext) -> SequenceNode:
        inner = ctx.child(self.name)
        nodes = [inner.build(s) for s in self.steps]
        order = self._order(nodes)
        return self._sequence(ctx, [self.steps[i] for i in order], [nodes[i] for i in order])

    def _label(self, i: int) -> str:
        return self.steps[i].name or type(self.steps[i]).__name__

    def _order(self, nodes: list[IRNode]) -> list[int]:
        faces = [interface(n) for n in nodes]
        where = f"dag {self.name!r}" if self.name else "dag"
        writer: dict[str, int] = {}
        clashes = []
        for i, (reads, writes) in enumerate(faces):
            for name in sorted(writes):
                if name not in writer:
                    writer[name] = i
                    continue
                first, second = self._label(writer[name]), self._label(i)
                # Order matters only when one refines the other's value (reads what it writes).
                fix = (f"use flow({first}, {second}) to apply them in that order, the later winning"
                       if name in reads or name in faces[writer[name]][0]
                       else f"rename one, e.g. {second}.relabel(writes={{{name!r}: {name + '_2'!r}}})")
                clashes.append(f"{first} and {second} both write {name!r}: {fix}")
        if clashes:
            raise WiringError(f"{where}: each name has one writer in a dag.\n  " + "\n  ".join(clashes))
        # Kahn's algorithm, always taking the earliest-written ready member,
        # so members already in dependency order keep their written order.
        left, order = list(range(len(nodes))), []
        while left:
            for i in left:
                if not any(faces[i][0] & faces[j][1] for j in left if j != i):
                    break
            else:
                raise WiringError(f"{where}: {[self._label(i) for i in left]} depend on each other in a cycle")
            left.remove(i)
            order.append(i)
        return order


def dag(*steps: Any, name: str | None = None) -> Step:
    """Run steps in dependency order: a step reading a name runs after the step writing it.

    Two members writing one name is an error: use `flow` for a waterfall, or
    `.relabel(writes=...)` one of them when they are different values. A
    single unnamed step is returned unchanged. As with `flow`, intermediates
    are dropped from the output unless kept with `.emit(...)`; `.drop(...)`
    removes columns.

    Example::

        affordability = dag(affordable, ratio, disposable_income, name="affordability")
        affordability.emit("ratio", "disposable_income").run(df)
    """
    if not steps:
        raise WiringError("dag() needs at least one step")
    if len(steps) == 1 and name is None:
        return as_step(steps[0])
    return DagStep(tuple(map(as_step, steps)), name)
