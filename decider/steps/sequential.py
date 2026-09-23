from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

from decider.engine.ir.nodes import IRNode, SequenceNode
from decider.steps.base import Step, as_step

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


@dataclass(frozen=True, slots=True, eq=False)
class SequentialStep(Step):
    """Members run in written order; a later write of a name wins (the waterfall).

    Build one with `flow(...)` or `a | b | c`.
    """

    __module__ = "decider.steps"

    steps: tuple[Step, ...]
    name: str | None = None
    emits: tuple[str, ...] = ()
    drops: tuple[str, ...] = ()
    reads: tuple[tuple[str, str], ...] = ()
    writes: tuple[tuple[str, str], ...] = ()

    def emit(self, *names: str) -> SequentialStep:
        """A copy that also outputs these values: `name`, `name@path` (one version) or `name@*` (every version).

        Example::

            pipeline.emit("disposable_income", "term_cap@*")
        """
        return replace(self, emits=tuple(dict.fromkeys(self.emits + names)))

    def drop(self, *names: str) -> SequentialStep:
        """A copy that leaves these columns out of its output.

        Example::

            pipeline.drop("min_net_salary")
        """
        return replace(self, drops=tuple(dict.fromkeys(self.drops + names)))

    def to_ir(self, ctx: IRContext) -> SequenceNode:
        inner = ctx.child(self.name)
        return self._sequence(ctx, self.steps, [inner.build(s) for s in self.steps])

    def _sequence(self, ctx: IRContext, steps: Sequence[Step], nodes: Sequence[IRNode]) -> SequenceNode:
        children, emits, drops = [], list(self.emits), list(self.drops)
        for s, node in zip(steps, nodes):
            # An anonymous member is transparent: its children join this sequence.
            if s.name is None and isinstance(node, SequenceNode):
                children += node.children_
                emits += node.emits
                drops += node.drops
            else:
                children.append(node)
        return SequenceNode(ctx.origin(self), tuple(children), tuple(dict.fromkeys(emits)), tuple(dict.fromkeys(drops)))


def flow(*steps: Any, name: str | None = None) -> SequentialStep:
    """Run steps in written order; a later write of a name wins.

    `a | b | c` builds the same flow. Anonymous flows inside are merged in; a
    named flow stays one unit (its members sit under its name). Plain
    functions have no `|`, so a flow of only functions is written `flow(f, g)`
    or `step(f) | g`.

    Example::

        term = flow(term_cap, cap_by_income, cap_by_sector, name="term")
    """
    if not steps:
        raise ValueError("flow() needs at least one step")
    members: list[Step] = []
    emits: tuple[str, ...] = ()
    drops: tuple[str, ...] = ()
    for s in map(as_step, steps):
        if type(s) is SequentialStep and s.name is None and not s.reads and not s.writes:
            members += s.steps
            emits += s.emits
            drops += s.drops
        else:
            members.append(s)
    return SequentialStep(tuple(members), name, tuple(dict.fromkeys(emits)), tuple(dict.fromkeys(drops)))
