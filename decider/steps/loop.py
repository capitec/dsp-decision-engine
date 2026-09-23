from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from decider.engine.ir.nodes import LoopNode
from decider.steps.base import Step, as_step
from decider.steps.branch import condition_node

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


@dataclass(frozen=True, slots=True, eq=False)
class LoopStep(Step):
    """Runs a body while a condition holds. Build one with `loop()`."""

    __module__ = "decider.steps"

    name: str
    condition: Step
    body: Step
    carries: tuple[str, ...]
    max_iterations: int
    reads: tuple[tuple[str, str], ...] = ()
    writes: tuple[tuple[str, str], ...] = ()

    def to_ir(self, ctx: IRContext) -> LoopNode:
        inner = ctx.child(self.name)
        condition = condition_node(inner, self.condition, f"loop {self.name!r}")
        return LoopNode(ctx.origin(self), condition, inner.build(self.body), self.carries, self.max_iterations)


def loop(condition: Any, body: Any, *, carries: Sequence[str], max_iterations: int, name: str) -> LoopStep:
    """Run `body` while `condition` holds, checked before each iteration.

    Args:
        carries: the names the body reads at the start of an iteration and
            writes by its end.
        max_iterations: the most iterations any row may take; required, since
            compiled code can't be interrupted.

    Example::

        best = loop(should_continue, improve_offer, carries=["best_offer"], max_iterations=50, name="best")
    """
    if not carries:
        raise ValueError(f"loop {name!r} needs carries=[...]: the names each iteration updates")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError(f"loop {name!r}: max_iterations must be a positive int, got {max_iterations!r}")
    body = as_step(body)
    # An anonymous body would share the loop's path.
    if body.name is None:
        body = body.named("body")
    return LoopStep(name, as_step(condition), body, tuple(carries), max_iterations)
