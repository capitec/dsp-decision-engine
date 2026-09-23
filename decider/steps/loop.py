from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from decider.engine.ir.nodes import LoopNode
from decider.exceptions import WiringError
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
    """Run `body` while `condition` holds, checked per row before each iteration.

    The body may be one step or a flow of several. A row that reaches
    `max_iterations` stops there, without an error.

    Args:
        carries: the names each iteration updates: the condition and body
            see the latest values, the body writes every one of them, and
            they are all that leaves the loop.
        max_iterations: the most iterations any row may take; required, since
            compiled code can't be interrupted.

    Example::

        def unpaid(balance: float) -> bool:
            return balance > 0.0

        @step(output="balance")
        def pay(balance: float, payment: float) -> float:
            return balance * 1.01 - payment

        @step(output="months")
        def tick(months: int) -> int:
            return months + 1

        repay = loop(unpaid, flow(pay, tick, name="month"), carries=["balance", "months"],
                     max_iterations=360, name="repay")
    """
    if not carries:
        raise WiringError(f"loop {name!r} needs carries=[...]: the names each iteration updates")
    if type(max_iterations) is not int or max_iterations < 1:
        raise WiringError(f"loop {name!r}: max_iterations must be a positive int, got {max_iterations!r}")
    body = as_step(body)
    # An anonymous body would share the loop's path.
    if body.name is None:
        body = body.named("body")
    return LoopStep(name, as_step(condition), body, tuple(carries), max_iterations)
