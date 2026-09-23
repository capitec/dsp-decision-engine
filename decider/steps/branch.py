from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from decider.engine.ir.nodes import BranchNode, CallNode
from decider.exceptions import IRError, WiringError
from decider.steps.base import Step, as_step

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


def condition_node(ctx: IRContext, condition: Step, where: str) -> CallNode:
    """Build a condition's IR, which must be one call producing one value."""
    node = ctx.build(condition)
    if not isinstance(node, CallNode) or len(node.outputs or ()) != 1:
        raise IRError(f"{where}: the condition must be one function step producing one value")
    return node


@dataclass(frozen=True, slots=True, eq=False)
class BranchStep(Step):
    """Runs one of its arms per row, chosen by a condition. Build one with `branch()`."""

    __module__ = "decider.steps"

    name: str
    condition: Step
    arms: tuple[Step, ...]
    modifies: tuple[str, ...]
    reads: tuple[tuple[str, str], ...] = ()
    writes: tuple[tuple[str, str], ...] = ()

    def to_ir(self, ctx: IRContext) -> BranchNode:
        inner = ctx.child(self.name)
        condition = condition_node(inner, self.condition, f"branch {self.name!r}")
        if condition.outputs[0].annotation is bool and len(self.arms) != 2:
            raise WiringError(f"branch {self.name!r}: a bool condition takes 2 arms, got {len(self.arms)}")
        return BranchNode(ctx.origin(self), condition, tuple(inner.build(a) for a in self.arms), self.modifies)


def branch(condition: Any, *arms: Any, modifies: Sequence[str], name: str) -> BranchStep:
    """Run one arm per row: a bool condition picks the first arm (true) or the second; an int picks by index.

    A bool condition sends `True` to arm 0, while an int picks arm `i`, so
    switching a flag from bool to 0/1 swaps the arms. An arm may be one step
    or a flow of several; the arms of one branch must agree on the type of
    each name they modify.

    Args:
        modifies: the names the branch passes on; they keep their earlier
            value on rows whose arm doesn't write them. Anything else an arm
            (or the condition) writes stays inside the branch: reading it
            after the branch, or emitting it by bare name, is an error, but
            `.emit("name@path")` on the enclosing flow still outputs it.

    Example::

        by_sector = branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector")
        flow(term_cap, by_sector, name="term").emit("is_private@term/by_sector/is_private")
    """
    if len(arms) < 2:
        raise WiringError(f"branch {name!r} needs at least two arms, got {len(arms)}")
    if not modifies:
        raise WiringError(f"branch {name!r} needs modifies=[...]: the names its arms may change")
    # An anonymous arm would share the branch's path, so it is named by position.
    steps = tuple(a if a.name is not None else a.named(f"arm{i}") for i, a in enumerate(map(as_step, arms)))
    return BranchStep(name, as_step(condition), steps, tuple(modifies))
