"""`each`: run a child flow on every element of a list column, writing the enriched list back.

`each` lowers to a `ScatterGatherNode`: the child is a real IR child (visible to
`structure()` and `parameters()`), and the runner scatters the list into
element rows, runs the child over them, and gathers the result back.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from decider.engine.ir.nodes import CallNode, IRNode, ScatterGatherNode, iter_nodes
from decider.exceptions import IRError
from decider.steps.base import Step, as_step

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


class EachMode(Enum):
    """How an `each` step runs its child over a list column (kept for compatibility; both now scatter)."""

    PER_ROW = "per_row"
    BATCH = "batch"


@dataclass(frozen=True, slots=True, eq=False)
class EachStep(Step):
    """Runs `item` on every element of `column`, adding the child's outputs as fields. Build with `each()`."""

    __module__ = "decider.steps"

    name: str
    column: str
    item: Step
    mode: EachMode
    output: str | None = None
    id: str | None = None

    def to_ir(self, ctx: IRContext) -> ScatterGatherNode:
        child_ir = ctx.child(self.name).build(self.item)
        out_name = self.output if self.output is not None else self.column
        params, hoist = _hoist(child_ir)
        return ScatterGatherNode(ctx.origin(self), self.column, child_ir, None, out_name, params, hoist)


def _hoist(child_ir: IRNode) -> tuple[tuple, tuple[tuple[str, str], ...]]:
    # The child's params become this node's params, so `parameters()` lists them under the node's
    # path and a tuned value reaches the child. `hoist` maps each param name to the child path the
    # runner forwards it into.
    params, hoist, seen = [], [], {}
    for n in iter_nodes(child_ir):
        if not isinstance(n, CallNode):
            continue
        for d in n.params:
            if d.shared_key is not None:
                raise IRError(f"each child: shared param {d.shared_key!r} is not supported")
            if d.name in seen:
                raise IRError(f"each child: two steps have a param named {d.name!r}; rename one")
            seen[d.name] = n.origin.path
            params.append(d)
            hoist.append((d.name, n.origin.path))
    return tuple(params), tuple(hoist)


def each(column: str, item: Any, *, name: str | None = None,
         execution_mode: EachMode = EachMode.PER_ROW, output: str | None = None,
         id: str | None = None) -> EachStep:
    """Run `item` on every element of the list column `column`, writing the enriched list back.

    `item` is a flow of ordinary steps reading the item's fields; its outputs
    are added as new fields of each item. The list is read under `column` and,
    by default, written back to it in place.

    Args:
        output: the column to write the enriched list to (default: `column`,
            overwriting it). Set this to keep the original list untouched
            alongside the enriched one.

    Example::

        def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
            return weight > heavy_kg

        pipeline = flow(each("items", flow(heavy, name="item"), name="items"), name="order")
    """
    return EachStep(column if name is None else name, column, as_step(item), execution_mode, output, id=id)
