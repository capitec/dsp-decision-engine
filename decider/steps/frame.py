from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from decider.engine.ir.decls import Input, Output
from decider.engine.ir.nodes import CallNode
from decider.steps.base import Step

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


@dataclass(frozen=True, slots=True, eq=False)
class FrameStep(Step):
    """A `DataFrame -> DataFrame` step (joins, filters, model calls). Build one with `frame_step()`."""

    __module__ = "decider.steps"

    name: str
    fn: Callable
    inputs: tuple[str, ...] | None = None
    outputs: tuple[str, ...] | None = None
    reads: tuple[tuple[str, str], ...] = ()
    writes: tuple[tuple[str, str], ...] = ()

    def __call__(self, df: Any) -> Any:
        return self.fn(df)

    def to_ir(self, ctx: IRContext) -> CallNode:
        inputs = None if self.inputs is None else tuple(Input(n, Any) for n in self.inputs)
        outputs = None if self.outputs is None else tuple(Output(n, Any) for n in self.outputs)
        return CallNode(ctx.origin(self), "frame", self.fn, inputs, outputs, ())


def frame_step(
    fn: Callable | None = None,
    /,
    *,
    name: str | None = None,
    reads: list[str] | None = None,
    writes: list[str] | None = None,
) -> Any:
    """Make a `DataFrame -> DataFrame` function a step, directly or as a decorator.

    Args:
        reads: the columns it reads. `None` means unknown: names read after it
            are checked against its actual output when it runs.
        writes: the columns it adds or replaces; `None` means unknown.

    Example::

        @frame_step(reads=["client_id"], writes=["bureau_score"])
        def join_bureau(df: pl.DataFrame) -> pl.DataFrame:
            return df.join(BUREAU, on="client_id", how="left")
    """

    def make(fn: Callable) -> FrameStep:
        return FrameStep(
            name or fn.__name__, fn,
            None if reads is None else tuple(reads), None if writes is None else tuple(writes),
        )

    return make if fn is None else make(fn)
