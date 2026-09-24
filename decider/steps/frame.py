from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from decider.engine.ir.decls import Input, Output
from decider.engine.ir.nodes import CallNode
from decider.engine.params import call_with_defaults, harvest
from decider.steps.base import Step

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


@dataclass(frozen=True, slots=True, eq=False)
class FrameStep(Step):
    """A `DataFrame -> DataFrame` step (joins, filters, model calls). Build one with `frame_step()`.

    Still callable as the plain function, with `param()` defaults filled in.
    """

    __module__ = "decider.steps"

    name: str
    fn: Callable
    inputs: tuple[str, ...] | None = None
    outputs: tuple[str, ...] | None = None
    reads: tuple[tuple[str, str], ...] = ()
    writes: tuple[tuple[str, str], ...] = ()
    annotations: tuple[tuple[str, Any], ...] = ()

    def __call__(self, df: Any, **params: Any) -> Any:
        return call_with_defaults(self.fn, df, **params)

    def to_ir(self, ctx: IRContext) -> CallNode:
        types = dict(self.annotations)
        inputs = None if self.inputs is None else tuple(Input(n, types.get(n, Any)) for n in self.inputs)
        outputs = None if self.outputs is None else tuple(Output(n, Any) for n in self.outputs)
        return CallNode(ctx.origin(self), "frame", self.fn, inputs, outputs, harvest(self.fn)[1])


def frame_step(
    fn: Callable | None = None,
    /,
    *,
    name: str | None = None,
    reads: list[str] | dict[str, Any] | None = None,
    writes: list[str] | None = None,
) -> Any:
    """Make a `DataFrame -> DataFrame` function a step, directly or as a decorator.

    Arguments after the frame are `param()`s, set from the params document
    like a plain step's. With both `reads` and `writes` declared, the step can
    be `.relabel()`ed.

    Args:
        reads: the columns it reads. `None` means unknown: names read after it
            are checked against its actual output when it runs. A dict gives
            their types, as a plain step's annotations do: a served JSON
            request's ISO strings become `date`s where a type says so
            (`{"accounts": list[Account]}`, `Account` a TypedDict with a
            `date` field).
        writes: the columns it adds or replaces; `None` means unknown.

    Example::

        @frame_step(reads=["client_id"], writes=["bureau_score"])
        def join_bureau(df: pl.DataFrame, floor: float = param(300.0)) -> pl.DataFrame:
            return df.join(BUREAU, on="client_id", how="left").with_columns(
                pl.col("bureau_score").clip(lower_bound=floor))
    """

    def make(fn: Callable) -> FrameStep:
        return FrameStep(
            fn.__name__ if name is None else name, fn,
            None if reads is None else tuple(reads), None if writes is None else tuple(writes),
            annotations=tuple(reads.items()) if isinstance(reads, dict) else (),
        )

    return make if fn is None else make(fn)
