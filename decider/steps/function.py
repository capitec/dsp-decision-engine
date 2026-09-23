from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

from pydantic.fields import FieldInfo

from decider.engine.ir.decls import ParamDecl
from decider.engine.ir.nodes import CallNode
from decider.engine.params import call_with_defaults, harvest
from decider.steps.base import Step

if TYPE_CHECKING:
    from decider.engine.ir.context import IRContext


@dataclass(frozen=True, slots=True, eq=False)
class FunctionStep(Step):
    """One scalar function as a step; still callable as the plain function.

    Inputs, params and null policies come from the signature; `outputs`
    defaults to the function's name. Build one with `step()`.
    """

    # Origins name the public import path.
    __module__ = "decider.steps"

    name: str
    fn: Callable
    outputs: tuple[str, ...]
    nogil: bool = False
    reads: tuple[tuple[str, str], ...] = ()
    writes: tuple[tuple[str, str], ...] = ()
    bound: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        harvest(self.fn, self.outputs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return call_with_defaults(self.fn, *args, **{**dict(self.bound), **kwargs})

    def bind(self, **values: Any) -> FunctionStep:
        """A copy whose params default to `values`.

        Example::

            strict = cap_by_income.bind(cap=36.0)
        """
        names = [p.name for p in harvest(self.fn, self.outputs)[1]]
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise ValueError(f"{self.name}.bind(): unknown param(s) {unknown}; its params are {names}")
        return replace(self, bound=tuple({**dict(self.bound), **values}.items()))

    def to_ir(self, ctx: IRContext) -> CallNode:
        inputs, params, outputs = harvest(self.fn, self.outputs)
        bound = dict(self.bound)
        params = tuple(_bind(p, bound[p.name]) if p.name in bound else p for p in params)
        return CallNode(ctx.origin(self), "scalar", self.fn, inputs, outputs, params, nogil=self.nogil)


def _bind(decl: ParamDecl, value: Any) -> ParamDecl:
    info = None if decl.field_info is None else FieldInfo.merge_field_infos(decl.field_info, default=value)
    return replace(decl, default=value, required=False, field_info=info)


def step(
    fn: Callable | None = None,
    /,
    *,
    name: str | None = None,
    output: str | None = None,
    outputs: tuple[str, ...] | None = None,
    nogil: bool = False,
) -> Any:
    """Make a function a step, directly or as a decorator.

    Only needed to override defaults: plain functions convert wherever a step
    is expected. Two plain functions can't be joined with `|` (Python has no
    `|` for functions): write `step(f) | g` or `flow(f, g)`.

    Args:
        name: the step's name in paths (default: the function's name).
        output: the one name it writes (default: the function's name). Several
            steps writing one name in a `flow` is a waterfall: later wins.
        outputs: several names it writes; the return annotation must be a
            `tuple[...]` of the same length.
        nogil: release the GIL in compiled code.

    Example::

        @step(output="term_cap")
        def cap_by_income(term_cap: float, cap: float = param(48.0)) -> float:
            return min(term_cap, cap)

        @step(outputs=("band", "band_score"))
        def banding(ratio: float) -> tuple[int, float]:
            return (1, 10.0) if ratio > 2 else (0, 0.0)
    """
    if output is not None and outputs is not None:
        raise TypeError("step() takes output= or outputs=, not both")

    def make(fn: Callable) -> FunctionStep:
        names = (output,) if output is not None else tuple(outputs) if outputs is not None else (fn.__name__,)
        return FunctionStep(name or fn.__name__, fn, names, nogil)

    return make if fn is None else make(fn)
