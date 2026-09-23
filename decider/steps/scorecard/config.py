from __future__ import annotations

import math
import typing as t

import numpy as np
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator, model_validator

from decider.engine.ir.decls import Input, NullPolicy, Output, ParamDecl
from decider.engine.ir.nodes import CallNode, SequenceNode
from decider.steps.configurable import ConfigurableStep
from decider.steps.scorecard import kernels
from decider.steps.values import ParamRef, Value


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DefaultBin(_Model):
    """The points a variable scores when no bin matches, or its input is null.

    Example::

        {"value": 0, "name": "unknown"}
    """

    value: Value[float]
    name: t.Optional[str] = None


class BoundBin(DefaultBin):
    """Points for `lower_bound < x <= upper_bound`; a missing bound is taken from the neighbouring bin, or is open.

    Example::

        {"value": 10, "lower_bound": 25, "upper_bound": {"param": "old", "default": 60}}
    """

    lower_bound: t.Optional[Value[float]] = None
    upper_bound: t.Optional[Value[float]] = None


class ValuesBin(DefaultBin):
    """Points for an input equal to one of `items`; checked before any `BoundBin`.

    Example::

        {"value": 15, "items": ["gold", "platinum"]}
    """

    items: list[t.Union[int, float, str]] = Field(default_factory=list)


def _bin_tag(b: t.Any) -> str:
    return "values" if (("items" in b) if isinstance(b, dict) else isinstance(b, ValuesBin)) else "bound"


Bin = t.Annotated[
    t.Union[t.Annotated[BoundBin, Tag("bound")], t.Annotated[ValuesBin, Tag("values")]], Discriminator(_bin_tag)
]


def _num(v: t.Any) -> t.Any:
    return v.default if isinstance(v, ParamRef) else v


def _split(ctx: t.Any, **values: t.Any) -> tuple[tuple, tuple]:
    resolved = {arg: ctx.value(v, float, arg=arg) for arg, v in values.items()}
    return (tuple(v for v in resolved.values() if isinstance(v, ParamDecl)),
            tuple((a, v) for a, v in resolved.items() if not isinstance(v, ParamDecl)))


class ScoredVariable(_Model):
    """One input binned into points, written to `value_output_name` (`{variable_name}` is filled in).

    With `strict` (the default) consecutive bound bins must meet exactly; otherwise gaps score the default.

    Example::

        {"type": "scored", "variable_name": "age", "default": {"value": 0},
         "bins": [{"value": 5, "upper_bound": 25}, {"value": 10, "lower_bound": 25}]}
    """

    type: t.Literal["scored"] = "scored"
    variable_name: str
    bins: list[Bin]
    default: DefaultBin
    strict: bool = True
    value_output_name: str = "{variable_name}_score"

    @model_validator(mode="after")
    def _check_bins(self) -> ScoredVariable:
        seen: set = set()
        last, last_i, highest = None, -1, -math.inf
        for i, b in enumerate(self.bins):
            if isinstance(b, ValuesBin):
                dup = seen & set(b.items)
                if dup:
                    raise ValueError(f"Duplicate values {dup} found in ValuesBin at index {i}")
                seen |= set(b.items)
                continue
            lo, hi = _num(b.lower_bound), _num(b.upper_bound)
            if lo is not None and hi is not None and lo >= hi:
                raise ValueError(f"BoundBin at index {i} has lower_bound >= upper_bound")
            for label, bound in (("lower_bound", lo), ("upper_bound", hi)):
                if bound is not None:
                    if bound < highest:
                        raise ValueError(
                            f"BoundBin at index {i} has {label} {bound} which is less than previous highest_bound "
                            f"{highest}. Bound bins must be in order and non-overlapping."
                        )
                    highest = bound
            if last is not None:
                prev = _num(last.upper_bound)
                if prev is not None and lo is not None and ((self.strict and lo != prev) or lo < prev):
                    raise ValueError(
                        f"BoundBin at index {i} lower_bound ({lo}) must be "
                        f"{'equal to (in strict mode)' if self.strict else 'greater than or equal to'} previous "
                        f"upper_bound ({prev}) defined in BoundBin at index {last_i}. "
                        "Bound bins must be in order and non-overlapping."
                    )
                if last.upper_bound is None and b.lower_bound is None:
                    raise ValueError(f"Either BoundBin at index {last_i} must define a upper_bound "
                                     f"or BoundBin at index {i} must have a lower_bound.")
            last, last_i = b, i
        return self

    def output(self) -> str:
        return self.value_output_name.format(variable_name=self.variable_name)

    def nodes(self, ctx: t.Any, owner: ConfigurableStep) -> list[CallNode]:
        bound = [b for b in self.bins if isinstance(b, BoundBin)]
        values = [b for b in self.bins if isinstance(b, ValuesBin)]
        nb = len(bound)
        # `or`, not `is None`: a 0 bound counts as unset, as it always has, so existing scorecards score the same.
        lows = [b.lower_bound or (bound[i - 1].upper_bound if i else None) for i, b in enumerate(bound)]
        highs = [b.upper_bound or (bound[i + 1].lower_bound if i + 1 < nb else None) for i, b in enumerate(bound)]
        slots = [*lows, *highs, *(b.value for b in bound), *(b.value for b in values), self.default.value]
        decls: list[ParamDecl] = []
        lits, where = [], []
        for v in slots:
            r = ctx.value(v, float)
            if isinstance(r, ParamDecl):
                if r not in decls:
                    decls.append(r)
                where.append(decls.index(r))
                r = None
            else:
                where.append(-1)
            lits.append(math.nan if r is None else float(r))
        items = [x for b in values for x in b.items]
        text = any(isinstance(x, str) for x in items)
        consts = [("nb", nb), ("vals", np.array(lits)), ("items", np.array(items, object if text else np.float64)),
                  ("item_bin", np.array([k for k, b in enumerate(values) for _ in b.items], np.int64))]
        if decls:
            consts.append(("slots", np.array(where, np.int64)))
        x = Input(self.variable_name, str | None if text else float | None, NullPolicy.OPTIONAL, arg="x")
        return [CallNode(
            ctx.origin(owner, self.variable_name), "row", kernels.score_bins_tuned if decls else kernels.score_bins,
            (x,), (Output(self.output(), float),), tuple(decls), consts=tuple(consts),
        )]


class AdjustedVariable(_Model):
    """A scored variable rescaled as `score * scale + offset`, written to `variable_output_name`.

    `{variable_name}` and `{score_value_output_name}` are filled in; the unadjusted score is written too.

    Example::

        {"type": "adjusted", "scale": 2.0, "offset": -5.0, "variable": {"type": "scored", ...}}
    """

    type: t.Literal["adjusted"] = "adjusted"
    variable: ScoredVariable
    offset: Value[float] = 0.0
    scale: Value[float] = 1.0
    variable_output_name: str = "{variable_name}_adjusted_score"

    @property
    def variable_name(self) -> str:
        return self.variable.variable_name

    def output(self) -> str:
        return self.variable_output_name.format(
            variable_name=self.variable.variable_name, score_value_output_name=self.variable.output())

    def nodes(self, ctx: t.Any, owner: ConfigurableStep) -> list[CallNode]:
        params, consts = _split(ctx, scale=self.scale, offset=self.offset)
        node = CallNode(ctx.origin(owner, self.output()), "scalar", kernels.adjust,
                        (Input(self.variable.output(), float, arg="score"),), (Output(self.output(), float),),
                        params, consts=consts)
        return [*self.variable.nodes(ctx, owner), node]


class ConstantScore(_Model):
    """The same points for every row, written to `output_name`.

    Example::

        {"type": "constant", "score": 600, "output_name": "base_score"}
    """

    type: t.Literal["constant"] = "constant"
    score: Value[float]
    output_name: str = "constant_score"

    def output(self) -> str:
        return self.output_name

    def nodes(self, ctx: t.Any, owner: ConfigurableStep) -> list[CallNode]:
        params, consts = _split(ctx, score=self.score)
        return [CallNode(ctx.origin(owner, self.output_name), "scalar", kernels.constant, (),
                         (Output(self.output_name, float),), params, consts=consts)]


Variable = t.Annotated[t.Union[ScoredVariable, AdjustedVariable, ConstantScore], Field(discriminator="type")]


class ScorecardConfig(ConfigurableStep):
    """A points scorecard: each variable's points, and their sum in `output_name`.

    Every bound and every points value is a `Value[float]`, so moving a bin
    edge through the params document never rebuilds anything. Bins are
    checked in order, values bins first; a null input scores the default.
    A variable's params live under `<scorecard>/<variable_name>`.

    Example::

        card = ScorecardConfig.load({
            "type": "scorecard", "name": "card",
            "variables": [{
                "type": "scored", "variable_name": "age",
                "bins": [{"value": 5, "upper_bound": {"param": "young", "default": 25}},
                         {"value": 10, "lower_bound": {"param": "young", "default": 25}}],
                "default": {"value": 0},
            }],
        })
        card.run(df)                                              # age_score and score
        card.run(df, params={"card": {"age": {"young": 30.0}}})   # retuned edge
    """

    type: t.Literal["scorecard"] = "scorecard"
    variables: list[Variable] = Field(min_length=1)
    output_name: str = "score"

    @field_validator("variables")
    @classmethod
    def _unique_names(cls, variables: list) -> list:
        names = [v.variable_name for v in variables if not isinstance(v, ConstantScore)]
        dup = next((n for i, n in enumerate(names) if n in names[:i]), None)
        if dup is not None:
            raise ValueError(f"Duplicate variable_name '{dup}' found in ScoreCard variables. "
                             "Each ScoredVariable must have a unique variable_name.")
        return variables

    def to_ir(self, ctx: t.Any) -> SequenceNode:
        inner = ctx.child(self.name)
        nodes = [n for v in self.variables for n in v.nodes(inner, self)]
        scores = tuple(Input(v.output(), float) for v in self.variables)
        nodes.append(CallNode(inner.origin(self, self.output_name), "row", kernels.total, scores,
                              (Output(self.output_name, float),), ()))
        # Every variable's points are output, not just the total, as reason codes.
        return SequenceNode(ctx.origin(self), tuple(nodes), tuple(o.name for n in nodes for o in n.outputs))
