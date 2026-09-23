"""Features, thresholds and the conditions a tree node tests."""
from __future__ import annotations

import enum
import typing as t

from pydantic import BaseModel, BeforeValidator, Discriminator, Field, PrivateAttr, RootModel, Tag, model_validator

from decider.steps import expr
from decider.steps.values import ParamRef, Value


class RangeEndLogic(str, enum.Enum):
    """Which end of a range band is closed: `[lo, hi)` or `(lo, hi]`."""

    lower_inclusive = "lower_inclusive"
    upper_inclusive = "upper_inclusive"


class StringMatchType(str, enum.Enum):
    exact = "exact"
    starts_with = "starts_with"
    contains = "contains"
    ends_with = "ends_with"
    regex = "regex"


class NullHandling(str, enum.Enum):
    """What a string match does with a null string.

    `no_match` (the default): the match is false. `match`: the match is true.
    `error`: a null in the column is a `MissingInputError`.
    """

    match = "match"
    no_match = "no_match"
    error = "error"


class LogicOp(str, enum.Enum):
    AND = "and"
    OR = "or"
    NOT = "not"


def _input_ref(value: t.Any) -> t.Any:
    # Older documents spell a param reference `{"key": "name"}`.
    if isinstance(value, dict) and set(value) == {"key"}:
        return ParamRef(param=value["key"])
    return value


Number = t.Union[int, float]
Threshold = t.Annotated[Value[Number], BeforeValidator(_input_ref)]
"""A literal number or a `ParamRef`; `{"key": "x"}` is read as `{"param": "x"}`."""
Pattern = t.Annotated[Value[str], BeforeValidator(_input_ref)]
Patterns = t.Annotated[t.List[Pattern], Field(min_length=1)]
Values = t.Annotated[
    t.Union[t.Annotated[t.List[Threshold], Field(min_length=1)], ParamRef], BeforeValidator(_input_ref)
]


def get_field(value: t.Any, key: str, default: t.Any = None) -> t.Any:
    """`value[key]` for a dict, `value.key` otherwise; for discriminating raw and validated input alike.

    >>> get_field({"type": "leaf"}, "type"), get_field(object(), "type", "unary")
    ('leaf', 'unary')
    """
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def find(obj: t.Any, kind: type) -> t.Iterator[t.Any]:
    """Yield every `kind` instance nested anywhere in `obj`."""
    if isinstance(obj, kind):
        yield obj
    elif isinstance(obj, BaseModel):
        for value in obj.__dict__.values():
            yield from find(value, kind)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from find(value, kind)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from find(value, kind)


class SchemaModel(BaseModel):
    def required_features(self) -> set[str]:
        """Every input column this part of the tree reads."""
        return set().union(*(f.required_features() for f in find(self, Feature)))

    def required_params(self) -> set[str]:
        """The name of every param this part of the tree references."""
        return {ref.param for ref in find(self, ParamRef)}


class ComputedFeature(BaseModel):
    """A feature computed from other columns, e.g. `{"type": "computed", "expression": "x - y"}`."""

    type: t.Literal["computed"] = "computed"
    expression: str

    _expr: expr.Expr = PrivateAttr()

    @model_validator(mode="after")
    def _parse(self) -> ComputedFeature:
        try:
            self._expr = expr.parse(self.expression)
        except expr.ExprError as e:
            raise ValueError(f"computed feature {self.expression!r} is not a valid expression: {e}") from e
        return self

    @property
    def expr(self) -> expr.Expr:
        return self._expr


class Feature(RootModel[t.Union[ComputedFeature, str]]):
    """A column name, or a `ComputedFeature`."""

    def __str__(self) -> str:
        return self.root if isinstance(self.root, str) else self.root.expression

    def required_features(self) -> set[str]:
        return {self.root} if isinstance(self.root, str) else self.root.expr.dependencies()


class _UnaryOp(SchemaModel):
    type: t.Literal["unary"] = "unary"
    feature: Feature


class _Thresholded(_UnaryOp):
    threshold: Threshold


class UnaryLessThanEqual(_Thresholded):
    op: t.Literal["<="] = "<="


class UnaryLessThan(_Thresholded):
    op: t.Literal["<"] = "<"


class UnaryEqual(_Thresholded):
    op: t.Literal["=="] = "=="


class UnaryGreaterThan(_Thresholded):
    op: t.Literal[">"] = ">"


class UnaryGreaterThanEqual(_Thresholded):
    op: t.Literal[">="] = ">="


class UnaryNotEqual(_Thresholded):
    op: t.Literal["!="] = "!="


class _Bounds(SchemaModel):
    min: t.Optional[Threshold] = None
    max: t.Optional[Threshold] = None

    @model_validator(mode="after")
    def _bounds(self) -> _Bounds:
        if self.min is None and self.max is None:
            raise ValueError("At least one of min or max must be specified")
        return self


class UnaryBetween(_UnaryOp, _Bounds):
    """`min <= x <= max`, closed at both ends; either bound may be left out."""

    op: t.Literal["between"] = "between"


class UnaryIsIn(_UnaryOp):
    """`x` equals one of `values`; a single `ParamRef` means equality with that param."""

    op: t.Literal["isin"] = "isin"
    values: Values


class UnaryStringMatch(_UnaryOp):
    """`x` matches any of `patterns`."""

    op: t.Literal["string_match"] = "string_match"
    patterns: Patterns
    match_type: StringMatchType = StringMatchType.exact
    case_sensitive: bool = True
    trim_whitespace: bool = False
    null_handling: NullHandling = NullHandling.no_match


class UnaryIsTrue(_UnaryOp):
    op: t.Literal["is_true"] = "is_true"


class UnaryIsFalse(_UnaryOp):
    op: t.Literal["is_false"] = "is_false"


TUnaryOp = t.Annotated[
    t.Union[
        UnaryLessThanEqual, UnaryLessThan, UnaryEqual, UnaryGreaterThan, UnaryGreaterThanEqual,
        UnaryNotEqual, UnaryBetween, UnaryIsIn, UnaryStringMatch, UnaryIsTrue, UnaryIsFalse,
    ],
    Field(discriminator="op"),
]


class RangeCondition(_Bounds):
    """One band of a `CasesRanges`; which end is closed is the node's `end_logic`."""


class StringMatchCondition(SchemaModel):
    patterns: Patterns


class IsInCondition(SchemaModel):
    values: Values


def _condition_tag(value: t.Any) -> str:
    return get_field(value, "type") or "unary"


class CompositeCondition(SchemaModel):
    """AND/OR/NOT over nested conditions. As a tree node, branch 0 is taken when it holds, branch 1 otherwise."""

    type: t.Literal["composite"] = "composite"
    id: t.Optional[str] = None
    op: LogicOp
    conditions: t.List[TCondition]

    @model_validator(mode="after")
    def _logic(self) -> CompositeCondition:
        if self.op == LogicOp.NOT and len(self.conditions) != 1:
            raise ValueError("NOT operator must have exactly 1 condition")
        if not self.conditions:
            raise ValueError("a composite needs at least one condition")
        return self


TCondition = t.Annotated[
    t.Union[t.Annotated[TUnaryOp, Tag("unary")], t.Annotated[CompositeCondition, Tag("composite")]],
    Discriminator(_condition_tag),
]
CompositeCondition.model_rebuild()


def _static(bound: t.Any) -> bool:
    return bound is not None and not isinstance(bound, ParamRef)


def validate_range_conditions(conditions: t.Sequence[RangeCondition], strict: bool) -> None:
    """In strict mode, literal bands must be sorted and contiguous; param bounds are skipped."""
    if not conditions or not strict:
        return
    if all(not _static(rc.min) for rc in conditions):
        maxes = [(i, rc.max) for i, rc in enumerate(conditions) if _static(rc.max)]
        for (i, a), (j, b) in zip(maxes, maxes[1:]):
            if a >= b:
                raise ValueError(f"Ranges must be in sorted order. Range {i} has max={a}, but range {j} has max={b}.")
        return
    static = [
        (i, rc.min, rc.max) for i, rc in enumerate(conditions)
        if not isinstance(rc.min, ParamRef) and not isinstance(rc.max, ParamRef)
    ]
    for (i, lo, hi), (j, next_lo, _) in zip(static, static[1:]):
        if lo is not None and next_lo is not None and lo >= next_lo:
            raise ValueError(f"Ranges must be in sorted order. Range {i} has min={lo}, but range {j} has min={next_lo}.")
        if hi is not None and next_lo is not None and hi != next_lo:
            raise ValueError(
                f"Ranges are not continuous in strict mode. Range {i} ends at {hi} but range {j} starts at {next_lo}."
            )

