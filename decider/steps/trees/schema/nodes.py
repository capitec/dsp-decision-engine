"""The node kinds of a tree: what each node tests, without its children."""

import typing as t

from pydantic import Discriminator, Tag, model_validator

from decider.steps.trees.schema.conditions import (
    CompositeCondition,
    Feature,
    IsInCondition,
    RangeCondition,
    RangeEndLogic,
    SchemaModel,
    StringMatchCondition,
    StringMatchType,
    TUnaryOp,
    get_field,
    validate_range_conditions,
)


class LeafNode(SchemaModel):
    """Selects row `result_idx` of the tree's output table; -1 selects the default row."""

    type: t.Literal["leaf"] = "leaf"
    id: t.Optional[str] = None
    result_idx: int = -1


class UnaryNode(SchemaModel):
    """One condition. Branch 0 is taken when it holds, branch 1 otherwise."""

    type: t.Literal["unary"] = "unary"
    id: t.Optional[str] = None
    condition: TUnaryOp


CompositeNode = CompositeCondition


class _Ranges(SchemaModel):
    type: t.Literal["cases"] = "cases"
    op: t.Literal["ranges"] = "ranges"
    id: t.Optional[str] = None
    feature: Feature
    end_logic: RangeEndLogic = RangeEndLogic.lower_inclusive
    strict: bool = True


class _StringMatch(SchemaModel):
    type: t.Literal["cases"] = "cases"
    op: t.Literal["string_match"] = "string_match"
    id: t.Optional[str] = None
    feature: Feature
    match_type: StringMatchType = StringMatchType.exact
    case_sensitive: bool = True
    trim_whitespace: bool = False


class _IsIn(SchemaModel):
    type: t.Literal["cases"] = "cases"
    op: t.Literal["isin"] = "isin"
    id: t.Optional[str] = None
    feature: Feature


class CasesRanges(_Ranges):
    """Branch `i` is taken for the first band `conditions[i]` that holds; branch `len(conditions)` otherwise.

    With `strict` (the default), literal bands must be sorted and contiguous:
    `CasesRanges(feature="score", conditions=[{"max": 30.0}, {"min": 30.0}])`.
    """

    conditions: t.List[RangeCondition]

    @model_validator(mode="after")
    def _ranges(self) -> "CasesRanges":
        validate_range_conditions(self.conditions, self.strict)
        return self


class CasesStringMatch(_StringMatch):
    """Branch `i` is taken for the first `conditions[i]` whose patterns match; branch `len(conditions)` otherwise."""

    conditions: t.List[StringMatchCondition]


class CasesIsIn(_IsIn):
    """Branch `i` is taken for the first value set `conditions[i]` holding `x`; branch `len(conditions)` otherwise."""

    conditions: t.List[IsInCondition]


def node_tag(value: t.Any) -> str:
    kind = get_field(value, "type")
    return f"cases:{get_field(value, 'op', 'ranges')}" if kind == "cases" else str(kind)


NodeData = t.Annotated[
    t.Union[
        t.Annotated[LeafNode, Tag("leaf")],
        t.Annotated[UnaryNode, Tag("unary")],
        t.Annotated[CasesRanges, Tag("cases:ranges")],
        t.Annotated[CasesStringMatch, Tag("cases:string_match")],
        t.Annotated[CasesIsIn, Tag("cases:isin")],
        t.Annotated[CompositeNode, Tag("composite")],
    ],
    Discriminator(node_tag),
]
"""Any node kind, dispatched on `type` (and `op` for `cases`)."""


def arity(node: t.Any) -> int:
    """How many branches `node` has: 0 for a leaf, 2 for unary/composite, one per case plus otherwise."""
    if isinstance(node, LeafNode):
        return 0
    if isinstance(node, (UnaryNode, CompositeNode)):
        return 2
    return len(node.conditions) + 1
