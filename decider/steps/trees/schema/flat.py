"""Flat-rule documents: each node nests its own branches."""
from __future__ import annotations

import enum
import typing as t

from pydantic import BaseModel, Discriminator, Field, PrivateAttr, Tag, model_validator

from decider.steps.trees.schema.conditions import IsInCondition, RangeCondition, StringMatchCondition
from decider.steps.trees.schema.nodes import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    CompositeNode,
    LeafNode,
    UnaryNode,
    _IsIn,
    _Ranges,
    _StringMatch,
    node_tag,
)
from decider.steps.trees.schema.tree import Node, ParameterInfo, Rule, Tree, TreeOutput, add_node

LeafRule = LeafNode


class _ThenOtherwise(BaseModel):
    then: t.Optional[RuleType] = None
    otherwise: t.Optional[RuleType] = None

    def branches_by_index(self) -> list[tuple[int, t.Optional[RuleType]]]:
        return [(0, self.then), (1, self.otherwise)]


class UnaryRule(UnaryNode, _ThenOtherwise):
    """A `UnaryNode` with its `then` and `otherwise` rules nested; a missing one selects the default row."""

    node_type: t.ClassVar[type] = UnaryNode


class CompositeRule(CompositeNode, _ThenOtherwise):
    """A `CompositeNode` with its `then` and `otherwise` rules nested."""

    node_type: t.ClassVar[type] = CompositeNode


class CasesBranch(BaseModel):
    """Case `when` leads to `branches[then]`."""

    when: t.Union[RangeCondition, StringMatchCondition, IsInCondition]
    then: int


class _Cases(BaseModel):
    conditions: t.List[CasesBranch]
    otherwise: int
    branches: t.List[RuleType]

    @model_validator(mode="after")
    def _indices(self) -> _Cases:
        for i in [c.then for c in self.conditions] + [self.otherwise]:
            if not 0 <= i < len(self.branches):
                raise ValueError(f"branch index {i} is out of range for {len(self.branches)} branch(es)")
        return self

    def branches_by_index(self) -> list[tuple[int, t.Optional[RuleType]]]:
        order = [c.then for c in self.conditions] + [self.otherwise]
        return [(i, self.branches[i]) for i in order]


class CasesRangesRule(_Ranges, _Cases):
    """A `CasesRanges` whose cases point into a nested `branches` list, with `otherwise` indexing it too."""

    node_type: t.ClassVar[type] = CasesRanges


class CasesStringMatchRule(_StringMatch, _Cases):
    node_type: t.ClassVar[type] = CasesStringMatch


class CasesIsInRule(_IsIn, _Cases):
    node_type: t.ClassVar[type] = CasesIsIn


RuleType = t.Annotated[
    t.Union[
        t.Annotated[LeafRule, Tag("leaf")],
        t.Annotated[UnaryRule, Tag("unary")],
        t.Annotated[CasesRangesRule, Tag("cases:ranges")],
        t.Annotated[CasesStringMatchRule, Tag("cases:string_match")],
        t.Annotated[CasesIsInRule, Tag("cases:isin")],
        t.Annotated[CompositeRule, Tag("composite")],
    ],
    Discriminator(node_tag),
]


def _payload(rule: t.Any) -> t.Any:
    cls = rule.node_type
    fields = {k: getattr(rule, k) for k in cls.model_fields if k not in ("id", "conditions")}
    if "conditions" in cls.model_fields:
        fields["conditions"] = [getattr(c, "when", c) for c in rule.conditions]
    return cls(**fields)


def _flatten(rule: t.Any, nid: str, nodes: dict[str, Node], defaults: dict[str, t.Any]) -> str:
    nid = rule.id or nid
    if isinstance(rule, LeafNode):
        add_node(nodes, nid, rule, (), defaults)
        return nid
    # ponytail: recursion depth follows rule nesting; pydantic already recursed this deep to validate it.
    children = [
        None if child is None else _flatten(child, f"{nid}.{i}", nodes, defaults)
        for i, child in rule.branches_by_index()
    ]
    add_node(nodes, nid, _payload(rule), children, defaults)
    return nid


class RuleMeta(BaseModel):
    name: t.Optional[str] = None
    description: t.Optional[str] = None


class RuleRoot(BaseModel):
    meta: RuleMeta = Field(default_factory=RuleMeta)
    rule: RuleType


class PrioritizationMode(str, enum.Enum):
    first_match = "first_match"
    all = "all"


_CODE_FIELDS = ("output_fn", "post_process_fn", "format_prioritized_fn")


class _FlatDocument(BaseModel):
    name: str = "output"
    output: TreeOutput
    parameters: t.Dict[str, ParameterInfo] = Field(default_factory=dict)

    _tree: Tree = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def _no_code(cls, data: t.Any) -> t.Any:
        if isinstance(data, dict) and (code := [k for k in _CODE_FIELDS if data.get(k)]):
            raise ValueError(f"{code} name Python functions, which tree documents no longer accept")
        return data

    @model_validator(mode="after")
    def _normalise(self) -> _FlatDocument:
        defaults = {k: p.default_value for k, p in self.parameters.items() if p.default_value is not None}
        nodes: dict[str, Node] = {}
        rules = tuple(
            Rule(root=_flatten(root.rule, str(i), nodes, defaults), name=root.meta.name)
            for i, root in enumerate(self._roots())
        )
        self._tree = Tree(name=self.name, mode=self._mode(), rules=rules, nodes=nodes, output=self.output)
        return self

    def to_tree(self) -> Tree:
        """This document as the format-independent `Tree`."""
        return self._tree


class FlatRuleDocument(_FlatDocument):
    """One nested rule.

    Nodes without an `id` get one from their path: the root is `"0"`, its
    `then` branch `"0.0"`, a cases node's `branches[2]` `"<id>.2"`.

    Example (the `V3TreeDocument` example's logic, with path ids)::

        FlatRuleDocument(
            rule={"rule": {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 30},
                           "then": {"type": "leaf", "result_idx": 0}}},
            output={"data": [{"group": "young"}], "default": {"group": "adult"}, "dtypes": [("group", "String")]},
        ).to_tree()
    """

    type: t.Literal["flat_rule"] = "flat_rule"
    rule: RuleRoot

    def _roots(self) -> list[RuleRoot]:
        return [self.rule]

    def _mode(self) -> str:
        return "first_match"


class PrioritizedFlatRuleDocument(_FlatDocument):
    """Several nested rules in priority order; see `Tree` for `mode`. Root ids default to `"0"`, `"1"`, ..."""

    type: t.Literal["prioritized_flat_rule"] = "prioritized_flat_rule"
    rules: t.List[RuleRoot]
    mode: PrioritizationMode = PrioritizationMode.first_match

    def _roots(self) -> list[RuleRoot]:
        return self.rules

    def _mode(self) -> str:
        return self.mode.value


for _model in (_ThenOtherwise, UnaryRule, CompositeRule, _Cases, CasesRangesRule, CasesStringMatchRule,
               CasesIsInRule, RuleRoot, FlatRuleDocument, PrioritizedFlatRuleDocument):
    _model.model_rebuild()
