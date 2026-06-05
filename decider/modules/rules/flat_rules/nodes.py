"""Flat rules execution nodes.

All structural definitions (operators, conditions, base nodes) live in
dspd.components.common.nodes. This module adds only execution logic:
  - build_expression() for all rule nodes
  - WithUnaryBranches: then/otherwise embedded children
  - WithCasesBranches: branches list + otherwise index

Re-exports common types so existing imports keep working.
"""

import typing as t
import uuid
from pydantic import BaseModel, Field, model_validator
import typing_extensions as t_ext
import polars as pl
from dataclasses import dataclass

from ..common.shared import InputRef
from ..common.feature import Feature as _Feature
from ..common.nodetypes import (
    BaseRule,
    TNodeType,
    TLogicOp,
    TStringMatchType,
    RangeEndLogic as CommonRangeEndLogic,
)

# Import everything from common.nodes
from ..common.nodes import (
    TUnaryOp,
    RangeCondition,
    StringMatchCondition,
    IsInCondition,
    CasesBranch,
    TCaseCondition,
    CompositeCondition,
    TCondition,
    _UnaryOpConditionWrapper,
    BaseUnaryNode,
    _CasesRangesCore,
    _CasesStringMatchCore,
    _CasesIsInCore,
    BaseCompositeNode,
    validate_range_conditions,
    UnaryLessThanEqual,
    UnaryLessThan,
    UnaryEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryNotEqual,
    UnaryBetween,
    UnaryIsIn,
    UnaryStringMatch,
    UnaryIsNull,
    UnaryIsNotNull,
    UnaryIsTrue,
    UnaryIsFalse,
)

# Re-export common enums for backward compatibility
LogicOp = TLogicOp
StringMatchType = TStringMatchType
RangeEndLogic = CommonRangeEndLogic

if t.TYPE_CHECKING:
    from .tree import RuleMeta


# =============================================================================
# Branch Stack and Builder Config
# =============================================================================


class IndexedBranch(t.NamedTuple):
    """Track which branch was taken in the decision tree.

    For unary rules:
        index=0 means 'then' branch (condition true)
        index=None means 'otherwise' branch (condition false)

    For cases rules:
        index=0,1,2... means conditions[0], conditions[1], conditions[2]...
        index=None means 'otherwise' branch (no conditions matched)
    """

    index: t.Optional[int]
    rule: "RuleType"


TBranchStack = t.Tuple[IndexedBranch, ...]


@dataclass
class BuilderConfig:
    """Configuration for building rule expressions."""

    build_result_function: t.Callable
    output_literals: t.List[pl.Expr]
    default_literal: t.Optional[pl.Expr] = None
    rule_idx: int = 0
    root_meta: "t.Optional[RuleMeta]" = None

    @property
    def default_expr(self):
        return (
            self.default_literal if self.default_literal is not None else pl.lit(None)
        )


# =============================================================================
# Leaf Rule
# =============================================================================


class LeafRule(BaseRule):
    """Terminal node that returns a result."""

    type: t.Literal[TNodeType.LEAF] = TNodeType.LEAF
    id: t.Optional[str] = Field(default=None)
    result_idx: int = Field(default=-1)

    @model_validator(mode="after")
    def ensure_id(self) -> t_ext.Self:
        if self.id is None:
            self.id = str(uuid.uuid4())
        return self

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: "BuilderConfig",
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        return config.build_result_function(
            inputs=inputs,
            branch_stack=branch_stack,
            config=config,
            result_idx=self.result_idx,
        )

    def get_required_features(self) -> t.Set[str]:
        return set()

    def get_required_parameters(self) -> t.Set[str]:
        return set()


# =============================================================================
# Mixins for embedded children
# =============================================================================


class WithUnaryBranches(BaseModel):
    """Mixin: adds embedded then/otherwise children (flat_rules style)."""

    then: t.Optional["RuleType"] = Field(default=None)
    otherwise: t.Optional["RuleType"] = Field(default=None)

    def _get_then_rule(self) -> "RuleType":
        return self.then if self.then is not None else LeafRule(result_idx=-1)

    def _get_otherwise_rule(self) -> "RuleType":
        return self.otherwise if self.otherwise is not None else LeafRule(result_idx=-1)

    def get_branch_required_features(self) -> t.Set[str]:
        features = set()
        if self.then:
            features.update(self.then.get_required_features())
        if self.otherwise:
            features.update(self.otherwise.get_required_features())
        return features

    def get_branch_required_parameters(self) -> t.Set[str]:
        params = set()
        if self.then:
            params.update(self.then.get_required_parameters())
        if self.otherwise:
            params.update(self.otherwise.get_required_parameters())
        return params


class WithCasesBranches(BaseModel):
    """Mixin: adds branches list + otherwise index (flat_rules style)."""

    otherwise: int = Field(description="Default branch index if no conditions match")
    branches: t.List["RuleType"] = Field(description="Array of branch rules")

    def get_branch_required_features(self) -> t.Set[str]:
        features = set()
        for branch in self.branches:
            features.update(branch.get_required_features())
        return features

    def get_branch_required_parameters(self) -> t.Set[str]:
        params = set()
        for branch in self.branches:
            params.update(branch.get_required_parameters())
        return params


# =============================================================================
# Unary Rule
# =============================================================================


class UnaryRule(BaseUnaryNode, WithUnaryBranches):
    """Single condition with embedded then/otherwise branches."""

    @model_validator(mode="after")
    def ensure_id(self) -> t_ext.Self:
        if self.id is None:
            self.id = str(uuid.uuid4())
        return self

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        condition_expr = self.condition.build_condition(inputs, parameters)
        then_expr = self._get_then_rule().build_expression(
            inputs,
            branch_stack + (IndexedBranch(index=0, rule=self),),
            config,
            parameters,
        )
        otherwise_expr = self._get_otherwise_rule().build_expression(
            inputs,
            branch_stack + (IndexedBranch(index=None, rule=self),),
            config,
            parameters,
        )
        return pl.when(condition_expr).then(then_expr).otherwise(otherwise_expr)

    def get_required_features(self) -> t.Set[str]:
        return super().get_required_features() | self.get_branch_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return super().get_required_parameters() | self.get_branch_required_parameters()


# =============================================================================
# Cases Rules
# =============================================================================


class CasesRanges(_CasesRangesCore, WithCasesBranches):
    """Multi-way range branching with embedded branches."""

    conditions: t.List[CasesBranch] = Field(
        description="List of range conditions mapped to branch indices"
    )

    @model_validator(mode="after")
    def ensure_id_and_validate(self) -> t_ext.Self:
        if self.id is None:
            self.id = str(uuid.uuid4())
        validate_range_conditions(
            [cb.when for cb in self.conditions if isinstance(cb.when, RangeCondition)],
            self.strict,
        )
        return self

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        feature_expr = self.feature.build_expression(inputs, parameters)
        out_expr = pl

        for idx, case_branch in enumerate(self.conditions):
            assert isinstance(case_branch.when, RangeCondition)
            condition_expr = case_branch.when.build_range_condition(
                feature_expr=feature_expr,
                end_logic=self.end_logic,
                parameters=parameters,
            )
            branch_expr = self.branches[case_branch.then].build_expression(
                inputs,
                branch_stack + (IndexedBranch(index=idx, rule=self),),
                config,
                parameters,
            )
            out_expr = out_expr.when(condition_expr).then(branch_expr)

        otherwise_expr = self.branches[self.otherwise].build_expression(
            inputs,
            branch_stack + (IndexedBranch(index=None, rule=self),),
            config,
            parameters,
        )

        if out_expr is pl:
            return pl.when(feature_expr.is_not_null()).then(otherwise_expr).otherwise(otherwise_expr)
        return out_expr.otherwise(otherwise_expr)

    def get_required_features(self) -> t.Set[str]:
        return super().get_required_features() | self.get_branch_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        params = self.feature.get_required_parameters()
        for cb in self.conditions:
            if isinstance(cb.when, RangeCondition):
                if isinstance(cb.when.min, InputRef):
                    params.add(cb.when.min.key)
                if isinstance(cb.when.max, InputRef):
                    params.add(cb.when.max.key)
        params.update(super().get_required_parameters() | self.get_branch_required_parameters())
        return params


class CasesStringMatch(_CasesStringMatchCore, WithCasesBranches):
    """Multi-way string matching with embedded branches."""

    conditions: t.List[CasesBranch] = Field(
        description="List of pattern conditions mapped to branch indices"
    )

    @model_validator(mode="after")
    def ensure_id(self) -> t_ext.Self:
        if self.id is None:
            self.id = str(uuid.uuid4())
        return self

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        feature_expr = self.feature.build_expression(inputs, parameters)
        if self.trim_whitespace:
            feature_expr = feature_expr.str.strip_chars()
        out_expr = pl

        for idx, case_branch in enumerate(self.conditions):
            assert isinstance(case_branch.when, StringMatchCondition)
            condition_expr = case_branch.when.build_match_condition(
                feature_expr=feature_expr,
                match_type=(
                    self.match_type.value
                    if isinstance(self.match_type, TStringMatchType)
                    else self.match_type
                ),
                case_sensitive=self.case_sensitive,
                parameters=parameters,
            )
            branch_expr = self.branches[case_branch.then].build_expression(
                inputs,
                branch_stack + (IndexedBranch(index=idx, rule=self),),
                config,
                parameters,
            )
            out_expr = out_expr.when(condition_expr).then(branch_expr)

        otherwise_expr = self.branches[self.otherwise].build_expression(
            inputs,
            branch_stack + (IndexedBranch(index=None, rule=self),),
            config,
            parameters,
        )

        if out_expr is pl:
            return pl.when(feature_expr.is_not_null()).then(otherwise_expr).otherwise(otherwise_expr)
        return out_expr.otherwise(otherwise_expr)

    def get_required_features(self) -> t.Set[str]:
        return super().get_required_features() | self.get_branch_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        params = self.feature.get_required_parameters()
        for cb in self.conditions:
            if isinstance(cb.when, StringMatchCondition):
                for pattern in cb.when.patterns:
                    if isinstance(pattern, InputRef):
                        params.add(pattern.key)
        params.update(super().get_required_parameters() | self.get_branch_required_parameters())
        return params


class CasesIsIn(_CasesIsInCore, WithCasesBranches):
    """Multi-way categorical branching with embedded branches."""

    conditions: t.List[CasesBranch] = Field(
        description="List of value sets mapped to branch indices"
    )

    @model_validator(mode="after")
    def ensure_id(self) -> t_ext.Self:
        if self.id is None:
            self.id = str(uuid.uuid4())
        return self

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        feature_expr = self.feature.build_expression(inputs, parameters)
        out_expr = pl

        for idx, case_branch in enumerate(self.conditions):
            assert isinstance(case_branch.when, IsInCondition)
            if isinstance(case_branch.when.values, InputRef):
                condition_expr = feature_expr == case_branch.when.values.resolve(
                    parameters
                )
            else:
                condition_expr = feature_expr.is_in(case_branch.when.values)

            branch_expr = self.branches[case_branch.then].build_expression(
                inputs,
                branch_stack + (IndexedBranch(index=idx, rule=self),),
                config,
                parameters,
            )
            out_expr = out_expr.when(condition_expr).then(branch_expr)

        otherwise_expr = self.branches[self.otherwise].build_expression(
            inputs,
            branch_stack + (IndexedBranch(index=None, rule=self),),
            config,
            parameters,
        )

        if out_expr is pl:
            return pl.when(feature_expr.is_not_null()).then(otherwise_expr).otherwise(otherwise_expr)
        return out_expr.otherwise(otherwise_expr)

    def get_required_features(self) -> t.Set[str]:
        return super().get_required_features() | self.get_branch_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        params = self.feature.get_required_parameters()
        for cb in self.conditions:
            if isinstance(cb.when, IsInCondition) and isinstance(cb.when.values, InputRef):
                params.add(cb.when.values.key)
        params.update(super().get_required_parameters() | self.get_branch_required_parameters())
        return params


# Discriminated union for cases rules
TCasesVariant = t.Annotated[
    t.Union[CasesRanges, CasesStringMatch, CasesIsIn],
    Field(discriminator="op"),
]


from pydantic import RootModel


class CasesRule(RootModel[TCasesVariant]):
    """Wrapper for all Cases rule variants (discriminated by 'op' field)."""

    root: TCasesVariant

    @property
    def type(self) -> TNodeType:
        return TNodeType.CASES

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        return self.root.build_expression(inputs, branch_stack, config, parameters)

    def get_required_features(self) -> t.Set[str]:
        return self.root.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return self.root.get_required_parameters()


# =============================================================================
# Composite Rule
# =============================================================================


class CompositeRule(BaseCompositeNode, WithUnaryBranches):
    """Composite AND/OR/NOT rule with embedded then/otherwise branches."""

    @model_validator(mode="after")
    def ensure_id(self) -> t_ext.Self:
        if self.id is None:
            self.id = str(uuid.uuid4())
        return self

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        if not self.conditions:
            # pl.lit(False) is scalar and won't broadcast per-row; use a per-row false
            composite_condition = pl.int_range(pl.len()) < 0
        else:
            # Delegate to CompositeCondition's shared build_condition logic
            _tmp = CompositeCondition(op=self.op, conditions=self.conditions)
            composite_condition = _tmp.build_condition(inputs, parameters)

        then_expr = self._get_then_rule().build_expression(
            inputs,
            branch_stack + (IndexedBranch(index=0, rule=self),),
            config,
            parameters,
        )
        otherwise_expr = self._get_otherwise_rule().build_expression(
            inputs,
            branch_stack + (IndexedBranch(index=None, rule=self),),
            config,
            parameters,
        )
        return pl.when(composite_condition).then(then_expr).otherwise(otherwise_expr)

    def get_required_features(self) -> t.Set[str]:
        return super().get_required_features() | self.get_branch_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return super().get_required_parameters() | self.get_branch_required_parameters()


# =============================================================================
# Root Structure
# =============================================================================


class RuleMeta(BaseModel):
    name: t.Optional[str] = None
    description: t.Optional[str] = None


class RuleRoot(BaseModel):
    meta: RuleMeta = Field(default_factory=RuleMeta)
    rule: "RuleType"


class FlatRuleTree(BaseModel):
    rules: t.List[RuleRoot] = Field(description="List of independent rule trees")


# =============================================================================
# Top-level Rule Union
# =============================================================================

RuleType = t.Annotated[
    t.Union[
        LeafRule,
        "UnaryRule",
        CasesRule,
        "CompositeRule",
    ],
    Field(discriminator="type"),
]


# =============================================================================
# Rebuild models with forward references
# =============================================================================

UnaryRule.model_rebuild()
WithUnaryBranches.model_rebuild()
WithCasesBranches.model_rebuild()
CasesRanges.model_rebuild()
CasesStringMatch.model_rebuild()
CasesIsIn.model_rebuild()
CompositeRule.model_rebuild()
RuleRoot.model_rebuild()
