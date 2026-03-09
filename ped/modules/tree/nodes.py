import typing as t
import regex as re
import enum
import polars as pl
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from dataclasses import dataclass, field


class ConditionTypeProtocol(t.Protocol):
    def get_required_parameters(self) -> t.Set[str]: ...


_TConditionType = t.TypeVar("_TConditionType", bound=ConditionTypeProtocol)

class BranchType(BaseModel, t.Generic[_TConditionType]):
    when: _TConditionType
    then: "NodeType"


@dataclass
class BuilderConfig:
    build_result_function: t.Callable
    output_literals: t.List[pl.Expr]
    default_literal: t.Optional[pl.Expr] = None


class IndexedBranch(t.NamedTuple):
    index: t.Optional[int]
    branch: "Branch"

DefaultBranch = IndexedBranch(index=None, branch=None)
TBranchStack = t.Tuple[IndexedBranch,...]


def default_result_builder(
    inputs: t.Dict[str, pl.Expr],
    branch_stack: TBranchStack,
    config: BuilderConfig,
    result_idx: int,
) -> pl.Expr:
    """Default output function: returns the output literal at result_idx,
    or the default literal when result_idx is -1 (the no-match leaf)."""
    if result_idx == -1:
        return config.default_literal
    return config.output_literals[result_idx]

class LeafNodeType(BaseModel):
    """Base configuration for a node type."""
    IS_LEAF: t.ClassVar[bool] = True
    type: t.Literal["leaf"] = "leaf"
    result_idx: int = Field(
        default=-1,
        description="Index into the output literals table. -1 marks a default/no-match leaf."
    )

    def get_required_features(self) -> t.Set[str]:
        return set()

    def get_required_parameters(self) -> t.Set[str]:
        return set()

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        """Builds the Polars expression for this node by calling the builder function with the appropriate inputs and returning the result at the specified index."""
        return config.build_result_function(
            inputs=inputs,
            branch_stack=branch_stack,
            config=config,
            result_idx=self.result_idx,
        )


class BaseNodeType(BaseModel, t.Generic[_TConditionType], ABC):
    """Base configuration for a node type."""
    IS_LEAF: t.ClassVar[bool] = False
    type: str = Field(description="Type of the node")
    feature: str = Field(description="Feature to apply the node logic on")
    branches: t.List[BranchType[_TConditionType]] = Field(description="Branching logic for the node")
    otherwise: "NodeType" = Field(description="Default branch if no conditions are met")

    def get_nodes_required_parameters(self) -> t.Set[str]:
        return set().union(*(b.when.get_required_parameters() for b in self.branches))

    def get_required_parameters(self) -> t.Set[str]:
        return self.get_nodes_required_parameters().union(
            *(b.then.get_required_parameters() for b in self.branches),
            self.otherwise.get_required_parameters(),
        )

    def get_required_features(self) -> t.Set[str]:
        return {self.feature}.union(
            *(b.then.get_required_features() for b in self.branches),
            self.otherwise.get_required_features(),
        )

    @abstractmethod
    def build_condition(
        self,
        feature_expr: pl.Expr,
        branch_idx: int,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr],
    ) -> pl.Expr:
        """Builds the Polars expression for the condition of a given branch index"""
        ...

    def build_expression(
        self,
        inputs: t.Dict[str, pl.Expr],
        branch_stack: TBranchStack,
        config: BuilderConfig,
        parameters: t.Optional[pl.Expr] = None,
    ) -> pl.Expr:
        """Builds the Polars expression for this node based on the input expressions and the branching logic."""
        out_expr = pl
        for idx, branch in enumerate(self.branches):
            condition_expr = self.build_condition(
                feature_expr=inputs[self.feature],
                branch_idx=idx,
                inputs=inputs,
                parameters=parameters,
            )
            out_expr = out_expr.when(condition_expr).then(
                branch.then.build_expression(
                    inputs=inputs, 
                    branch_stack=branch_stack + (IndexedBranch(index=idx, branch=branch),), 
                    config=config,
                    parameters=parameters,
                )
            )
        default_expr = self.otherwise.build_expression(
            inputs=inputs, 
            branch_stack=branch_stack + (DefaultBranch,), 
            config=config,
            parameters=parameters,
        )
        if out_expr is pl: return default_expr
        return out_expr.otherwise(default_expr)

class RangeEndLogic(enum.StrEnum):
    lower_inclusive = "lower_inclusive"
    upper_inclusive = "upper_inclusive"

class InputRef(BaseModel):
    """Allow models to more dynamically get parameters from the payload or dataframe."""
    key: str = Field(description="Input key from the graph execution context")

    def resolve(self, parameters: t.Optional[pl.Expr]):
        return parameters.struct.field(self.key)

class MinMaxConditionType(BaseModel):
    """Condition type for range-based branching."""
    min: t.Optional[t.Union[InputRef,float]] = Field(default=None, description="Minimum value for the range")
    max: t.Optional[t.Union[InputRef,float]] = Field(default=None, description="Maximum value for the range")

    def get_required_parameters(self):
        out = set()
        if isinstance(self.min, InputRef):
            out.add(self.min.key)
        if isinstance(self.max, InputRef):
            out.add(self.max.key)
        return out
    def min_expr(self, parameters: pl.Expr) -> t.Optional[pl.Expr]:
        if isinstance(self.min, InputRef):
            return self.min.resolve(parameters)
        elif self.min is not None:
            return pl.lit(self.min)
        return None
    def max_expr(self, parameters: pl.Expr) -> t.Optional[pl.Expr]:
        if isinstance(self.max, InputRef):
            return self.max.resolve(parameters)
        elif self.max is not None:
            return pl.lit(self.max)
        return None

class EqualConditionType(BaseModel):
    """Condition type for equality-based branching."""
    equal: t.Union[InputRef, float]

    def get_required_parameters(self) -> t.Set[str]:
        return {self.equal.key} if isinstance(self.equal, InputRef) else set()

    def equal_expr(self, parameters: pl.Expr) -> pl.Expr:
        if isinstance(self.equal, InputRef):
            return self.equal.resolve(parameters)
        return pl.lit(self.equal)

RangeConditionType = t.Union[EqualConditionType, MinMaxConditionType]

class RangesNodeType(BaseNodeType[RangeConditionType]):
    """Node type for range-based branching."""
    type: t.Literal["ranges"] = "ranges"
    end_logic: RangeEndLogic = Field(
        default=RangeEndLogic.lower_inclusive,
        description="Logic for determining which range a boundary value belongs to. "
        "If 'lower_inclusive', a value equal to the boundary goes to the lower range. "
        "If 'upper_inclusive', it goes to the upper range."
    )
    strict: bool = Field(
        default=True,
        description="Whether to enforce strict range boundaries. If True, values that fall into gaps between ranges will raise an error. If false no error will be raised unless otherwise is not set."
    )
    # This is a precomputed field that is derived from branches
    # it ensures only the first min value or the last max value are None and that there are no gaps or overlaps
    _completed_branches: t.List[RangeConditionType] = PrivateAttr()

    @model_validator(mode="after")
    def validate_and_complete_branches(self) -> "t.Self":
        # TODO confirm this code didnt really give it much thought
        completed_branches = []
        last_max = None
        for branch in self.branches:
            condition = branch.when
            if isinstance(condition, EqualConditionType):
                completed_branches.append(condition)
                # last_max = condition.equal
            elif isinstance(condition, MinMaxConditionType):
                min_val = condition.min
                max_val = condition.max
                if min_val is None and last_max is not None:
                    min_val = last_max
                if max_val is None and min_val is not None:
                    max_val = min_val
                if self.strict and min_val is not None and last_max is not None:
                    if min_val != last_max:
                        raise ValueError("Gaps between ranges are not allowed in strict mode")
                    if not isinstance(max_val, InputRef) and not isinstance(last_max, InputRef) and max_val < last_max:
                        raise ValueError("Overlapping ranges are not allowed in strict mode")
                completed_branches.append(MinMaxConditionType(min=min_val, max=max_val))
                last_max = max_val
            else:
                raise ValueError(f"Unsupported condition type: {type(condition)}")
        self._completed_branches = completed_branches
        return self

    def build_condition(self, feature_expr, branch_idx, inputs, parameters):
        condition = self._completed_branches[branch_idx]
        if isinstance(condition, EqualConditionType):
            return feature_expr == condition.equal_expr(parameters)
        elif isinstance(condition, MinMaxConditionType):
            min_expr = condition.min_expr(parameters)
            max_expr = condition.max_expr(parameters)
            if self.end_logic == RangeEndLogic.upper_inclusive:
                min_cond = feature_expr >= min_expr if min_expr is not None else None
                max_cond = feature_expr < max_expr if max_expr is not None else None
            else:
                min_cond = feature_expr > min_expr if min_expr is not None else None
                max_cond = feature_expr <= max_expr if max_expr is not None else None
            if min_cond is not None and max_cond is not None:
                return min_cond & max_cond
            return min_cond or max_cond
        else:
            raise ValueError(f"Unsupported condition type: {type(condition)}")


class StringPatternMatchType(enum.StrEnum):
    exact = "exact"
    starts_with = "starts_with"
    contains = "contains"
    ends_with = "ends_with"
    regex = "regex"

class StringPatternConditionType(BaseModel):
    """Condition type for string pattern-based branching."""
    pattern: str|t.Set[str]|InputRef
    match_type: StringPatternMatchType = StringPatternMatchType.exact
    case_sensitive: bool = True

    def get_required_parameters(self):
        return set() if not isinstance(self.pattern, InputRef) else {self.pattern.key}

    def get_condition(self, feature: pl.Expr, inputs: t.Dict[str, pl.Expr], parameters: pl.Expr) -> pl.Expr:
        # --- Resolve pattern(s) into expressions + optional literals list
        if isinstance(self.pattern, InputRef):
            pat_exprs: t.List[pl.Expr] = [self.pattern.resolve(parameters)]
            pat_literals: t.Optional[t.List[str]] = None
        else:
            raw: t.List[str] = list(self.pattern) if isinstance(self.pattern, (set, frozenset)) else [self.pattern]
            pat_exprs = [pl.lit(p) for p in raw]
            pat_literals = raw

        # --- Regex: unify both paths — static patterns are pre-joined into a single literal expression
        if self.match_type == StringPatternMatchType.regex:
            pattern_expr = pl.lit("|".join(f"(?:{p})" for p in pat_literals)) if pat_literals is not None else pat_exprs[0]
            if not self.case_sensitive:
                pattern_expr = pl.concat_str(pl.lit("(?i)"), pattern_expr)
            return feature.str.contains(pattern_expr, literal=False)

        # --- Non-regex: normalise case on both sides
        feat = feature.str.to_lowercase() if not self.case_sensitive else feature
        pat_exprs = [p.str.to_lowercase() if not self.case_sensitive else p for p in pat_exprs]


        # --- Build condition expression
        if self.match_type == StringPatternMatchType.exact:
            # is_in is more efficient for multiple static literals; == covers single / InputRef
            if pat_literals is not None and len(pat_literals) > 1:
                if not self.case_sensitive: pat_literals = [p.lower() for p in pat_literals]
                return feat.is_in(pat_literals)
            return feat == pat_exprs[0]

        if self.match_type == StringPatternMatchType.contains:
            exprs = [feat.str.contains(p, literal=True) for p in pat_exprs]
        elif self.match_type == StringPatternMatchType.starts_with:
            exprs = [feat.str.starts_with(p) for p in pat_exprs]
        elif self.match_type == StringPatternMatchType.ends_with:
            exprs = [feat.str.ends_with(p) for p in pat_exprs]
        else:
            raise ValueError(f"Unsupported match type: {self.match_type}")

        result = exprs[0]
        for expr in exprs[1:]:
            result = result | expr
        return result


class StringNodeType(BaseNodeType[StringPatternConditionType]):
    """Node type for string equality-based branching."""
    type: t.Literal["string"] = "string"

    def build_condition(self, feature_expr, branch_idx, inputs, parameters=None):
        condition: StringPatternConditionType = self.branches[branch_idx].when
        return condition.get_condition(feature_expr, inputs, parameters)


# ---------------------------------------------------------------------------
# Numerical node  (single comparison: feature op threshold)
# ---------------------------------------------------------------------------

NumericalOp = t.Literal["<=", "<", "==", ">", ">="]

class NumericalConditionType(BaseModel):
    """Condition type for a single numerical comparison (feature op threshold)."""
    op: NumericalOp = "<="
    threshold: t.Union[InputRef, float]

    def get_required_parameters(self) -> t.Set[str]:
        return {self.threshold.key} if isinstance(self.threshold, InputRef) else set()

    def threshold_expr(self, parameters: t.Optional[pl.Expr]) -> pl.Expr:
        if isinstance(self.threshold, InputRef):
            return self.threshold.resolve(parameters)
        return pl.lit(self.threshold)

    def get_condition(self, feature: pl.Expr, parameters: t.Optional[pl.Expr]) -> pl.Expr:
        t_expr = self.threshold_expr(parameters)
        if self.op == "<=": return feature <= t_expr
        if self.op == "<":  return feature < t_expr
        if self.op == "==": return feature == t_expr
        if self.op == ">":  return feature > t_expr
        if self.op == ">=": return feature >= t_expr
        raise ValueError(f"Unsupported operator: {self.op}")


class NumericalNodeType(BaseNodeType[NumericalConditionType]):
    """Node type for numerical comparison branching (feature op threshold)."""
    type: t.Literal["numerical"] = "numerical"

    def build_condition(self, feature_expr, branch_idx, inputs, parameters=None):
        condition: NumericalConditionType = self.branches[branch_idx].when
        return condition.get_condition(feature_expr, parameters)


# ---------------------------------------------------------------------------
# Categorical node  (feature is_in a list of values)
# ---------------------------------------------------------------------------

_CatValue = t.Union[int, float, str]

class CategoricalConditionType(BaseModel):
    """Condition type for categorical membership branching (feature is_in categories)."""
    categories: t.Union[t.List[_CatValue], InputRef]
    case_sensitive: bool = True  # only applies when category values are strings

    def get_required_parameters(self) -> t.Set[str]:
        return {self.categories.key} if isinstance(self.categories, InputRef) else set()

    def get_condition(self, feature: pl.Expr, parameters: t.Optional[pl.Expr]) -> pl.Expr:
        if isinstance(self.categories, InputRef):
            # Single category value sourced from a runtime parameter
            cat_expr = self.categories.resolve(parameters)
            if not self.case_sensitive:
                return feature.str.to_lowercase() == cat_expr.str.to_lowercase()
            return feature == cat_expr

        cats = self.categories
        if not self.case_sensitive:
            cats = [c.lower() if isinstance(c, str) else c for c in cats]
            return feature.str.to_lowercase().is_in(cats)
        return feature.is_in(cats)


class CategoricalNodeType(BaseNodeType[CategoricalConditionType]):
    """Node type for categorical membership branching."""
    type: t.Literal["categorical"] = "categorical"

    def build_condition(self, feature_expr, branch_idx, inputs, parameters=None):
        condition: CategoricalConditionType = self.branches[branch_idx].when
        return condition.get_condition(feature_expr, parameters)


NodeType = t.Annotated[
    t.Union[
        LeafNodeType,
        RangesNodeType,
        NumericalNodeType,
        CategoricalNodeType,
        StringNodeType,
    ], Field(discriminator="type")
]

TBranch = t.Union[
    BranchType[RangeConditionType],
    BranchType[NumericalConditionType],
    BranchType[CategoricalConditionType],
    BranchType[StringPatternConditionType],
]

RangesNodeType.model_rebuild()
NumericalNodeType.model_rebuild()
CategoricalNodeType.model_rebuild()
StringNodeType.model_rebuild()