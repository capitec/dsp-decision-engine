import typing as t
import re
import enum
import polars as pl
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    from .tree import RootMetadata


class NodePosition(BaseModel):
    x: float = 0.0
    y: float = 0.0


class NodeMeta(BaseModel):
    """Execution-agnostic metadata attached to a node.
    Currently used to preserve UI layout positions through round-trip conversion.
    """
    position: t.Optional[NodePosition] = None


def _get_ui_position(meta: "NodeMeta") -> "t.Any":
    """Resolve meta.position to a v2 UI Position, defaulting to (0, 0)."""
    from ped.modules.tree.ui.v2.nodes import Position
    pos = meta.position
    return Position(x=pos.x, y=pos.y) if pos else Position(x=0.0, y=0.0)


# class ConditionTypeProtocol(t.Protocol):
#     def get_required_parameters(self) -> t.Set[str]: ...


_TConditionType = t.TypeVar("_TConditionType")

class BranchType(BaseModel, t.Generic[_TConditionType]):
    when: t.Union[_TConditionType, t.List[_TConditionType]]
    then: "NodeType"

    @property
    def whens(self) -> t.List[_TConditionType]:
        return self.when if isinstance(self.when, list) else [self.when]


@dataclass
class BuilderConfig:
    build_result_function: t.Callable
    output_literals: t.List[pl.Expr]
    default_literal: t.Optional[pl.Expr] = None
    root_meta: "t.Optional[RootMetadata]" = None


class IndexedBranch(t.NamedTuple):
    index: t.Optional[int]
    branch: "Branch"

DefaultBranch = IndexedBranch(index=None, branch=None)
TBranchStack = t.Tuple[IndexedBranch,...]


class LeafNodeType(BaseModel):
    """Base configuration for a node type."""
    IS_LEAF: t.ClassVar[bool] = True
    type: t.Literal["leaf"] = "leaf"
    result_idx: int = Field(
        default=-1,
        description="Index into the output literals table. -1 marks a default/no-match leaf."
    )
    meta: NodeMeta = Field(default_factory=NodeMeta, description="Optional execution-agnostic metadata (e.g. UI position).")

    def get_required_features(self) -> t.Set[str]:
        return set()

    def get_required_parameters(self) -> t.Set[str]:
        return set()

    def _as_ui_node(self, node_id: str):
        from ped.modules.tree.ui.v2.nodes import PositionedNode, LeafNode as V2Leaf
        return PositionedNode(id=node_id, position=_get_ui_position(self.meta), data=V2Leaf(leaf_value=self.result_idx))

    def to_ui_nodes(self):
        from uuid import uuid4
        nid = str(uuid4())
        return nid, [self._as_ui_node(nid)], []

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
    meta: NodeMeta = Field(default_factory=NodeMeta, description="Optional execution-agnostic metadata (e.g. UI position).")

    def get_nodes_required_parameters(self) -> t.Set[str]:
        return set().union(*(
            when.get_required_parameters() 
            for b in self.branches
            for when in b.whens
        ))

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
    def _as_ui_node(self, node_id: str):
        """Return the UI PositionedNode for this node only (no children)."""
        ...

    def to_ui_nodes(self):
        """Recursively convert this node and all children to v2 UI nodes and edges."""
        from uuid import uuid4
        from ped.modules.tree.ui.v1.edges import MultiSourceEdge, MultiEdgeData
        nid = str(uuid4())
        all_nodes = [self._as_ui_node(nid)]
        all_edges = []
        for i, branch in enumerate(self.branches):
            child_id, child_nodes, child_edges = branch.then.to_ui_nodes()
            all_nodes.extend(child_nodes)
            all_edges.extend(child_edges)
            all_edges.append(MultiSourceEdge(
                id=f"{nid}:{child_id}", source=nid, target=child_id,
                data=MultiEdgeData(sourceIndex=[i]),
            ))
        oth_id, oth_nodes, oth_edges = self.otherwise.to_ui_nodes()
        all_nodes.extend(oth_nodes)
        all_edges.extend(oth_edges)
        all_edges.append(MultiSourceEdge(
            id=f"{nid}:{oth_id}", source=nid, target=oth_id,
            data=MultiEdgeData(sourceIndex=[len(self.branches)]),
        ))
        return nid, all_nodes, all_edges

    @abstractmethod
    def build_conditions(
        self,
        feature_expr: pl.Expr,
        branch_idx: int,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr],
    ) -> t.List[pl.Expr]:
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
            condition_exprs = self.build_conditions(
                feature_expr=inputs[self.feature],
                branch_idx=idx,
                inputs=inputs,
                parameters=parameters,
            )
            then_expr = branch.then.build_expression(
                    inputs=inputs, 
                    branch_stack=branch_stack + (IndexedBranch(index=idx, branch=branch),), 
                    config=config,
                    parameters=parameters,
                )
            for condition_expr in condition_exprs:
                out_expr = out_expr\
                    .when(condition_expr)\
                    .then(then_expr)
        default_expr = self.otherwise.build_expression(
            inputs=inputs, 
            branch_stack=branch_stack + (DefaultBranch,), 
            config=config,
            parameters=parameters,
        )
        if out_expr is pl: return default_expr
        return out_expr.otherwise(default_expr)

class RangeEndLogic(str, enum.Enum):
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
            for condition in branch.whens:
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

    def _as_ui_node(self, node_id: str):
        from ped.modules.tree.ui.v2.nodes import PositionedNode, RangeNode as V2Range
        thresholds = [
            c.max for b in self.branches
            for c in b.whens
            if isinstance(c, MinMaxConditionType) and c.max is not None
        ]
        return PositionedNode(
            id=node_id, position=_get_ui_position(self.meta),
            data=V2Range(feature=self.feature, thresholds=thresholds,
                         default_left=self.end_logic == RangeEndLogic.upper_inclusive),
        )

    def build_conditions(
        self,
        feature_expr: pl.Expr,
        branch_idx: int,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr],
    ) -> t.List[pl.Expr]:
        branch = self.branches[branch_idx]
        conditions = []
        for when in branch.whens:
            if isinstance(when, EqualConditionType):
                conditions.append(feature_expr == when.equal_expr(parameters))
            elif isinstance(when, MinMaxConditionType):
                min_expr = when.min_expr(parameters)
                max_expr = when.max_expr(parameters)
                if self.end_logic == RangeEndLogic.upper_inclusive:
                    min_cond = feature_expr >= min_expr if min_expr is not None else None
                    max_cond = feature_expr < max_expr if max_expr is not None else None
                else:
                    min_cond = feature_expr > min_expr if min_expr is not None else None
                    max_cond = feature_expr <= max_expr if max_expr is not None else None
                if min_cond is not None and max_cond is not None:
                    conditions.append(min_cond & max_cond)
                else:
                    conditions.append(min_cond if max_cond is None else max_cond)
            else:
                raise ValueError(f"Unsupported condition type: {type(when)}")
        return conditions


class StringPatternMatchType(str, enum.Enum):
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

    def _as_ui_node(self, node_id: str):
        from ped.modules.tree.ui.v2.nodes import PositionedNode, StringMatchNode as V2Str
        all_conditions = [when for b in self.branches for when in b.whens]
        if len(all_conditions) == 1:
            pat = all_conditions[0].pattern
            patterns = list(pat) if isinstance(pat, (set, frozenset)) else [pat]
            match_any = isinstance(pat, (set, frozenset))
        else:
            patterns = [when.pattern for when in all_conditions]
            match_any = False
        first_cond = all_conditions[0]
        return PositionedNode(
            id=node_id, position=_get_ui_position(self.meta),
            data=V2Str(feature=self.feature, patterns=patterns,
                       match_type=first_cond.match_type, case_sensitive=first_cond.case_sensitive,
                       match_any=match_any),
        )

    def to_ui_nodes(self):
        """Single branch → defer to base class (match_any=True path in _as_ui_node).
        Multi-branch → explode any set-patterns to individual UI indices and
        deduplicate shared child subtrees via MultiSourceEdge.sourceIndex."""
        if len(self.branches) == 1:
            return super().to_ui_nodes()

        from uuid import uuid4
        from ped.modules.tree.ui.v2.nodes import PositionedNode, StringMatchNode as V2Str
        from ped.modules.tree.ui.v1.edges import MultiSourceEdge, MultiEdgeData

        nid = str(uuid4())

        # Explode each branch's pattern(s) into a flat list, tracking which UI
        # indices belong to each branch in one pass.
        # e.g. patterns [(a,b,c), (d,e), (f)] → flat [a,b,c,d,e,f]
        #   branch 0 → sourceIndex=[0,1,2], branch 1 → sourceIndex=[3,4], branch 2 → sourceIndex=[5]
        flat_patterns: t.List[str] = []
        branch_ui_indices: t.List[t.List[int]] = []
        ui_idx = 0
        for branch in self.branches:
            branch_pattern_indices: t.List[int] = []
            for when in branch.whens:
                pat = when.pattern
                members = sorted(pat) if isinstance(pat, (set, frozenset)) else [pat]
                for p in members:
                    flat_patterns.append(p)
                    branch_pattern_indices.append(ui_idx)
                    ui_idx += 1
            branch_ui_indices.append(branch_pattern_indices)

        first_cond = self.branches[0].whens[0]
        all_nodes = [PositionedNode(
            id=nid, position=_get_ui_position(self.meta),
            data=V2Str(
                feature=self.feature,
                patterns=flat_patterns,
                match_type=first_cond.match_type,
                case_sensitive=first_cond.case_sensitive,
                match_any=False,
            ),
        )]
        all_edges: t.List = []

        for b_idx, indices in enumerate(branch_ui_indices):
            child_id, child_nodes, child_edges = self.branches[b_idx].then.to_ui_nodes()
            all_nodes.extend(child_nodes)
            all_edges.extend(child_edges)
            all_edges.append(MultiSourceEdge(
                id=f"{nid}:{child_id}", source=nid, target=child_id,
                data=MultiEdgeData(sourceIndex=indices),
            ))

        oth_id, oth_nodes, oth_edges = self.otherwise.to_ui_nodes()
        all_nodes.extend(oth_nodes)
        all_edges.extend(oth_edges)
        all_edges.append(MultiSourceEdge(
            id=f"{nid}:{oth_id}", source=nid, target=oth_id,
            data=MultiEdgeData(sourceIndex=[len(flat_patterns)]),
        ))
        return nid, all_nodes, all_edges

    def build_conditions(
        self,
        feature_expr: pl.Expr,
        branch_idx: int,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr] = None,
    ) -> t.List[pl.Expr]:
        branch = self.branches[branch_idx]
        return [when.get_condition(feature_expr, inputs, parameters) for when in branch.whens]


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

    @model_validator(mode="after")
    def validate_conditions(self) -> "t.Self":
        assert len(self.branches) <= 1, "Numerical at most one branch and an otherwise"
        assert len(self.branches) == 0 or len(self.branches[0].whens) == 1, "Numerical conditions must have exactly one 'when'"
        return self

    def _as_ui_node(self, node_id: str):
        from ped.modules.tree.ui.v2.nodes import PositionedNode, NumericalNode as V2Num, VariableReference
        cond: NumericalConditionType = self.branches[0].whens[0]
        thr = VariableReference(name=cond.threshold.key) if isinstance(cond.threshold, InputRef) else cond.threshold
        return PositionedNode(
            id=node_id, position=_get_ui_position(self.meta),
            data=V2Num(feature=self.feature, comparison_op=cond.op, threshold=thr),
        )

    def build_conditions(
        self,
        feature_expr: pl.Expr,
        branch_idx: int,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr] = None,
    ) -> t.List[pl.Expr]:
        branch = self.branches[branch_idx]
        return [when.get_condition(feature_expr, parameters) for when in branch.whens]


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

    def _as_ui_node(self, node_id: str):
        from ped.modules.tree.ui.v2.nodes import PositionedNode, CategoricalNode as V2Cat, VariableReference
        cond: CategoricalConditionType = self.branches[0].whens[0]
        cats = [VariableReference(name=cond.categories.key)] if isinstance(cond.categories, InputRef) else list(cond.categories)
        return PositionedNode(
            id=node_id, position=_get_ui_position(self.meta),
            data=V2Cat(feature=self.feature, category_list=cats),
        )

    def build_conditions(
        self,
        feature_expr: pl.Expr,
        branch_idx: int,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr] = None,
    ) -> t.List[pl.Expr]:
        branch = self.branches[branch_idx]
        return [when.get_condition(feature_expr, parameters) for when in branch.whens]


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

BranchType.model_rebuild()
RangesNodeType.model_rebuild()
NumericalNodeType.model_rebuild()
CategoricalNodeType.model_rebuild()
StringNodeType.model_rebuild()