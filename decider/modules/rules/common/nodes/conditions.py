"""Condition types shared between flat_rules and tree v3.

RangeCondition, StringMatchCondition, IsInCondition, CasesBranch —
all shared directly. CompositeCondition and TCondition are defined
here so both systems use the exact same composite logic.
"""

import typing as t
from pydantic import BaseModel, Field, model_validator, Discriminator, Tag
import typing_extensions as t_ext
import polars as pl

from dspd.components.common.shared import InputRef
from dspd.components.common.nodetypes import TStringMatchType, TLogicOp, RangeEndLogic
from dspd.components.common.nodes.operators import TUnaryOp

# =============================================================================
# Range
# =============================================================================


class RangeCondition(BaseModel):
    """Range condition with optional min/max bounds."""

    min: t.Optional[t.Union[float, int, InputRef]] = None
    max: t.Optional[t.Union[float, int, InputRef]] = None

    @model_validator(mode="after")
    def validate_bounds(self) -> t_ext.Self:
        if self.min is None and self.max is None:
            raise ValueError("At least one of min or max must be specified")
        return self

    def _resolve_bound(
        self,
        bound: t.Optional[t.Union[float, int, InputRef]],
        parameters: t.Optional[pl.Expr],
    ) -> t.Optional[pl.Expr]:
        if bound is None:
            return None
        if isinstance(bound, InputRef):
            return bound.resolve(parameters)
        return pl.lit(bound)

    def build_range_condition(
        self,
        feature_expr: pl.Expr,
        end_logic: RangeEndLogic,
        parameters: t.Optional[pl.Expr],
    ) -> pl.Expr:
        min_expr = self._resolve_bound(self.min, parameters)
        max_expr = self._resolve_bound(self.max, parameters)

        if end_logic == RangeEndLogic.lower_inclusive:
            min_cond = feature_expr >= min_expr if min_expr is not None else None
            max_cond = feature_expr < max_expr if max_expr is not None else None
        else:
            min_cond = feature_expr > min_expr if min_expr is not None else None
            max_cond = feature_expr <= max_expr if max_expr is not None else None

        if min_cond is not None and max_cond is not None:
            return min_cond & max_cond
        elif min_cond is not None:
            return min_cond
        else:
            return max_cond


# =============================================================================
# String match
# =============================================================================


class StringMatchCondition(BaseModel):
    """String match condition — patterns are a mix of static strings and InputRefs."""

    patterns: t.List[t.Union[str, InputRef]] = Field(
        description="Patterns to match (OR logic) - can mix static strings and InputRefs"
    )

    @model_validator(mode="after")
    def validate_patterns(self) -> t_ext.Self:
        if not self.patterns:
            raise ValueError("patterns list must contain at least one pattern")
        return self

    def _handle_static_patterns(
        self,
        feature_expr: pl.Expr,
        static_patterns: t.List[str],
        match_type: str,
        case_sensitive: bool,
    ) -> t.Optional[pl.Expr]:
        if not static_patterns:
            return None

        if match_type == "exact":
            return feature_expr.is_in(static_patterns)
        elif match_type == "regex":
            combined_pattern = "|".join(f"(?:{p})" for p in static_patterns)
            return feature_expr.str.contains(combined_pattern, literal=False)
        else:
            if match_type == "contains":
                exprs = [
                    feature_expr.str.contains(p, literal=True) for p in static_patterns
                ]
            elif match_type == "starts_with":
                exprs = [feature_expr.str.starts_with(p) for p in static_patterns]
            else:  # ends_with
                exprs = [feature_expr.str.ends_with(p) for p in static_patterns]

            result = exprs[0]
            for expr in exprs[1:]:
                result = result | expr
            return result

    def _handle_dynamic_patterns(
        self,
        feature_expr: pl.Expr,
        dynamic_refs: t.List[InputRef],
        match_type: str,
        case_sensitive: bool,
        parameters: t.Optional[pl.Expr],
    ) -> t.List[pl.Expr]:
        conditions = []
        for ref in dynamic_refs:
            pat_expr = ref.resolve(parameters)
            if not case_sensitive:
                pat_expr = pat_expr.str.to_lowercase()

            if match_type == "exact":
                conditions.append(feature_expr == pat_expr)
            elif match_type == "contains":
                conditions.append(feature_expr.str.contains(pat_expr, literal=True))
            elif match_type == "starts_with":
                conditions.append(feature_expr.str.starts_with(pat_expr))
            elif match_type == "ends_with":
                conditions.append(feature_expr.str.ends_with(pat_expr))
            elif match_type == "regex":
                conditions.append(feature_expr.str.contains(pat_expr, literal=False))
        return conditions

    def build_match_condition(
        self,
        feature_expr: pl.Expr,
        match_type: str,
        case_sensitive: bool,
        parameters: t.Optional[pl.Expr],
    ) -> pl.Expr:
        feat = feature_expr.str.to_lowercase() if not case_sensitive else feature_expr

        static_patterns = [p for p in self.patterns if isinstance(p, str)]
        dynamic_refs = [p for p in self.patterns if isinstance(p, InputRef)]

        if not case_sensitive:
            static_patterns = [p.lower() for p in static_patterns]

        conditions = []
        static_cond = self._handle_static_patterns(
            feat, static_patterns, match_type, case_sensitive
        )
        if static_cond is not None:
            conditions.append(static_cond)

        conditions.extend(
            self._handle_dynamic_patterns(
                feat, dynamic_refs, match_type, case_sensitive, parameters
            )
        )

        if not conditions:
            return pl.lit(False)

        result = conditions[0]
        for cond in conditions[1:]:
            result = result | cond
        return result


# =============================================================================
# IsIn
# =============================================================================


class IsInCondition(BaseModel):
    values: t.Union[t.List[t.Union[int, float]], InputRef]


# =============================================================================
# Cases branch wrapper
# =============================================================================

TCaseCondition = t.Union[RangeCondition, StringMatchCondition, IsInCondition]


class CasesBranch(BaseModel):
    """A single case: when condition → then branch_index."""

    when: TCaseCondition
    then: int = Field(description="Index into branches array")


# =============================================================================
# Composite condition (shared, recursive)
# =============================================================================


def _condition_discriminator(value: t.Any) -> str:
    """Return tag based on the 'type' field of the incoming value."""
    if isinstance(value, dict):
        return value.get("type", "unary")
    return getattr(value, "type", "unary")


TCondition = t.Annotated[
    t.Union[
        t.Annotated[TUnaryOp, Tag("unary")],
        t.Annotated["CompositeCondition", Tag("composite")],
    ],
    Discriminator(_condition_discriminator),
]

# Backward-compat alias — nothing actually uses this as a class now
_UnaryOpConditionWrapper = None


class CompositeCondition(BaseModel):
    """Nested composite condition (AND/OR/NOT of other conditions)."""

    type: t.Literal["composite"] = "composite"
    id: t.Optional[str] = Field(default=None)
    op: TLogicOp
    conditions: t.List[TCondition] = Field(description="List of conditions to combine")

    @model_validator(mode="after")
    def validate_and_ensure_id(self) -> t_ext.Self:
        if self.op == TLogicOp.NOT and len(self.conditions) != 1:
            raise ValueError("NOT operator must have exactly 1 condition")
        if self.id is None:
            import uuid

            self.id = str(uuid.uuid4())
        return self

    def get_required_features(self) -> t.Set[str]:
        features = set()
        for cond in self.conditions:
            features.update(cond.get_required_features())
        return features

    def get_required_parameters(self) -> t.Set[str]:
        params = set()
        for cond in self.conditions:
            params.update(cond.get_required_parameters())
        return params

    def build_condition(
        self,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr],
    ) -> pl.Expr:
        if not self.conditions:
            return pl.lit(False)

        cond_exprs = [
            cond.build_condition(inputs, parameters) for cond in self.conditions
        ]

        if self.op == TLogicOp.NOT:
            return ~cond_exprs[0]

        result = cond_exprs[0]
        for expr in cond_exprs[1:]:
            result = result & expr if self.op == TLogicOp.AND else result | expr
        return result


# Rebuild for forward ref
CompositeCondition.model_rebuild()
