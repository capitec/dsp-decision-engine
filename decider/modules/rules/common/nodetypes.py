"""Common node types and operators shared between flat rules and tree v3."""

import typing as t
import enum
from pydantic import BaseModel, Field

# =============================================================================
# Base Metadata Types
# =============================================================================


class NodePosition(BaseModel):
    """UI position for a node."""

    x: float = 0.0
    y: float = 0.0


class NodeMeta(BaseModel):
    """Execution-agnostic metadata attached to a node.

    Currently used to preserve UI layout positions through round-trip conversion.
    """

    position: t.Optional[NodePosition] = None


class BaseRule(BaseModel):
    """Base class for all rule/node types with optional metadata."""

    meta: t.Optional[NodeMeta] = Field(
        default=None,
        description="Optional metadata (e.g., UI position)",
    )


# =============================================================================
# Enums - Shared operators
# =============================================================================


class RangeEndLogic(str, enum.Enum):
    """Logic for range boundary handling."""

    lower_inclusive = "lower_inclusive"
    upper_inclusive = "upper_inclusive"


class TPrimitiveOperators(str, enum.Enum):
    """Primitive comparison operators for unary conditions."""

    LEQ = "<="
    LT = "<"
    EQ = "=="
    GT = ">"
    GEQ = ">="
    NEQ = "!="


class TStringMatchType(str, enum.Enum):
    """String matching strategies."""

    exact = "exact"
    starts_with = "starts_with"
    contains = "contains"
    ends_with = "ends_with"
    regex = "regex"


class TLogicOp(str, enum.Enum):
    """Logical operators for composite conditions."""

    AND = "and"
    OR = "or"
    NOT = "not"


class TNodeType(str, enum.Enum):
    """Node type discriminators."""

    LEAF = "leaf"
    UNARY = "unary"
    CASES = "cases"
    COMPOSITE = "composite"


# =============================================================================
# Core node types - Base models shared between systems
# =============================================================================


class LeafNodeCore(BaseRule):
    """Core leaf node - terminal node that returns a result."""

    type: t.Literal["leaf"] = "leaf"
    result_idx: int = Field(
        default=-1,
        description="Index into output table. -1 indicates default/no-match.",
    )


# =============================================================================
# Core condition types - Used in both flat rules and tree v3
# =============================================================================


class MinMaxConditionCore(BaseModel):
    """Core range condition with min/max bounds."""

    min: t.Optional[t.Union[float, int]] = Field(
        default=None, description="Minimum value (null for unbounded)"
    )
    max: t.Optional[t.Union[float, int]] = Field(
        default=None, description="Maximum value (null for unbounded)"
    )
