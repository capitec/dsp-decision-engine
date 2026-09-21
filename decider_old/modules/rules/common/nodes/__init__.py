"""Common node definitions shared between flat_rules and tree v3."""

from .operators import (
    _BaseUnaryOp,
    _ThresholdedUnaryOp,
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
    TUnaryOp,
)

from .conditions import (
    RangeCondition,
    StringMatchCondition,
    IsInCondition,
    CasesBranch,
    TCaseCondition,
    CompositeCondition,
    TCondition,
    _UnaryOpConditionWrapper,
)

from .unary import BaseUnaryNode
from .cases import (
    _CasesRangesCore,
    _CasesStringMatchCore,
    _CasesIsInCore,
    BaseCasesRanges,
    BaseCasesStringMatch,
    BaseCasesIsIn,
    validate_range_conditions,
)
from .composite import BaseCompositeNode

__all__ = [
    # Operators
    "_BaseUnaryOp",
    "_ThresholdedUnaryOp",
    "UnaryLessThanEqual",
    "UnaryLessThan",
    "UnaryEqual",
    "UnaryGreaterThan",
    "UnaryGreaterThanEqual",
    "UnaryNotEqual",
    "UnaryBetween",
    "UnaryIsIn",
    "UnaryStringMatch",
    "UnaryIsNull",
    "UnaryIsNotNull",
    "UnaryIsTrue",
    "UnaryIsFalse",
    "TUnaryOp",
    # Conditions
    "RangeCondition",
    "StringMatchCondition",
    "IsInCondition",
    "CasesBranch",
    "TCaseCondition",
    "CompositeCondition",
    "TCondition",
    "_UnaryOpConditionWrapper",
    # Base nodes
    "BaseUnaryNode",
    "_CasesRangesCore",
    "_CasesStringMatchCore",
    "_CasesIsInCore",
    "BaseCasesRanges",
    "BaseCasesStringMatch",
    "BaseCasesIsIn",
    "validate_range_conditions",
    "BaseCompositeNode",
]
