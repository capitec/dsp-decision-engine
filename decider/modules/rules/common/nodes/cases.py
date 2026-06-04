"""Base cases nodes — feature + condition lists + required feature/param traversal.

The strict range validation lives here once as a standalone function callable
by both flat_rules (conditions: List[CasesBranch]) and nodes_ui (conditions: List[RangeCondition]).

Structure:
  _CasesRangesCore      — feature/op/end_logic/strict, get_required_features (no conditions)
  _CasesStringMatchCore — feature/op/match settings, get_required_features (no conditions)
  _CasesIsInCore        — feature/op, get_required_features (no conditions)
  validate_range_conditions() — standalone, accepts List[RangeCondition]
  BaseCasesRanges       — extends _CasesRangesCore with conditions: List[RangeCondition] (for nodes_ui)
  BaseCasesStringMatch  — extends _CasesStringMatchCore with conditions: List[StringMatchCondition]
  BaseCasesIsIn         — extends _CasesIsInCore with conditions: List[IsInCondition]

flat_rules extends the *Core classes and defines its own CasesBranch-typed conditions.
nodes_ui extends the Base* classes directly.
"""

import typing as t
from pydantic import BaseModel, Field, model_validator
import typing_extensions as t_ext

from dspd.components.common.nodetypes import BaseRule, TStringMatchType, RangeEndLogic
from dspd.components.common.feature import Feature as _Feature
from dspd.components.common.shared import InputRef
from dspd.components.common.nodes.conditions import (
    RangeCondition,
    StringMatchCondition,
    IsInCondition,
)

# =============================================================================
# Shared validation
# =============================================================================


def validate_range_conditions(conditions: t.List[RangeCondition], strict: bool) -> None:
    """Validate a list of RangeConditions for sorted/continuous order.

    Raises ValueError if strict=True and conditions are not valid.
    """
    if not conditions or not strict:
        return

    all_min_none = all(
        rc.min is None or isinstance(rc.min, InputRef) for rc in conditions
    )

    if all_min_none:
        static_max_values = [
            (i, rc.max)
            for i, rc in enumerate(conditions)
            if not isinstance(rc.max, InputRef) and rc.max is not None
        ]
        for j in range(len(static_max_values) - 1):
            if static_max_values[j][1] >= static_max_values[j + 1][1]:
                raise ValueError(
                    f"Ranges must be in sorted order. "
                    f"Range {static_max_values[j][0]} has max={static_max_values[j][1]}, "
                    f"but range {static_max_values[j+1][0]} has max={static_max_values[j+1][1]}."
                )
        return

    static_ranges = [
        (i, rc.min, rc.max)
        for i, rc in enumerate(conditions)
        if not isinstance(rc.min, InputRef) and not isinstance(rc.max, InputRef)
    ]

    if len(static_ranges) < 2:
        return

    for j in range(len(static_ranges) - 1):
        current_idx, current_min, current_max = static_ranges[j]
        next_idx, next_min, next_max = static_ranges[j + 1]

        current_min_val = current_min if current_min is not None else float("-inf")
        next_min_val = next_min if next_min is not None else float("-inf")

        if (
            current_min_val >= next_min_val
            and current_min is not None
            and next_min is not None
        ):
            raise ValueError(
                f"Ranges must be in sorted order. "
                f"Range {current_idx} has min={current_min}, "
                f"but range {next_idx} has min={next_min}."
            )

        if current_max is not None and next_min is not None and current_max != next_min:
            raise ValueError(
                f"Ranges are not continuous in strict mode. "
                f"Range {current_idx} ends at {current_max} but range {next_idx} starts at {next_min}."
            )


# =============================================================================
# Core base classes (no conditions field — subclasses define conditions)
# =============================================================================


class _CasesRangesCore(BaseRule):
    """Core fields for range-based cases nodes, shared between both systems."""

    type: t.Literal["cases"] = "cases"
    id: t.Optional[str] = None
    feature: _Feature
    op: t.Literal["ranges"] = "ranges"
    end_logic: RangeEndLogic = Field(default=RangeEndLogic.lower_inclusive)
    strict: bool = Field(default=True)

    def get_required_features(self) -> t.Set[str]:
        return self.feature.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return self.feature.get_required_parameters()


class _CasesStringMatchCore(BaseRule):
    """Core fields for string match cases nodes, shared between both systems."""

    type: t.Literal["cases"] = "cases"
    id: t.Optional[str] = None
    feature: _Feature
    op: t.Literal["string_match"] = "string_match"
    match_type: TStringMatchType = Field(default=TStringMatchType.exact)
    case_sensitive: bool = True
    trim_whitespace: bool = False

    def get_required_features(self) -> t.Set[str]:
        return self.feature.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return self.feature.get_required_parameters()


class _CasesIsInCore(BaseRule):
    """Core fields for isin cases nodes, shared between both systems."""

    type: t.Literal["cases"] = "cases"
    id: t.Optional[str] = None
    feature: _Feature
    op: t.Literal["isin"] = "isin"

    def get_required_features(self) -> t.Set[str]:
        return self.feature.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        return self.feature.get_required_parameters()


# =============================================================================
# nodes_ui base classes (conditions as plain condition objects, no branch indices)
# =============================================================================


class BaseCasesRanges(_CasesRangesCore):
    """Multi-way range branching base — used by nodes_ui where edges define routing."""

    conditions: t.List[RangeCondition] = Field(
        description="List of range conditions (order matches source index)"
    )

    @model_validator(mode="after")
    def _validate(self) -> t_ext.Self:
        validate_range_conditions(self.conditions, self.strict)
        return self

    def get_required_parameters(self) -> t.Set[str]:
        params = self.feature.get_required_parameters()
        for cond in self.conditions:
            if isinstance(cond.min, InputRef):
                params.add(cond.min.key)
            if isinstance(cond.max, InputRef):
                params.add(cond.max.key)
        return params


class BaseCasesStringMatch(_CasesStringMatchCore):
    """Multi-way string matching base — used by nodes_ui."""

    conditions: t.List[StringMatchCondition] = Field(
        description="List of pattern conditions (order matches source index)"
    )

    def get_required_parameters(self) -> t.Set[str]:
        params = self.feature.get_required_parameters()
        for cond in self.conditions:
            for pattern in cond.patterns:
                if isinstance(pattern, InputRef):
                    params.add(pattern.key)
        return params


class BaseCasesIsIn(_CasesIsInCore):
    """Multi-way categorical branching base — used by nodes_ui."""

    conditions: t.List[IsInCondition] = Field(
        description="List of value sets (order matches source index)"
    )

    def get_required_parameters(self) -> t.Set[str]:
        params = self.feature.get_required_parameters()
        for cond in self.conditions:
            if isinstance(cond.values, InputRef):
                params.add(cond.values.key)
        return params
