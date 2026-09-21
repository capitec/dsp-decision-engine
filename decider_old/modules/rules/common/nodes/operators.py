"""Unary operator definitions shared between flat_rules and tree v3.

All operators have:
  - feature: the Feature to test
  - op: literal discriminator
  - build_condition(): polars expression (used by flat_rules)
  - get_required_features/parameters() (used by both systems)

flat_rules wraps these directly.
nodes_ui imports TUnaryOp and uses them inside UnaryNode/CompositeNode.
"""

import typing as t
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator
import typing_extensions as t_ext
import polars as pl

from ..shared import InputRef
from ..feature import Feature as _Feature
from ..nodetypes import TStringMatchType, TNullHandling


class _BaseUnaryOp(BaseModel, ABC):
    """Base class for all unary operators."""

    type: t.Literal["unary"] = "unary"
    feature: _Feature = Field(description="Feature name to test")

    @abstractmethod
    def build_condition(
        self,
        inputs: t.Dict[str, pl.Expr],
        parameters: t.Optional[pl.Expr],
    ) -> pl.Expr: ...

    @abstractmethod
    def _get_required_params(self) -> t.Set[str]: ...

    def get_required_features(self) -> t.Set[str]:
        return self.feature.get_required_features()

    def get_required_parameters(self) -> t.Set[str]:
        params = self.feature.get_required_parameters()
        params.update(self._get_required_params())
        return params


class _ThresholdedUnaryOp(_BaseUnaryOp, ABC):
    """Base for numeric comparison operators."""

    threshold: t.Union[float, int, InputRef] = Field(
        description="Comparison value (number or InputRef for runtime variable)"
    )

    def _resolve_threshold(self, parameters: t.Optional[pl.Expr]) -> pl.Expr:
        if isinstance(self.threshold, InputRef):
            return self.threshold.resolve(parameters)
        return pl.lit(self.threshold)

    def _get_required_params(self) -> t.Set[str]:
        if isinstance(self.threshold, InputRef):
            return {self.threshold.key}
        return set()


class UnaryLessThanEqual(_ThresholdedUnaryOp):
    op: t.Literal["<="] = "<="

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(
            inputs, parameters
        ) <= self._resolve_threshold(parameters)


class UnaryLessThan(_ThresholdedUnaryOp):
    op: t.Literal["<"] = "<"

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(
            inputs, parameters
        ) < self._resolve_threshold(parameters)


class UnaryEqual(_ThresholdedUnaryOp):
    op: t.Literal["=="] = "=="

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(
            inputs, parameters
        ) == self._resolve_threshold(parameters)


class UnaryGreaterThan(_ThresholdedUnaryOp):
    op: t.Literal[">"] = ">"

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(
            inputs, parameters
        ) > self._resolve_threshold(parameters)


class UnaryGreaterThanEqual(_ThresholdedUnaryOp):
    op: t.Literal[">="] = ">="

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(
            inputs, parameters
        ) >= self._resolve_threshold(parameters)


class UnaryNotEqual(_ThresholdedUnaryOp):
    op: t.Literal["!="] = "!="

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(
            inputs, parameters
        ) != self._resolve_threshold(parameters)


class UnaryBetween(_BaseUnaryOp):
    op: t.Literal["between"] = "between"
    min: t.Optional[t.Union[float, int, InputRef]] = Field(default=None)
    max: t.Optional[t.Union[float, int, InputRef]] = Field(default=None)

    @model_validator(mode="after")
    def validate_bounds(self) -> t_ext.Self:
        if self.min is None and self.max is None:
            raise ValueError("At least one of min or max must be specified")
        return self

    def _get_required_params(self) -> t.Set[str]:
        params = set()
        if isinstance(self.min, InputRef):
            params.add(self.min.key)
        if isinstance(self.max, InputRef):
            params.add(self.max.key)
        return params

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

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        feature_expr = self.feature.build_expression(inputs, parameters)
        min_expr = self._resolve_bound(self.min, parameters)
        max_expr = self._resolve_bound(self.max, parameters)

        if min_expr is not None and max_expr is not None:
            return (feature_expr >= min_expr) & (feature_expr <= max_expr)
        elif min_expr is not None:
            return feature_expr >= min_expr
        else:
            return feature_expr <= max_expr


class UnaryIsIn(_BaseUnaryOp):
    op: t.Literal["isin"] = "isin"
    values: t.Union[t.List[t.Union[int, float]], InputRef] = Field(
        description="List of acceptable values or InputRef for runtime variable"
    )

    @model_validator(mode="after")
    def validate_values(self) -> t_ext.Self:
        if isinstance(self.values, list) and not self.values:
            raise ValueError("values list must contain at least one element")
        return self

    def _get_required_params(self) -> t.Set[str]:
        if isinstance(self.values, InputRef):
            return {self.values.key}
        return set()

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        feature_expr = self.feature.build_expression(inputs, parameters)
        if isinstance(self.values, InputRef):
            return feature_expr == self.values.resolve(parameters)
        return feature_expr.is_in(self.values)


class UnaryStringMatch(_BaseUnaryOp):
    op: t.Literal["string_match"] = "string_match"
    patterns: t.List[t.Union[str, InputRef]] = Field(
        description="Patterns to match (OR logic) - can mix static strings and InputRefs"
    )
    match_type: TStringMatchType = Field(default=TStringMatchType.exact)
    case_sensitive: bool = Field(default=True)
    trim_whitespace: bool = Field(default=False)
    null_handling: TNullHandling = Field(
        default=TNullHandling.no_match,
        description=(
            "How to handle null feature values. "
            "'no_match' (default) — nulls never match any pattern. "
            "'match' — nulls match every pattern (wildcard). "
            "'error' — raise at runtime if a null is encountered."
        ),
    )

    @model_validator(mode="after")
    def validate_patterns(self) -> t_ext.Self:
        if not self.patterns:
            raise ValueError("patterns list must contain at least one pattern")
        return self

    def _get_required_params(self) -> t.Set[str]:
        params = set()
        for pattern in self.patterns:
            if isinstance(pattern, InputRef):
                params.add(pattern.key)
        return params

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        from .conditions import StringMatchCondition

        feature_expr = self.feature.build_expression(inputs, parameters)
        if self.trim_whitespace:
            feature_expr = feature_expr.str.strip_chars()
        matcher = StringMatchCondition(patterns=self.patterns)
        match_cond = matcher.build_match_condition(
            feature_expr=feature_expr,
            match_type=(
                self.match_type.value
                if isinstance(self.match_type, TStringMatchType)
                else self.match_type
            ),
            case_sensitive=self.case_sensitive,
            parameters=parameters,
            null_handling=self.null_handling,
        )
        return match_cond


class UnaryIsNull(_BaseUnaryOp):
    op: t.Literal["is_null"] = "is_null"

    def _get_required_params(self) -> t.Set[str]:
        return set()

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(inputs, parameters).is_null()


class UnaryIsNotNull(_BaseUnaryOp):
    op: t.Literal["is_not_null"] = "is_not_null"

    def _get_required_params(self) -> t.Set[str]:
        return set()

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(inputs, parameters).is_not_null()


class UnaryIsTrue(_BaseUnaryOp):
    op: t.Literal["is_true"] = "is_true"

    def _get_required_params(self) -> t.Set[str]:
        return set()

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(inputs, parameters) == pl.lit(True)


class UnaryIsFalse(_BaseUnaryOp):
    op: t.Literal["is_false"] = "is_false"

    def _get_required_params(self) -> t.Set[str]:
        return set()

    def build_condition(
        self, inputs: t.Dict[str, pl.Expr], parameters: t.Optional[pl.Expr]
    ) -> pl.Expr:
        return self.feature.build_expression(inputs, parameters) == pl.lit(False)


TUnaryOp = t.Annotated[
    t.Union[
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
    ],
    Field(discriminator="op"),
]
