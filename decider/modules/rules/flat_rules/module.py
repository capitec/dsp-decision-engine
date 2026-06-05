"""Flat rules module — compiles rules into executable Polars expressions."""

import typing as t
import enum
import polars as pl
from pydantic import Field
from dataclasses import dataclass

from ....serializable.function import DefinedFunction
from ..common.shared import WithTreeOutput, InputRef
from ..common.parameters import WithParameters
from .nodes import BuilderConfig, RuleRoot, FlatRuleTree, RuleMeta
from .impl import (
    execute_rule_root,
    execute_prioritized_rules,
    execute_flat_rule_tree,
    build_parameters_expr,
    default_result_builder,
    extract_value,
)
from ....serializable.schema import PolarsSchema
from decider.modules.core import BaseExecuteModule

if t.TYPE_CHECKING:
    from decider.executor import Executor
    from decider.types import TInputType, TOutputType


class PrioritizationMode(str, enum.Enum):
    """How to handle multiple rules in a FlatRuleTree."""

    first_match = "first_match"  # Return first rule that matches (prioritized)
    all = "all"  # Return all rule results as a struct


@dataclass
class OptimRunPolarsExpression:
    """Optimized Polars expression executor (no feature extraction)."""

    expr: pl.Expr

    def get_output(
        self,
        input_frame: pl.DataFrame,
    ) -> pl.DataFrame:
        return input_frame.select(self.expr)


@dataclass
class RunPolarsExpression:
    """Standard Polars expression executor with feature extraction."""

    expr: pl.Expr
    features: t.List[str]
    parameters_expr: t.Optional[pl.Expr] = None

    def execute(self, input_frame: pl.DataFrame) -> pl.DataFrame:
        return input_frame.select(self.expr.struct.unnest())


class FlatRuleModule(WithTreeOutput, BaseExecuteModule, WithParameters):
    """Single rule compiled as a Polars expression."""

    type: t.Literal["flat_rule"]
    name: str = "output"
    rule: RuleRoot
    output_fn: t.Optional[DefinedFunction] = None
    use_optimized_execution: bool = False

    def get_required_parameters(self) -> t.Set[str]:
        """Get all parameters required by this rule."""
        return self.rule.rule.get_required_parameters()

    def get_required_features(self) -> t.Set[str]:
        """Get all features required by this rule."""
        return self.rule.rule.get_required_features()

    def build_expression(self) -> RunPolarsExpression:
        """Compile rule into a RunPolarsExpression that can be executed as Hamilton nodes."""
        output_fn = (
            self.output_fn.get_function() if self.output_fn else default_result_builder
        )
        config = BuilderConfig(
            build_result_function=output_fn,
            output_literals=self.output_literals,
            default_literal=self.default_literal,
        )

        # Build the main rule expression
        required_features = self.get_required_features()
        inputs = {col: pl.col(col) for col in required_features}

        # Build parameters expression if needed
        parameters_expr = None
        if self.parameters:
            param_schema = self.parameter_schema
            default_literals = {
                name: (
                    info._polars_literal
                    if info._polars_literal is not None
                    else pl.lit(None)
                )
                for name, info in self.parameters.items()
            }
            parameters_expr = build_parameters_expr(
                runtime_params=(
                    pl.col(self.parameters_col) if self.parameters_col else None
                ),
                parameter_schema=param_schema,
                default_literals=default_literals,
            )

        # Execute the rule to get the result expression
        result_expr = execute_rule_root(
            rule_root=self.rule,
            builder_config=config,
            parameters=parameters_expr,
            **inputs,
        )

        extra_features = []
        if self.parameters:
            extra_features.append(self.parameters_col)

        if self.use_optimized_execution:
            return OptimRunPolarsExpression(expr=result_expr)

        return RunPolarsExpression(
            expr=result_expr,
            features=list(required_features) + extra_features,
            parameters_expr=parameters_expr,
        )

    def execute(self, inputs: "TInputType", _executor: "Executor") -> "TOutputType":
        frame = inputs["input"]
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        if self.parameters and self.parameters_col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(self.parameters_col))
        compiled = self.build_expression()
        return frame.select(compiled.expr.struct.unnest()).lazy()


class PrioritizedFlatRuleModule(WithTreeOutput, BaseExecuteModule, WithParameters):
    """Multiple flat rules evaluated in priority order; first match wins."""

    type: t.Literal["prioritized_flat_rule"]
    name: str = "output"

    input_schema: t.Optional[PolarsSchema] = Field(
        default=None, description="Input schema for casting inputs at runtime"
    )
    rules: t.List[RuleRoot] = Field(
        description="List of rules to evaluate in priority order"
    )
    mode: PrioritizationMode = Field(
        default=PrioritizationMode.first_match,
        description="'first_match' returns the first rule that matches; 'all' returns all results.",
    )
    use_optimized_execution: bool = False
    output_fn: t.Optional[DefinedFunction] = None
    post_process_fn: t.Optional[DefinedFunction] = None
    format_prioritized_fn: t.Optional[DefinedFunction] = None

    def get_required_parameters(self) -> t.Set[str]:
        """Get all parameters required by any rule."""
        params = set()
        for rule in self.rules:
            params.update(rule.rule.get_required_parameters())
        return params

    def get_required_features(self) -> t.Set[str]:
        """Get all features required by any rule."""
        features = set()
        for rule in self.rules:
            features.update(rule.rule.get_required_features())
        return features

    def build_expression(self) -> RunPolarsExpression:
        """Compile prioritized flat rules into a RunPolarsExpression."""
        output_fn = (
            self.output_fn.get_function() if self.output_fn else default_result_builder
        )
        post_process_fn = (
            self.post_process_fn.get_function()
            if self.post_process_fn
            else extract_value
        )
        format_prioritized_fn = (
            self.format_prioritized_fn.get_function()
            if self.format_prioritized_fn
            else None
        )

        config = BuilderConfig(
            build_result_function=output_fn,
            output_literals=self.output_literals,
            default_literal=self.default_literal,
        )

        # Collect all required features and parameters from all rules
        required_features = self.get_required_features()
        inputs = {col: pl.col(col) for col in required_features}

        # Build parameters expression if needed
        parameters_expr = None
        if self.parameters:
            param_schema = self.parameter_schema
            default_literals = {
                name: (
                    info._polars_literal
                    if info._polars_literal is not None
                    else pl.lit(None)
                )
                for name, info in self.parameters.items()
            }
            parameters_expr = build_parameters_expr(
                runtime_params=(
                    pl.col(self.parameters_col) if self.parameters_col else None
                ),
                parameter_schema=param_schema,
                default_literals=default_literals,
            )

        # Execute prioritized rules
        if self.mode == PrioritizationMode.first_match:
            result_expr = execute_prioritized_rules(
                rules=self.rules,
                builder_config=config,
                parameters=parameters_expr,
                post_process_fn=post_process_fn,
                format_prioritized_fn=format_prioritized_fn,
                **inputs,
            )
        elif self.mode == PrioritizationMode.all:
            # Execute all rules and return as struct
            results = execute_flat_rule_tree(
                tree=FlatRuleTree(rules=self.rules),
                builder_config=config,
                parameters=parameters_expr,
                **inputs,
            )
            result_expr = pl.struct(
                *[
                    e.alias(self.rules[i].meta.name or f"rule_{i}")
                    for i, e in enumerate(results)
                ]
            )
        else:
            raise ValueError(f"Unsupported prioritization mode: {self.mode}")

        extra_features = []
        if self.parameters:
            extra_features.append(self.parameters_col)

        if self.use_optimized_execution:
            return OptimRunPolarsExpression(expr=result_expr)

        return RunPolarsExpression(
            expr=result_expr,
            features=list(required_features) + extra_features,
            parameters_expr=parameters_expr,
        )

    def execute(self, inputs: "TInputType", _executor: "Executor") -> "TOutputType":
        frame = inputs["input"]
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        if self.parameters and self.parameters_col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).alias(self.parameters_col))
        compiled = self.build_expression()
        return frame.select(compiled.expr.struct.unnest()).lazy()
