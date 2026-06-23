"""Execution logic for flat rules system.

This module provides functions to execute flat rules on Polars expressions and
DataFrames, with support for parameters, prioritized multi-rule execution, and
custom output formatting.
"""

import typing as t
import functools
import polars as pl
from .nodes import RuleRoot, FlatRuleTree, TBranchStack, RuleType, BuilderConfig
from dataclasses import dataclass, field

# =============================================================================
# Result Builder Functions
# =============================================================================


def default_result_builder(
    inputs: t.Dict[str, pl.Expr],
    branch_stack: TBranchStack,
    config: BuilderConfig,
    result_idx: int,
) -> pl.Expr:
    """Default output function: returns the output literal at result_idx,
    or the default literal when result_idx is -1 (the no-match leaf)."""
    if result_idx == -1:
        return config.default_expr
    return config.output_literals[result_idx]


# Single source of truth for OutputFn signature
OutputFn = t.Type[default_result_builder]


# =============================================================================
# Parameter Building
# =============================================================================


def build_parameters_expr(
    runtime_params: t.Optional[pl.Expr],
    parameter_schema: pl.Schema,
    default_literals: t.Dict[str, pl.Expr],
) -> t.Optional[pl.Expr]:
    """Build a parameters struct expression with smart fallback.

    For each parameter in the schema:
    - If runtime param exists and is not None, use it
    - Otherwise, fallback to default_literals[key]

    Runtime params are cast to the complete schema before coalescing
    to handle missing fields in the runtime struct.

    Args:
        runtime_params: Optional expression for runtime parameters column
        parameter_schema: Polars schema defining all parameter fields and types
        default_literals: Dict mapping parameter names to default literal expressions

    Returns:
        Struct expression with merged runtime and default parameters, or None if no params
    """
    if not parameter_schema:
        return None

    if runtime_params is None:
        # No runtime column, build struct from defaults only
        if default_literals:
            return pl.struct(
                *[expr.alias(name) for name, expr in default_literals.items()]
            )
        return None

    # Cast runtime params to complete schema (adds missing fields as None)
    runtime_casted = runtime_params.cast(pl.Struct(parameter_schema))

    # Build merged struct with per-field fallback
    merged_fields = []
    for name in parameter_schema:
        default_expr = default_literals.get(name)
        if default_expr is not None:
            # Runtime[name] if not None, else default
            merged_fields.append(
                pl.coalesce([runtime_casted.struct.field(name), default_expr]).alias(
                    name
                )
            )
        else:
            # No default, use runtime value (may be None)
            merged_fields.append(runtime_casted.struct.field(name).alias(name))

    return pl.struct(*merged_fields)


# =============================================================================
# Single Rule Execution
# =============================================================================


def execute_rule(
    rule: RuleType,
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr] = None,
    **inputs: pl.Expr,
) -> pl.Expr:
    """Execute a single rule and return the resulting Polars expression.

    Args:
        rule: The rule to execute (can be any RuleType)
        builder_config: Configuration with result builder and output literals
        parameters: Optional parameters struct expression
        **inputs: Feature columns as pl.Expr (e.g., age=pl.col('age'))

    Returns:
        Polars expression representing the rule's output
    """
    return rule.build_expression(
        inputs=inputs,
        branch_stack=(),
        config=builder_config,
        parameters=parameters,
    )


def execute_rule_root(
    rule_root: RuleRoot,
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr] = None,
    **inputs: pl.Expr,
) -> pl.Expr:
    """Execute a RuleRoot (rule with metadata) and return the resulting expression.

    Args:
        rule_root: RuleRoot containing meta and rule
        builder_config: Configuration with result builder and output literals
        parameters: Optional parameters struct expression
        **inputs: Feature columns as pl.Expr

    Returns:
        Polars expression representing the rule's output
    """
    from dataclasses import replace

    # Inject rule metadata into config
    config_with_meta = replace(builder_config, root_meta=rule_root.meta)

    return execute_rule(
        rule=rule_root.rule,
        builder_config=config_with_meta,
        parameters=parameters,
        **inputs,
    )


# =============================================================================
# Multiple Rule Execution
# =============================================================================


def execute_rule_list(
    rules: t.List[RuleRoot],
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr] = None,
    **inputs: pl.Expr,
) -> t.List[pl.Expr]:
    """Execute multiple rules and return one output expression per rule.

    Args:
        rules: List of RuleRoots to execute
        builder_config: Configuration with result builder and output literals
        parameters: Optional parameters struct expression
        **inputs: Feature columns as pl.Expr

    Returns:
        List of Polars expressions, one per rule
    """
    from dataclasses import replace

    return [
        execute_rule_root(
            rule_root=rule_root,
            builder_config=replace(builder_config, rule_idx=i),
            parameters=parameters,
            **inputs,
        )
        for i, rule_root in enumerate(rules)
    ]


def execute_flat_rule_tree(
    tree: FlatRuleTree,
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr] = None,
    **inputs: pl.Expr,
) -> t.List[pl.Expr]:
    """Execute a FlatRuleTree and return one expression per rule.

    Args:
        tree: FlatRuleTree containing multiple rules
        builder_config: Configuration with result builder and output literals
        parameters: Optional parameters struct expression
        **inputs: Feature columns as pl.Expr

    Returns:
        List of Polars expressions, one per rule in the tree
    """
    return execute_rule_list(
        rules=tree.rules,
        builder_config=builder_config,
        parameters=parameters,
        **inputs,
    )


# =============================================================================
# Prioritized Multi-Rule Execution
# =============================================================================


def _with_rule_idx(expr: pl.Expr, rule_idx: int) -> pl.Expr:
    """Enrich an indexed result struct {idx, val} with the originating rule index.

    Args:
        expr: Expression returning {idx, val} struct
        rule_idx: Index of the rule that produced this result

    Returns:
        Expression returning {idx, rule_idx, val} struct
    """
    return pl.struct(
        expr.struct.field("idx").alias("idx"),
        pl.lit(rule_idx).alias("rule_idx"),
        expr.struct.field("val").alias("val"),
    )


def default_get_prioritized_result(
    failed_rule_results: t.List[pl.Expr],
    current_result: t.Optional[pl.Expr],
    default_expr: pl.Expr,
) -> pl.Expr:
    """Default prioritization logic: return current result or last failed result.

    Args:
        failed_rule_results: List of results from rules that fell through (idx=-1)
        current_result: Current rule's result (if it matched), or None (if we're in otherwise)
        default_expr: Default fallback expression

    Returns:
        Expression to use as the prioritized result
    """
    if current_result is None:
        if len(failed_rule_results) == 0:
            return default_expr
        return failed_rule_results[-1]
    return current_result


def prioritize_results(
    results: t.List[pl.Expr],
    default_expr: pl.Expr,
    format_prioritized_fn: t.Optional[
        t.Callable[[t.List[pl.Expr], t.Optional[pl.Expr], pl.Expr], pl.Expr]
    ] = None,
) -> pl.Expr:
    """Return the first result (lowest rule index) where idx != -1.

    Falls back to a sentinel default struct {idx=-1, rule_idx=-1, val=default_expr}
    when every rule fell through.

    Args:
        results: List of result expressions (each returns {idx, val} struct)
        default_expr: Default expression to use when all rules fall through
        format_prioritized_fn: Optional custom prioritization logic

    Returns:
        Expression returning {idx, rule_idx, val} struct with the prioritized result
    """
    default_struct = pl.struct(
        pl.lit(-1).alias("idx"),
        pl.lit(-1).alias("rule_idx"),
        default_expr.alias("val"),
    )

    if format_prioritized_fn is None:
        format_prioritized_fn = default_get_prioritized_result

    out_expr = pl
    previous_results = []

    for i, rule_res in enumerate(results):
        indexed_res = _with_rule_idx(rule_res, i)
        out_expr = out_expr.when(rule_res.struct.field("idx") != -1).then(
            format_prioritized_fn(previous_results, indexed_res, default_struct)
        )
        previous_results.append(indexed_res)

    if out_expr is pl:
        return default_struct
    return out_expr.otherwise(
        format_prioritized_fn(previous_results, None, default_struct)
    )


def wrap_output_fn_for_index(fn: OutputFn) -> OutputFn:
    """Wrap an OutputFn so its return value is embedded in a {idx, val} struct.

    idx is the result_idx of the matched leaf (-1 for the default/no-match leaf).
    Uses **kwargs so the wrapper never needs updating if OutputFn's signature changes.
    """

    @functools.wraps(fn)
    def inner(**kwargs: t.Any) -> pl.Expr:
        val = fn(**kwargs)
        return pl.struct(
            pl.lit(kwargs["result_idx"]).alias("idx"),
            val.alias("val"),
        )

    return inner  # type: ignore[return-value]


def extract_value(prioritized: pl.Expr) -> pl.Expr:
    """Default post-processor: unwrap the val field from a prioritized result struct.

    Args:
        prioritized: Expression returning {idx, rule_idx, val} struct

    Returns:
        Expression returning just the val field
    """
    return prioritized.struct.field("val")


def execute_prioritized_rules(
    rules: t.List[RuleRoot],
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr] = None,
    post_process_fn: t.Optional[t.Callable[[pl.Expr], pl.Expr]] = None,
    format_prioritized_fn: t.Optional[
        t.Callable[[t.List[pl.Expr], t.Optional[pl.Expr], pl.Expr], pl.Expr]
    ] = None,
    **inputs: pl.Expr,
) -> pl.Expr:
    """Execute multiple rules and return the result of the first one that matches.

    Falls back to default_literal if none match.

    Args:
        rules: List of RuleRoots to execute in priority order
        builder_config: Configuration with result builder and output literals
        parameters: Optional parameters struct expression
        post_process_fn: Applied to the full prioritized struct {idx, rule_idx, val}
                        (defaults to extract_value which returns just val)
        format_prioritized_fn: Optional custom prioritization logic
        **inputs: Feature columns as pl.Expr

    Returns:
        Expression with the prioritized result (post-processed)
    """
    from dataclasses import replace

    # Process the default result first
    default_result_expr = builder_config.build_result_function(
        inputs=inputs,
        branch_stack=(),
        config=builder_config,
        result_idx=-1,
    )

    # Wrap the result builder to inject {idx, val} structs
    wrapped_config = replace(
        builder_config,
        build_result_function=wrap_output_fn_for_index(
            builder_config.build_result_function
        ),
    )

    # Execute all rules
    results = execute_rule_list(
        rules=rules,
        builder_config=wrapped_config,
        parameters=parameters,
        **inputs,
    )

    # Prioritize results
    prioritized = prioritize_results(
        results=results,
        default_expr=default_result_expr,
        format_prioritized_fn=format_prioritized_fn,
    )

    # Post-process (default: extract value)
    post_process_fn = post_process_fn or extract_value
    return post_process_fn(prioritized)


def execute_prioritized_flat_rule_tree(
    tree: FlatRuleTree,
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr] = None,
    post_process_fn: t.Optional[t.Callable[[pl.Expr], pl.Expr]] = None,
    format_prioritized_fn: t.Optional[
        t.Callable[[t.List[pl.Expr], t.Optional[pl.Expr], pl.Expr], pl.Expr]
    ] = None,
    **inputs: pl.Expr,
) -> pl.Expr:
    """Execute a FlatRuleTree with prioritization (first matching rule wins).

    Args:
        tree: FlatRuleTree to execute
        builder_config: Configuration with result builder and output literals
        parameters: Optional parameters struct expression
        post_process_fn: Applied to the full prioritized struct
        format_prioritized_fn: Optional custom prioritization logic
        **inputs: Feature columns as pl.Expr

    Returns:
        Expression with the prioritized result
    """
    return execute_prioritized_rules(
        rules=tree.rules,
        builder_config=builder_config,
        parameters=parameters,
        post_process_fn=post_process_fn,
        format_prioritized_fn=format_prioritized_fn,
        **inputs,
    )


# =============================================================================
# DataFrame Integration
# =============================================================================


def execute_rule_on_frame(
    frame: pl.LazyFrame,
    rule: RuleRoot,
    builder_config: BuilderConfig,
    result_col: str = "result",
    parameters_col: str = "parameters",
    default_parameters: t.Optional[t.Dict[str, t.Any]] = None,
) -> pl.LazyFrame:
    """Execute a single rule over a LazyFrame, appending the result as a new column.

    Args:
        frame: Input LazyFrame
        rule: RuleRoot to execute
        builder_config: Configuration with result builder and output literals
        result_col: Name of the output column
        parameters_col: Name of the parameters column in the DataFrame
        default_parameters: Default parameter values (fallback when runtime params are None)

    Returns:
        LazyFrame with result_col added

    Raises:
        ValueError: If required features are missing from the frame
    """
    # Get required features from the rule
    # TODO: Implement get_required_features() on RuleType classes
    # For now, we'll need to pass all columns
    schema_names = frame.collect_schema().names()
    inputs: t.Dict[str, pl.Expr] = {col: pl.col(col) for col in schema_names}

    # Handle parameters
    parameters = None
    if parameters_col in schema_names:
        parameters = pl.col(parameters_col)
    elif default_parameters:
        # Build parameters from defaults
        parameter_literals = {k: pl.lit(v) for k, v in default_parameters.items()}
        parameters = pl.struct(
            *[expr.alias(name) for name, expr in parameter_literals.items()]
        )

    result_expr = execute_rule_root(
        rule_root=rule,
        builder_config=builder_config,
        parameters=parameters,
        **inputs,
    )

    return frame.with_columns(result_expr.alias(result_col))


def execute_flat_rule_tree_on_frame(
    frame: pl.LazyFrame,
    tree: FlatRuleTree,
    builder_config: BuilderConfig,
    result_col: str = "result",
    use_prioritization: bool = True,
    parameters_col: str = "parameters",
    default_parameters: t.Optional[t.Dict[str, t.Any]] = None,
) -> pl.LazyFrame:
    """Execute a FlatRuleTree over a LazyFrame, appending the result as a new column.

    Args:
        frame: Input LazyFrame
        tree: FlatRuleTree to execute
        builder_config: Configuration with result builder and output literals
        result_col: Name of the output column
        use_prioritization: If True, use first-match prioritization; if False, execute
                           all rules and return a list of results
        parameters_col: Name of the parameters column in the DataFrame
        default_parameters: Default parameter values (fallback when runtime params are None)

    Returns:
        LazyFrame with result_col added
    """
    schema_names = frame.collect_schema().names()
    inputs: t.Dict[str, pl.Expr] = {col: pl.col(col) for col in schema_names}

    # Handle parameters
    parameters = None
    if parameters_col in schema_names:
        parameters = pl.col(parameters_col)
    elif default_parameters:
        parameter_literals = {k: pl.lit(v) for k, v in default_parameters.items()}
        parameters = pl.struct(
            *[expr.alias(name) for name, expr in parameter_literals.items()]
        )

    if use_prioritization:
        result_expr = execute_prioritized_flat_rule_tree(
            tree=tree,
            builder_config=builder_config,
            parameters=parameters,
            **inputs,
        )
    else:
        # Execute all rules, return list
        results = execute_flat_rule_tree(
            tree=tree,
            builder_config=builder_config,
            parameters=parameters,
            **inputs,
        )
        result_expr = pl.concat_list(results)

    return frame.with_columns(result_expr.alias(result_col))
