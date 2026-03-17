import typing as t
import functools
import polars as pl
from .tree import SubTreeRoot
from .nodes import BuilderConfig, TBranchStack
from dataclasses import replace



def default_result_builder(
    inputs: t.Dict[str, pl.Expr],
    branch_stack: 'TBranchStack',
    config: BuilderConfig,
    result_idx: int,
) -> pl.Expr:
    """Default output function: returns the output literal at result_idx,
    or the default literal when result_idx is -1 (the no-match leaf)."""
    if result_idx == -1:
        return config.default_literal
    return config.output_literals[result_idx]


# Single source of truth — OutputFn always matches default_result_builder's signature.
# Update that function and this alias updates automatically.
OutputFn = t.Type[default_result_builder]




def build_parameters_expr(
    runtime_params: t.Optional[pl.Expr],
    default_params: t.Dict[str, t.Any],
) -> t.Optional[pl.Expr]:
    """Return a parameters struct expression.

    Runtime parameters take precedence; when no runtime column is present the
    default_parameters dict is materialised as a literal struct expression.
    """
    if runtime_params is not None:
        return runtime_params
    if default_params:
        return pl.lit(default_params)
    return None


def execute_tree(
    tree_root: SubTreeRoot,
    builder_config: BuilderConfig,
    _parameter_expression: t.Optional[pl.Expr]=None,
    **inputs: pl.Expr,
) -> pl.Expr:
    """Execute a single decision tree and return the resulting Polars expression."""

    return tree_root.node.build_expression(
        inputs=inputs,
        branch_stack=(),
        config=replace(builder_config, root_meta=tree_root.meta),
        parameters=_parameter_expression,
    )


def execute_tree_list(
    roots: t.List[SubTreeRoot],
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr]=None,
    **inputs: pl.Expr,
) -> t.List[pl.Expr]:
    """Execute multiple decision trees and return one output expression per tree."""
    return [
        execute_tree(**inputs, tree_root=root, builder_config=builder_config, parameters=parameters) 
        for root in roots
    ]


# ---------------------------------------------------------------------------
# Prioritised multi-tree execution
# ---------------------------------------------------------------------------

def _with_tree_idx(expr: pl.Expr, tree_idx: int) -> pl.Expr:
    """Enrich an indexed result struct {idx, val} with the originating tree index."""
    return pl.struct(
        expr.struct.field("idx").alias("idx"),
        pl.lit(tree_idx).alias("tree_idx"),
        expr.struct.field("val").alias("val"),
    )


def prioritize_results(
    results: t.List[pl.Expr],
    default_expr: pl.Expr,
) -> pl.Expr:
    """Return the first result (lowest tree index) where idx != -1 (a real branch
    was taken), enriched with tree_idx.  Falls back to a sentinel default struct
    {idx=-1, tree_idx=-1, val=default_expr} when every tree fell through."""
    default_struct = pl.struct(
        pl.lit(-1).alias("idx"),
        pl.lit(-1).alias("tree_idx"),
        default_expr.alias("val"),
    )

    out_expr = pl
    for i, tree_res in enumerate(results):
        indexed_res = _with_tree_idx(tree_res, i)
        out_expr = out_expr.when(tree_res.struct.field("idx") != -1).then(indexed_res)

    if out_expr is pl:
        return default_struct
    return out_expr.otherwise(default_struct)


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
    """Default post-processor: unwrap the val field from a prioritized result struct."""
    return prioritized.struct.field("val")


def execute_prioritized_trees(
    roots: t.List[SubTreeRoot],
    builder_config: BuilderConfig,
    parameters: t.Optional[pl.Expr]=None,
    post_process_fn: t.Optional[t.Callable[[pl.Expr], pl.Expr]] = None,
    **inputs: pl.Expr,
) -> pl.Expr:
    """Execute multiple trees and return the result of the first one (by position)
    that takes a real branch.  Falls back to default_expr if none match.

    post_process_fn is applied to the full prioritized struct {idx, tree_idx, val}
    and defaults to extract_value which simply returns the val field.
    """
    builder_config = replace(builder_config, build_result_function=wrap_output_fn_for_index(builder_config.build_result_function))

    post_process_fn = post_process_fn or extract_value

    results = execute_tree_list(**inputs, roots=roots, builder_config=builder_config, parameters=parameters)
    prioritized = prioritize_results(results=results, default_expr=builder_config.default_literal)
    return post_process_fn(prioritized)


# def execute_tree_on_frame(
#     frame: pl.LazyFrame,
#     tree: Tree,
#     result_col: str = "result",
#     output_fn: t.Optional[OutputFn] = None,
# ) -> pl.LazyFrame:
#     """Execute a decision tree over a LazyFrame, appending the result as a new column.

#     Required feature columns are selected from the frame by name.  The parameters
#     column is forwarded when it exists in the frame schema; if the tree needs
#     parameters but no column and no default_parameters are configured, a ValueError
#     is raised.
#     """
#     required_features = tree.get_required_features()
#     inputs: t.Dict[str, pl.Expr] = {col: pl.col(col) for col in required_features}

#     schema_names = frame.collect_schema().names()
#     if tree.parameters_col in schema_names:
#         inputs[tree.parameters_col] = pl.col(tree.parameters_col)
#     elif tree.get_required_parameters() and not tree.default_parameters:
#         raise ValueError(
#             f"Tree requires parameters {tree.get_required_parameters()} but column "
#             f"'{tree.parameters_col}' was not found in the frame and no "
#             f"default_parameters are configured on the Tree."
#         )

#     result_expr = execute_tree(inputs=inputs, tree=tree, output_fn=output_fn)
#     return frame.with_columns(result_expr.alias(result_col))
