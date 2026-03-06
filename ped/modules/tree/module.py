import typing as t
import enum
import polars as pl
from pydantic import Field
from ped.modules.core import BaseModule, PEDNode
from .tree import Tree
from .impl import OutputFn, execute_tree, execute_prioritized_trees


class PrioritizationMode(enum.StrEnum):
    first_match = "first_match"
    all = "all"


class TreeModule(BaseModule):
    """A single decision tree evaluated as a Polars expression."""
    type: t.Literal["tree"] = "tree"
    tree: Tree
    result_col: str = Field(default="result", description="Name of the output column / node")
    output_fn: t.Optional[OutputFn] = Field(default=None, exclude=True)

    output_name: t.Optional[str] = "result"

    def expand_nodes(self) -> t.List[PEDNode]:
        tree = self.tree
        output_fn = self.output_fn

        def _run(**inputs: pl.Expr) -> pl.Expr:
            return execute_tree(inputs=inputs, tree=tree, output_fn=output_fn)

        required = tree.get_required_features()
        if tree.get_required_parameters():
            required = required | {tree.parameters_col}

        node = PEDNode(
            name=self.result_col,
            callable=_run,
            input_map={col: col for col in required},
        )
        return [node]


class PrioritizedTreeModule(BaseModule):
    """Multiple decision trees evaluated in order; the first real match wins."""
    type: t.Literal["prioritized_tree"] = "prioritized_tree"
    trees: t.List[Tree]
    default_expr: t.Optional[pl.Expr] = Field(
        default=None,
        exclude=True,
        description="Fallback Polars expression when no tree matches. Defaults to pl.lit(None).",
    )
    result_col: str = Field(default="result", description="Name of the output column / node")
    mode: PrioritizationMode = Field(
        default=PrioritizationMode.first_match,
        description="'first_match' returns the first tree that matches; 'all' is reserved for future use.",
    )
    output_fn: t.Optional[OutputFn] = Field(default=None, exclude=True)
    post_process_fn: t.Optional[t.Callable[[pl.Expr], pl.Expr]] = Field(default=None, exclude=True)

    output_name: t.Optional[str] = "result"

    def expand_nodes(self) -> t.List[PEDNode]:
        if self.mode != PrioritizationMode.first_match:
            raise NotImplementedError(f"Prioritization mode '{self.mode}' is not yet implemented.")

        trees = self.trees
        default_expr = self.default_expr or pl.lit(None)
        output_fn = self.output_fn
        post_process_fn = self.post_process_fn

        def _run(**inputs: pl.Expr) -> pl.Expr:
            return execute_prioritized_trees(
                inputs=inputs,
                trees=trees,
                default_expr=default_expr,
                output_fn=output_fn,
                post_process_fn=post_process_fn,
            )

        required: t.Set[str] = set()
        for tree in trees:
            required |= tree.get_required_features()
            if tree.get_required_parameters():
                required.add(tree.parameters_col)

        node = PEDNode(
            name=self.result_col,
            callable=_run,
            input_map={col: col for col in required},
        )
        return [node]