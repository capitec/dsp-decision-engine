import typing as t
import enum
from pydantic import Field
from ped.modules.core import BaseModule, PEDNode
from ped.serializable.function import DefinedFunction
from .tree import Tree
from .shared import WithTreeOutput
from .impl import execute_tree, execute_prioritized_trees, build_parameters_expr, default_result_builder, extract_value
from .nodes import NodeType, BuilderConfig


class PrioritizationMode(str, enum.Enum):
    first_match = "first_match"
    all = "all"


class TreeModule(Tree, BaseModule):
    """A single decision tree evaluated as a Polars expression."""
    type: t.Literal["tree"]
    result_col: str = Field(default="result", description="Name of the output column / node")
    output_fn: t.Optional[DefinedFunction] = None

    output_name: t.Optional[str] = "result"

    def expand_nodes(self) -> t.List[PEDNode]:
        # tree = self
        # output_fn = self.output_fn.get_function() if self.output_fn else None

        # def _run(**inputs: pl.Expr) -> pl.Expr:
        #     return execute_tree(inputs=inputs, tree=tree, output_fn=output_fn)

        # required = tree.get_required_features()
        # if tree.get_required_parameters():
        #     required = required | {tree.parameters_col}

        # node = PEDNode(
        #     name=self.result_col,
        #     callable=_run,
        #     input_map={col: col for col in required},
        # )
        # return [node]
        output_fn = self.output_fn.get_function() if self.output_fn else default_result_builder
        config = BuilderConfig(
            build_result_function=output_fn,
            output_literals=self._output_literals,
            default_literal=self._default_literal,
        )
        required_parameters = self.tree.get_required_parameters()
        res = []
        extra_input = {}
        if required_parameters:
            res.append(PEDNode.from_callable(
                name="_parameter_expression",
                func=build_parameters_expr,
                input_map={"runtime_params": self.parameters_col},
                static_kwargs={"default_params": self.default_parameters},
            )) 
            extra_input["_parameter_expression"] = "_parameter_expression"
        res.append(PEDNode.from_callable(
            name=self.result_col,
            func=execute_tree,
            input_map={col: col for col in self.get_required_features()}|extra_input,
            static_kwargs={"tree": self.root, "builder_config": config},
        ))
        return res



class PrioritizedTreeModule(WithTreeOutput, BaseModule):
    """Multiple decision trees evaluated in order; the first real match wins."""
    type: t.Literal["prioritized_tree"]
    result_col: str = Field(default="result", description="Name of the output column / node")
    output_name: t.Optional[str] = "result"

    roots: t.List[NodeType]
    parameters_col: str = "parameters"
    default_parameters: t.Dict[str, t.Any] = Field(default_factory=dict)
    mode: PrioritizationMode = Field(
        default=PrioritizationMode.first_match,
        description="'first_match' returns the first tree that matches; 'all' is reserved for future use.",
    )
    output_fn: t.Optional[DefinedFunction] = None
    post_process_fn: t.Optional[DefinedFunction] = None

    def _as_trees(self) -> t.List[Tree]:
        """Build individual Tree objects (one per root) for execution."""
        return [
            Tree(
                root=root,
                output=self.output,
                schema=self.schema,
                default=self.default,
                parameters_col=self.parameters_col,
                default_parameters=self.default_parameters,
            )
            for root in self.roots
        ]

    def to_ui_tree(self) -> "V2Tree":
        """Convert this module back to a v2 UI tree (inverse of v2 Tree.to_tree_module())."""
        from ped.modules.tree.ui.v2.tree import Tree as V2Tree
        all_nodes, all_edges, subtree_roots = [], [], []
        for root in self.roots:
            root_id, nodes, edges = root.to_ui_nodes()
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            subtree_roots.append(root_id)
        return V2Tree(
            nodes=all_nodes, edges=all_edges, subtrees=subtree_roots,
            output=self.output, schema=self.schema, default=self.default,
        )

    def expand_nodes(self) -> t.List[PEDNode]:
        output_fn = self.output_fn.get_function() if self.output_fn else default_result_builder
        post_process_fn = self.post_process_fn.get_function() if self.post_process_fn else extract_value
        config = BuilderConfig(
            build_result_function=output_fn,
            output_literals=self._output_literals,
            default_literal=self._default_literal,
        )
        required_parameters = set().union(*[r.get_required_parameters() for r in self.roots])
        required_vars = set().union(*[r.get_required_features() for r in self.roots])
        res = []
        extra_input = {}
        if required_parameters:
            res.append(PEDNode.from_callable(
                name="_parameter_expression",
                func=build_parameters_expr,
                input_map={"runtime_params": self.parameters_col},
                static_kwargs={"default_params": self.default_parameters},
            )) 
            extra_input["parameters"] = "_parameter_expression"
        if self.mode == PrioritizationMode.first_match:
            res.append(PEDNode.from_callable(
                name=self.result_col,
                func=execute_prioritized_trees,
                input_map={col: col for col in required_vars}|extra_input,
                static_kwargs={"roots": self.roots, "builder_config": config, "post_process_fn": post_process_fn},
            ))
        elif self.mode == PrioritizationMode.all:
            res.append(PEDNode.from_callable(
                name=self.result_col,
                func=execute_tree_list,
                input_map={col: col for col in required_vars}|extra_input,
                static_kwargs={"roots": self.roots, "builder_config": config},
            ))

        return res
