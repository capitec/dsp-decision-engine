import typing
import pandas as pd
import collections.abc
from functools import partial
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, model_validator, ConfigDict
from spockflow.nodes import VariableNode


TOutput = typing.Union[typing.Callable[..., pd.DataFrame], pd.DataFrame, str]
TCond = typing.Union[typing.Callable[..., pd.Series], pd.Series, str]


class ConditionedNode(BaseModel):
    # TODO fix for pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: typing.Union[TOutput, "ChildTree", None] = None
    condition: typing.Optional[TCond] = None

    @staticmethod
    def _length_attr(attr):
        if attr is None:
            return 1
        if isinstance(attr, str):
            return 1
        if not isinstance(attr, collections.abc.Sized):
            return 1
        return len(attr)

    def __len__(self):
        len_attr = self._length_attr(self.value)
        if len_attr == 1:
            len_attr = self._length_attr(self.condition)
        return len_attr

    @model_validator(mode="after")
    def check_compatible_lengths(self) -> Self:
        len_value = self._length_attr(self.value)
        if len_value == 1:
            return self
        len_condition = self._length_attr(self.condition)
        if len_condition == 1:
            return self
        if len_condition != len_value:
            raise ValueError("Condition and value lengths incompatible")
        return self


class ChildTree(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    nodes: typing.List[ConditionedNode] = Field(default_factory=list)
    default_value: typing.Optional[TOutput] = None

    def __len__(
        self,
    ):  # TODO could maybe cache this on bigger trees if the values havent changed
        node_len = ConditionedNode._length_attr(self.default_value)
        if node_len != 1:
            return node_len
        for node in self.nodes:
            if (node_len := len(node)) != 1:
                return node_len
        return 1

    @model_validator(mode="after")
    def check_compatible_lengths(self) -> Self:
        child_tree_len = ConditionedNode._length_attr(self.default_value)
        for node in self.nodes:
            len_value = len(node)
            if len_value == 1:
                continue
            if child_tree_len == 1:
                child_tree_len = len_value
            elif child_tree_len != len_value:
                raise ValueError(
                    f"Lengths of values or conditions in the tree is incompatible. Found {child_tree_len} != {len_value}."
                )
        return self

    def add_node(self, value: TOutput, condition: TCond, **kwargs) -> ConditionedNode:
        len_value = ConditionedNode._length_attr(value)
        len_condition = ConditionedNode._length_attr(condition)
        if len_value != 1 and len_condition != 1 and len_value != len_condition:
            raise ValueError(
                f"Cannot add node as the length of the value ({len_value}) is not compatible with the length of the condition ({len_condition})."
            )
        if len_value != 1 or len_condition != 1:
            # TODO adding this allows better validation but requires circular loops so hard for pydantic to serialise
            # len_tree = len(self.root_tree.root)
            len_tree = len(self)
            if len_value != 1 and len_tree != 1 and len_value != len_tree:
                raise ValueError(
                    f"Cannot add node as the length of the value ({len_value}) incompatible with tree {len_tree}."
                )
            if len_condition != 1 and len_tree != 1 and len_condition != len_tree:
                raise ValueError(
                    f"Cannot add node as the length of the condition ({len_condition}) incompatible with tree {len_tree}."
                )
        node = ConditionedNode(value=value, condition=condition, **kwargs)
        self.nodes.append(node)
        return node

    def set_default(self, value: TOutput):
        if self.default_value is not None:
            raise ValueError("Default value already set")
        len_value = ConditionedNode._length_attr(value)
        if len_value != 1:
            # TODO adding this allows better validation but requires circular loops so hard for pydantic to serialise
            # len_tree = len(self.root_tree.root)
            len_tree = len(self)
            if len_tree != 1 and len_value != len_tree:
                raise ValueError(
                    f"Cannot set default as length of value ({len_value}) incompatible with tree {len_tree}."
                )

    def merge_into(self, other: Self):
        len_subtree = len(other)
        if len_subtree != 1:
            len_tree = len(self)
            if len_tree != 1 and len_subtree != len_tree:
                raise ValueError(
                    f"Subtree length ({len_subtree}) incompatible with tree ({len_tree})."
                )

        if other.default_value is not None and self.default_value is not None:
            raise ValueError(
                f"Cannot merge two subtrees both containing default values"
            )
        self.nodes.extend(other.nodes)


class WrappedTreeFunction(ABC):
    @abstractmethod
    def __call__(self, *args: typing.Any, **kwds: typing.Any) -> typing.Any: ...
    @abstractmethod
    def condition(
        self, output: TOutput = None, condition: TCond = None, **kwargs
    ) -> typing.Union[Self, TCond]: ...
    @abstractmethod
    def set_default(self, output: TOutput): ...
    @abstractmethod
    def include_subtree(
        self, subtree: typing.Union[Self, "ChildTree"], condition=None, **kwargs
    ): ...


class Tree(VariableNode):
    doc: str = "This executes a user defined decision tree"
    root: ChildTree = Field(default_factory=ChildTree)

    def compile(self):
        from .compiled import CompiledNumpyTree

        return CompiledNumpyTree(self)

    def _generate_runtime_nodes(
        self, config: "typing.Dict[str, typing.Any]", compiled_node: "CompiledNumpyTree"
    ) -> "typing.List[node.Node]":
        from hamilton import node

        return [
            node.Node.from_fn(
                compiled_node._flattened_tree.conditions[exec_cond], name=exec_cond
            )
            for exec_cond in compiled_node.execution_conditions
            if callable(compiled_node._flattened_tree.conditions[exec_cond])
        ]

    def _inject_child_tree_from_node(
        self, *args, child_node: ConditionedNode, function, **kwargs
    ):
        if child_node.value is None:
            child_node.value = ChildTree()
        if not isinstance(child_node.value, ChildTree):
            raise ValueError(
                f"Subtree must have no value set or already be associated to a subtree. Found value = {child_node.value}"
            )
        return function(*args, **kwargs, child_tree=child_node.value)

    def _wrap_function(self, condition, child_node: ConditionedNode):
        import inspect

        # Currently only wrap functions can maybe consider wrapping callable
        if not inspect.isfunction(condition):

            def raise_cannot_wrap_error(*args, **kwargs):
                if len(args) == 1 and isinstance(args[0], typing.Callable):
                    raise ValueError("Cannot wrap function when condition is specified")
                raise NotImplementedError()

            condition = raise_cannot_wrap_error
        # condition.condition = self._add_condition_to_child(child_node)
        condition.condition = partial(
            self._inject_child_tree_from_node,
            child_node=child_node,
            function=self.condition,
        )
        condition.set_default = partial(
            self._inject_child_tree_from_node,
            child_node=child_node,
            function=self.set_default,
        )
        condition.include_subtree = partial(
            self._inject_child_tree_from_node,
            child_node=child_node,
            function=self.include_subtree,
        )
        return condition

    def _identify_loops(self, *nodes: "ConditionedNode"):
        q = list(nodes)
        seen = set()
        while q:
            el = q.pop()
            if id(el) in seen:
                raise ValueError("Tree must not contain any loops")
            seen.add(id(el))
            if isinstance(el.value, ChildTree):
                q.extend(el.value.nodes)
                if isinstance(el.value.default_value, ChildTree):
                    q.extend(el.value.default_value.nodes)

    @staticmethod
    def _remove_nodes_from_end(*nodes: "ConditionedNode", child_tree: ChildTree):
        # Not the most pythonic method can maybe be improved
        # Done to remove elements in the order they were added
        nodes_to_remove = list(nodes)
        last_el_idx = len(child_tree.nodes) - 1
        for rev_i, node in enumerate(reversed(child_tree.nodes)):
            i = last_el_idx - rev_i
            if node in nodes_to_remove:
                child_tree.nodes.pop(i)
            if len(nodes_to_remove) <= 0:
                return True
        return False

    def copy(self, deep=True):
        return self.model_copy(deep=deep)

    # def parameterize_condition(self) # TODO
    def condition(
        self,
        output: TOutput = None,
        condition: typing.Union[TCond, None] = None,
        child_tree: ChildTree = None,
        **kwargs,
    ) -> typing.Callable[..., WrappedTreeFunction]:
        """
        Define a condition in the decision tree.

        Args:
            output (Optional[TOutput]): The output or action associated with this condition.
            condition (Optional[TCond]): The condition to evaluate.
            child_tree (ChildTree, optional): The subtree to add the condition to. Defaults to self.root.
            **kwargs: Additional keyword arguments passed to the tree node.

        Returns:
            Callable[..., WrappedTreeFunction]: A callable function that wraps the condition
            and integrates it into the decision tree.

        Notes:
            - If `child_tree` is not provided, it defaults to `self.root`.
            - The function `wrapper` adds a node to `child_tree` with the specified `output`, `condition`, and `kwargs`.
            - It ensures that loops in the decision tree are identified and managed to prevent infinite recursion.
            - Returns a wrapped function that incorporates the condition into the decision tree structure.

        Raises:
            ValueError: If a loop is detected in the decision tree structure.

        Usage:
            Define a condition in the decision tree by specifying `output` and `condition` parameters,
            optionally providing additional `kwargs` for customization.
            If `condition` is provided, directly adds the condition to the tree.
            If `condition` is not provided initially, returns a function (`wrapper`) that can later be used
            to add the condition to the tree.

        Example:
            Consider defining a condition 'A > 5' with an output action 'Reject' in a decision tree:

            >>> tree.set_default(output=pd.DataFrame({"value":['Reject']}), condition="a")

        """
        if child_tree is None:
            child_tree = self.root

        def wrapper(condition: TCond):
            nonlocal output
            if isinstance(output, Tree):
                output = output.root
            node = child_tree.add_node(value=output, condition=condition, **kwargs)
            try:
                self._identify_loops(node)
            except ValueError as e:
                # Try to keep the order
                self._remove_nodes_from_end(node, child_tree=child_tree)
                raise e from e
            return self._wrap_function(condition, node)

        # Allow this to be used as both a wrapper and not
        if condition is not None:
            return wrapper(condition)
        return wrapper

    def set_default(self, output: TOutput, child_tree: ChildTree = None):
        """
        Set the default output or action for the decision tree.

        Args:
            output (TOutput): The default output or action to set for the decision tree.
            child_tree (ChildTree, optional): The subtree to set the default for. Defaults to self.root.

        Raises:
            ValueError: If a default value is already set for the subtree (`child_tree`).

        Notes:
            - If `child_tree` is not provided, the default is set for `self.root`.
            - Checks if a default value is already assigned to `child_tree`. If so, raises an error.
            - Converts `output` to the root node of `output` if `output` is an instance of `Tree`.
            - Sets `child_tree.default_value` to `output`, establishing it as the default action
            when no specific conditions are met in the decision tree.

        Usage:
            Set a default action or output for a decision tree using the `output` parameter.
            Optionally, specify `child_tree` to set the default within a specific subtree of the decision tree.

        Example:
            Setting a default action 'Log' for a decision tree:

            >>> tree.set_default(output=pd.DataFrame({"value":['Log']}))

        """
        if child_tree is None:
            child_tree = self.root
        if child_tree.default_value is not None:
            raise ValueError("Default value already set")
        if isinstance(output, Tree):
            output = output.root
        child_tree.default_value = output

    def include_subtree(
        self,
        subtree: typing.Union[Self, ChildTree],
        condition=None,
        child_tree: ChildTree = None,
        **kwargs,
    ):
        """Include a subtree into the current decision tree structure.

        Args:
            subtree (Union[Tree, ChildTree]): The subtree to include. If `self`, refers to the current instance.
            condition (Optional): The condition under which to include the subtree. Defaults to None.
            child_tree (ChildTree, optional): The subtree or root node to merge into. Defaults to self.root.
            **kwargs: Additional keyword arguments passed to the tree node.

        Raises:
            ValueError: If a loop is detected in the decision tree structure.

        Notes:
            - If `child_tree` is not provided, defaults to `self.root`.
            - Checks if `subtree` is an instance of `Tree`. If so, assigns `subtree.root` to `subtree`.
            - Merges `subtree` into `child_tree` if no specific `condition` is provided.
            - Adds `subtree` as a node under `condition` within `child_tree` if `condition` is specified.
            - Calls `_identify_loops` to ensure there are no loops in the decision tree structure.
            If a loop is detected, attempts to remove recently added nodes to maintain order and raises an error.

        Usage:
            Include a subtree (`subtree`) into the current decision tree (`self` or `child_tree`).
            Optionally, specify a `condition` under which to include `subtree`.
            Additional `kwargs` can be used to customize the inclusion process.

        Example:
            Including a subtree `subtree` into the main decision tree `tree`:

            >>> tree.include_subtree(subtree)

            Including `subtree` under condition 'cond_subtree' in `tree`:

            >>> tree.include_subtree(subtree, condition=lambda a: 'cond_subtree')
        """
        if child_tree is None:
            child_tree = self.root

        if isinstance(subtree, Tree):
            subtree = subtree.root

        if condition is None:
            child_tree.merge_into(subtree)
            new_nodes = child_tree.nodes
        else:
            new_nodes = [child_tree.add_node(subtree, condition, **kwargs)]

        try:
            self._identify_loops(*new_nodes)
        except ValueError as e:
            # Try to keep the order
            self._remove_nodes_from_end(*new_nodes, child_tree=child_tree)
            raise e from e

    def visualize(self, get_value_name=None, get_condition_name=None):
        """
        Generate a visualization of the decision tree using Graphviz.

        Args:
            get_value_name (callable, optional): Function to retrieve names for node values. Defaults to None.
                If not provided, uses a default function `get_name` from `spockflow._util`.
            get_condition_name (callable, optional): Function to retrieve names for node conditions. Defaults to None.
                If not provided, uses a default function `get_name` from `spockflow._util`.

        Returns:
            graphviz.Digraph: A Graphviz representation of the decision tree.

        Notes:
            - Uses Graphviz (`graphviz.Digraph`) to create a visual representation of the decision tree.
            - Iterates through the nodes of the tree starting from the root and constructs nodes and edges accordingly.
            - If `get_value_name` or `get_condition_name` are not provided, default functions from `spockflow._util`
            are used to generate node names.
            - Nodes representing conditions are displayed as ellipses, and nodes representing values/actions are
            displayed as filled rectangles.
            - Edges between nodes represent the flow of decision-making in the tree.
            - The visualization includes both nodes with specific conditions and default nodes.

        Usage:
            Generate a graphical representation of the decision tree structure for visualization and analysis.
            Optionally, provide custom functions `get_value_name` and `get_condition_name` to customize the display
            names of node values and conditions.

        Example:
            Visualizing a decision tree `tree`:

            >>> dot = tree.visualize()

            Saving the visualization to a file:

            >>> dot.render('decision_tree', format='png', view=True)

        """
        import graphviz

        if get_value_name is None:
            from spockflow._util import get_name

            get_value_name = lambda x: get_name(x, None)
        if get_condition_name is None:
            from spockflow._util import get_name

            get_condition_name = lambda x: (
                x if isinstance(x, str) else get_name(x, None)
            )
        to_search = [(self.root, "root")]
        dot = graphviz.Digraph()

        while to_search:
            curr, curr_name = to_search.pop()
            dot.node(curr_name, curr_name)

            for node in curr.nodes:
                node_condition_name = get_condition_name(node.condition)
                dot.node(node_condition_name, node_condition_name)
                dot.edge(curr_name, node_condition_name)
                if hasattr(node.value, "nodes"):
                    to_search.extend([(node.value, node_condition_name)])
                elif node.value is not None:
                    node_value_name = get_value_name(node.value)
                    dot.node(
                        node_value_name,
                        node_value_name,
                        style="filled",
                        fillcolor="#ADDFFF",
                        shape="rectangle",
                    )
                    dot.edge(node_condition_name, node_value_name)

            if curr.default_value is not None:
                default_name = get_value_name(curr.default_value)
                dot.node(
                    default_name,
                    default_name,
                    style="filled",
                    fillcolor="#ADDFFF",
                    shape="rectangle",
                )
                dot.edge(curr_name, default_name)
        return dot
