import typing
import re
import keyword
import functools
import pandas as pd
import collections.abc
from functools import partial
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    ConfigDict,
    field_serializer,
    PrivateAttr,
    AfterValidator,
    model_serializer,
)
from spockflow.nodes import VariableNode
from ...dtable import DecisionTable
from hamilton.node import DependencyType

from spockflow._serializable import (
    DataFrame,
    Series,
    dump_df_to_dict,
    dump_series_to_dict,
)

if typing.TYPE_CHECKING:
    from .compiled import CompiledNumpyTree


def _is_valid_function_name(name):
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    assert (
        re.match(pattern, name) and name not in keyword.kwlist
    ), f"{name} must be a valid python function name"
    return name


class TableCondition(BaseModel):
    name: typing.Annotated[str, AfterValidator(_is_valid_function_name)]
    table: DecisionTable


TOutput = typing.Union[typing.Callable[..., pd.DataFrame], DataFrame, str]
TCond = typing.Union[typing.Callable[..., pd.Series], Series, str]
TCondRaw = typing.Union[typing.Callable[..., pd.Series], pd.Series, TableCondition, str]

_TABLE_VALUE_KEY = "value"


def _length_attr(attr):
    if attr is None:
        return 1
    if isinstance(attr, str):
        return 1
    if isinstance(attr, TableCondition):
        return 1
    if not isinstance(attr, collections.abc.Sized):
        return 1
    return len(attr)


def _serialize_value(value: typing.Union[TOutput, "ChildTree", None]):
    if value is None:
        return value
    if isinstance(value, typing.Callable):
        return value.__name__
    if isinstance(value, pd.DataFrame):
        return dump_df_to_dict(value)
        res = value.to_dict(orient="records")
        if len(res) == 1:
            return res[0]
        return res
    return value


class TableConditionedNode(BaseModel):
    condition_type: typing.Literal["table"] = "table"
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    values: typing.List[typing.Union[TOutput, "ChildTree", None]] = None
    condition_table: str
    priority: typing.Optional[typing.List[int]] = None

    @field_serializer("values")
    def serialize_values(
        self, values: typing.List[typing.Union[TOutput, "ChildTree", None]], _info
    ):
        return [_serialize_value(v) for v in values]

    @model_validator(mode="after")
    def check_compatible_lengths(self) -> Self:
        self.ensure_length()
        return self

    def __len__(self):
        len_values = (_length_attr(v) for v in self.values)
        try:
            return next(v for v in len_values if v != 1)
        except StopIteration:
            return 1

    def ensure_length(self, tree_length: int = 1):
        len_values = [_length_attr(v) for v in self.values] + [tree_length]
        try:
            non_unit_value = next(v for v in len_values if v != 1)
        except StopIteration:
            non_unit_value = 1
        if not all(v == 1 or v == non_unit_value for v in len_values):
            raise ValueError("Incompatible value lengths detected")

    def _check_compatible_table(self, table: DecisionTable):
        assert len(table.outputs) == 1, "Table must have exactly one output"
        assert (
            _TABLE_VALUE_KEY in table.outputs
        ), f'Table must contain "{_TABLE_VALUE_KEY}" as an output key'
        default_value = set()
        if table.default_value is not None:
            assert (
                len(table.default_value.keys()) == 1
            ), "Table must have exactly one output in the default values"
            assert (
                _TABLE_VALUE_KEY in table.default_value
            ), f'Table must contain "{_TABLE_VALUE_KEY}" as an output key in the default values'
            assert (
                len(table.default_value) == 1
            ), f"Default value must be a dataframe of length 1."
            default_value = {table.default_value[_TABLE_VALUE_KEY].values[0]}
        table_output_values = set(table.outputs[_TABLE_VALUE_KEY]) | default_value
        last_value = len(table_output_values)
        assert (
            set(range(0, last_value)) == table_output_values
        ), "Table output values must be sequential integer indicies"
        assert last_value == len(
            self.values
        ), "There must be one output value for each index in the tree outputs"
        if self.priority is not None:
            assert last_value == len(
                self.priority
            ), "There must be one priority item for each index in the tree outputs"


class ConditionedNode(BaseModel):
    # TODO fix for pd.DataFrame
    condition_type: typing.Literal["base"] = "base"
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    value: typing.Union[TOutput, "ChildTree", None] = None
    condition: typing.Optional[TCond] = None
    priority: typing.Optional[int] = None

    @field_serializer("condition")
    def serialize_condition(self, condition: typing.Optional[TCond], _info):
        if condition is None:
            return condition
        if isinstance(condition, typing.Callable):
            return condition.__name__
        if isinstance(condition, pd.Series):
            return dump_series_to_dict(condition)
            values = condition.tolist()
            return {condition.name: values if len(values) > 1 else values[0]}
        return condition

    @field_serializer("value")
    def serialize_value(self, value: typing.Union[TOutput, "ChildTree", None], _info):
        return _serialize_value(value)

    def __len__(self):
        len_attr = _length_attr(self.value)
        if len_attr == 1:
            len_attr = _length_attr(self.condition)
        return len_attr

    @model_validator(mode="after")
    def check_compatible_lengths(self) -> Self:
        self.ensure_length()
        return self

    def ensure_length(self, tree_length: int = 1):
        len_value = _length_attr(self.value)
        len_condition = _length_attr(self.condition)
        count_unit_length = (len_value == 1) + (len_condition == 1) + (tree_length == 1)
        if count_unit_length >= 2:
            return
        if len_value == len_condition == tree_length:
            return
        raise ValueError("Condition and value and tree lengths are incompatible")


TConditionedNode = typing.Annotated[
    typing.Union[ConditionedNode, TableConditionedNode],
    Field(discriminator="condition_type"),
]


class ChildTree(BaseModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    nodes: typing.List[TConditionedNode] = Field(default_factory=list)
    default_value: typing.Optional[TOutput] = None
    _decision_tables: typing.Dict[str, DecisionTable] = PrivateAttr(
        default_factory=dict
    )

    def __len__(
        self,
    ):  # TODO could maybe cache this on bigger trees if the values havent changed
        node_len = _length_attr(self.default_value)
        if node_len != 1:
            return node_len
        for node in self.nodes:
            if (node_len := len(node)) != 1:
                return node_len
        return 1

    @model_validator(mode="after")
    def check_compatible_lengths(self) -> Self:
        child_tree_len = _length_attr(self.default_value)
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

    @staticmethod
    def _merge_decision_tables(
        to_be_updated: typing.Dict[str, DecisionTable],
        other: typing.Dict[str, DecisionTable],
    ):
        for k, v in other.items():
            if k in to_be_updated:
                assert (
                    to_be_updated[k] is v
                ), f"Decision table {k} added twice with different values."
            else:
                to_be_updated[k] = v

    # def _get_condition_from_table(self, condition: TCondRaw):
    #     if not isinstance(condition, TableCondition): return condition
    #     self._merge_decision_tables(
    #         self._decision_tables,
    #         {condition.name: condition.table}
    #     )
    #     return TableConditionReference(table=condition.name)

    def add_node(
        self,
        value: typing.Union[TOutput, typing.List[TOutput]],
        condition: TCondRaw,
        priority: typing.Union[int, typing.List[int], None] = None,
        **kwargs,
    ) -> ConditionedNode:

        if isinstance(condition, TableCondition):
            node = TableConditionedNode(
                values=value,
                condition_table=condition.name,
                priority=priority,
                **kwargs,
            )
            self._merge_decision_tables(
                self._decision_tables, {condition.name: condition.table}
            )
            node._check_compatible_table(condition.table)
        else:
            node = ConditionedNode(
                value=value, condition=condition, priority=priority, **kwargs
            )
        node.ensure_length(len(self))
        self.nodes.append(node)
        return node

    def set_default(self, value: TOutput):
        if self.default_value is not None:
            raise ValueError("Default value already set")
        len_value = _length_attr(value)
        if len_value != 1:
            # TODO adding this allows better validation but requires circular loops so hard for pydantic to serialise
            # len_tree = len(self.root_tree.root)
            len_tree = len(self)
            if len_tree != 1 and len_value != len_tree:
                raise ValueError(
                    f"Cannot set default as length of value ({len_value}) incompatible with tree {len_tree}."
                )
        self.default_value = value

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

        self._merge_decision_tables(self._decision_tables, other._decision_tables)

        if other.default_value is not None:
            self.set_default(other.default_value)

        self.nodes.extend(other.nodes)

    def get_all_decision_tables(self):
        tables = self._decision_tables.copy()
        for node in self.nodes:
            if isinstance(node, TableConditionedNode):
                node_values = node.values
            else:
                node_values = [node.value]
            for nv in node_values:
                if isinstance(nv, ChildTree):
                    self._merge_decision_tables(tables, nv.get_all_decision_tables())
        return tables


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


def _check_table_result_eq(value: int, **kwargs: pd.DataFrame) -> pd.Series:
    table_result = kwargs[next(iter(kwargs.keys()))]
    return table_result[_TABLE_VALUE_KEY] == value


class Tree(VariableNode):
    doc: str = "This executes a user defined decision tree"
    root: ChildTree = Field(default_factory=ChildTree)
    decision_tables: typing.Dict[str, DecisionTable] = Field(default_factory=dict)

    def get_decision_tables(self):
        decision_tables = self.decision_tables.copy()
        self.root._merge_decision_tables(
            decision_tables, self.root.get_all_decision_tables()
        )
        return decision_tables

    @model_serializer()
    def serialize_model(self):
        return {
            "doc": self.doc,
            "root": self.root,
            "decision_tables": self.get_decision_tables(),
        }

    def compile(self):
        from .compiled import CompiledNumpyTree

        return CompiledNumpyTree(self)

    def _generate_nodes(
        self,
        name: str,
        config: "typing.Dict[str, typing.Any]",
        include_runtime_nodes: bool = False,
    ) -> "typing.List[node.Node]":
        from hamilton import node

        compiled_node = self.compile()
        output_nodes = super()._generate_nodes(
            name=name,
            config=config,
            include_runtime_nodes=include_runtime_nodes,
            compiled_node_override=compiled_node,
        )
        base_node_ = node.Node.from_fn(_check_table_result_eq, name="temporary_node")
        for table_name, compiled_table in compiled_node.decision_tables.items():
            output_nodes.extend(
                compiled_table._generate_nodes(table_name, config, True)
            )
            unique_values = set(compiled_table.outputs[_TABLE_VALUE_KEY])
            for v in unique_values:
                v = int(v)  # Just to be safe

                output_nodes.append(
                    base_node_.copy_with(
                        name=f"{table_name}_is_{v}",
                        doc_string=f"This is a function used to determine if the result of {table_name} is {v} to be used in a decision tree.",
                        callabl=functools.partial(_check_table_result_eq, value=v),
                        input_types={
                            table_name: (pd.DataFrame, DependencyType.REQUIRED)
                        },
                        include_refs=False,
                    )
                )

        return output_nodes

    def _generate_runtime_nodes(
        self, config: "typing.Dict[str, typing.Any]", compiled_node: "CompiledNumpyTree"
    ) -> "typing.List[node.Node]":
        # This is used to extract the condition functions when running outside of hamilton context
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
            if isinstance(el, TableConditionedNode):
                for v in el.values:
                    if isinstance(v, ChildTree):
                        q.extend(v.nodes)
                        if isinstance(v.default_value, ChildTree):
                            q.extend(v.default_value.nodes)

            elif isinstance(el.value, ChildTree):
                q.extend(el.value.nodes)
                if isinstance(el.value.default_value, ChildTree):
                    q.extend(el.value.default_value.nodes)

    @staticmethod
    def _remove_nodes_from_end(*nodes: "ConditionedNode", child_tree: ChildTree):
        # Not the most pythonic method can maybe be improved
        # Done to remove elements in the order they were added
        nodes_to_remove = set(map(id, list(nodes)))
        last_el_idx = len(child_tree.nodes) - 1
        for rev_i, node in enumerate(reversed(child_tree.nodes)):
            i = last_el_idx - rev_i
            if id(node) in nodes_to_remove:
                child_tree.nodes.pop(i)
                nodes_to_remove.discard(id(node))
                if len(nodes_to_remove) <= 0:
                    return True
        return False

    def copy(self, deep=True):
        return self.model_copy(deep=deep)

    # def parameterize_condition(self) # TODO
    def condition(
        self,
        output: TOutput = None,
        condition: typing.Optional[TCondRaw] = None,
        child_tree: ChildTree = None,
        priority: typing.Optional[int] = None,
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
            node = child_tree.add_node(
                value=output, condition=condition, priority=priority, **kwargs
            )
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
            - if `output` is an instance of `Tree`, add it as subtree.
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
            self.include_subtree(output, condition=None, child_tree=child_tree)
        else:
            child_tree.set_default(output)

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
                node_condition_name = get_condition_name(
                    node.condition_table
                    if isinstance(node, TableConditionedNode)
                    else node.condition
                )
                dot.node(node_condition_name, node_condition_name)
                dot.edge(curr_name, node_condition_name)
                if isinstance(node, TableConditionedNode):
                    node_values = node.values
                else:
                    node_values = [node.value]
                for val in node_values:
                    if hasattr(val, "nodes"):
                        to_search.extend([(val, node_condition_name)])
                    elif val is not None:
                        node_value_name = get_value_name(val)
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
