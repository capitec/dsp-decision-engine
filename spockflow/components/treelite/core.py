from spockflow.nodes import VariableNode, creates_node
from enum import StrEnum, auto, IntEnum
import treelite
import typing
from uuid import uuid4
import numpy as np
import pandas as pd
from itertools import chain
from typing_extensions import Annotated
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator

if typing.TYPE_CHECKING:
    from treelite.model_builder import ModelBuilder


@dataclass
class Position:
    x: float
    y: float


class PositionedNode(BaseModel):
    position: typing.Optional[Position]


class TestNode(PositionedNode):
    split_feature_id: int
    default_left: bool = False
    children: typing.List[str]


class RangeTestNode(TestNode):
    NODE_TYPE: typing.ClassVar[str] = "numerical_range_test_node"
    node_type: typing.Literal["numerical_range_test_node"] = NODE_TYPE
    thresholds: typing.List[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_thresholds(self) -> typing.Self:
        if len(self.thresholds) < 1:
            raise ValueError(f"Range test nodes should have at least one threshold")
        return self

    @model_validator(mode="after")
    def validate_children(self) -> typing.Self:
        # Ensure the number of children equals the number of thresholds + 1
        # Since N thresholds create N+1 ranges
        # We allow more children here but ignore them if present
        if len(self.children) < len(self.thresholds) + 1:
            raise ValueError(
                f"Range test node must have exactly {len(self.thresholds) + 1} child outputs "
                f"for {len(self.thresholds)} thresholds"
            )
        return self

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: typing.Dict[str, int],
        **kwargs,
    ):
        self._build_balanced_tree(
            builder,
            split_node_id=node_id,
            node_lookup=node_lookup,
            thresholds=self.thresholds,
            children=self.children[: len(self.thresholds) + 1],
        )

    def _build_balanced_tree(
        self,
        builder: "ModelBuilder",
        split_node_id: str,
        node_lookup: typing.Dict[str, int],
        thresholds: typing.List[float],
        children: typing.List[str],
    ):
        if not len(thresholds):
            return
        # Update the node lookup
        left_node_key = f"{split_node_id}_L"
        if left_node_key in node_lookup:
            left_node_key = f"{left_node_key}_{uuid4()}"
        right_node_key = f"{split_node_id}_R"
        if right_node_key in node_lookup:
            right_node_key = f"{right_node_key}_{uuid4()}"
        node_lookup[left_node_key] = len(node_lookup)
        node_lookup[right_node_key] = len(node_lookup)
        # Perform splits down the middle
        mid_index = len(thresholds) // 2
        builder.start_node(node_lookup[split_node_id])
        builder.numerical_test(
            feature_id=self.split_feature_id,
            threshold=thresholds[mid_index],
            default_left=self.default_left,
            opname="<",
            left_child_key=(
                node_lookup[children[0]]
                if len(thresholds) == 1
                else node_lookup[left_node_key]
            ),
            # We do lt or eq 2 here because if there are 2 thresholds [x,y]
            # Then this will be <y and the next two calls will be [x], [] so the left branch will handle
            # The last split and the right branch will directly point to the child
            right_child_key=(
                node_lookup[children[len(children) - 1]]
                if len(thresholds) <= 2
                else node_lookup[right_node_key]
            ),
        )
        builder.end_node()
        self._build_balanced_tree(
            builder,
            split_node_id=left_node_key,
            node_lookup=node_lookup,
            thresholds=thresholds[:mid_index],
            children=children[: mid_index + 1],  # We not leaving a child behind
        )
        self._build_balanced_tree(
            builder,
            split_node_id=right_node_key,
            node_lookup=node_lookup,
            thresholds=thresholds[mid_index + 1 :],
            children=children[mid_index + 1 :],
        )


class NumericalTestNode(TestNode):
    NODE_TYPE: typing.ClassVar[str] = "numerical_test_node"
    node_type: typing.Literal["numerical_test_node"] = NODE_TYPE
    threshold: float
    comparison_op: typing.Literal["<=", "<", "==", ">", ">="]

    @model_validator(mode="after")
    def validate_two_children(self) -> typing.Self:
        if len(self.children) < 2:
            raise ValueError("Numerical nodes must have at least 2 child outputs")
        return self

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: typing.Dict[str, int],
        **kwargs,
    ):
        builder.start_node(node_lookup[node_id])
        builder.numerical_test(
            feature_id=self.split_feature_id,
            threshold=self.threshold,
            default_left=self.default_left,
            opname=self.comparison_op,
            left_child_key=node_lookup[self.children[0]],
            right_child_key=node_lookup[self.children[1]],
        )
        builder.end_node()


class CategoricalTestNode(TestNode):
    NODE_TYPE: typing.ClassVar[str] = "categorical_test_node"
    node_type: typing.Literal["categorical_test_node"] = NODE_TYPE
    category_list: typing.List[int]
    # Basically if the value should be in or not in set
    category_list_right_child: bool

    @model_validator(mode="after")
    def validate_two_children(self) -> typing.Self:
        if len(self.children) < 2:
            raise ValueError("Categorical nodes must have at least 2 child outputs")
        return self

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: typing.Dict[str, int],
        **kwargs,
    ):
        builder.start_node(node_lookup[node_id])
        builder.categorical_test(
            feature_id=self.split_feature_id,
            default_left=self.default_left,
            left_child_key=node_lookup[self.children[0]],
            right_child_key=node_lookup[self.children[1]],
            category_list=self.category_list,
            category_list_right_child=self.category_list_right_child,
        )
        builder.end_node()


class OutputEncoding(StrEnum):
    ONE_HOT = auto()
    INDEX = auto()


def _one_hot(n_classes: int, class_index: int):
    return [int(i - 1 == class_index) for i in range(n_classes)]


class LeafNode(PositionedNode):
    DEFAULT_LEAF_VALUE: typing.ClassVar[str] = -1
    NODE_TYPE: typing.ClassVar[str] = "leaf"
    node_type: typing.Literal["leaf"] = NODE_TYPE
    leaf_value: int

    def build(
        self,
        builder: "ModelBuilder",
        node_id: str,
        node_lookup: typing.Dict[str, int],
        output_encoding: OutputEncoding,
        tree: "Tree",
    ):
        builder.start_node(node_lookup[node_id])
        builder.leaf(
            [node_lookup[node_id]]
            if output_encoding == OutputEncoding.INDEX
            else _one_hot(len(tree.tree_output.data), self.leaf_value)
        )
        builder.end_node()


class TreeOutput(BaseModel):
    columns: typing.List[str]
    data: typing.List[typing.List[str]]
    dtype: typing.Optional[typing.List[str]] = None
    # TODO add assertations here that
    # len(columns) == len(dtypes) == len(data[*])


class TreeMetadata(BaseModel):
    name: str
    description: str


NodeType = Annotated[
    typing.Union[RangeTestNode, NumericalTestNode, CategoricalTestNode, LeafNode],
    Field(discriminator="node_type"),
]

_L = typing.TypeVar("_L", bound=int)  # Leaf Nodes
_C = typing.TypeVar("_C", bound=int)  # Condition nodes
_B = typing.TypeVar("_B", bound=int)  # Bach
_T = typing.TypeVar("_T", bound=int)  # Subtrees
_F = typing.TypeVar("_F", bound=int)  # Features


@dataclass
class CompiledTreeliteTree:
    model: "treelite.Model"
    output_priority_mapping: np.ndarray[
        typing.Tuple[_L], np.dtype[np.integer[typing.Any]]
    ]
    leaf_output_mapping: np.ndarray[typing.Tuple[_L], np.dtype[np.integer[typing.Any]]]
    paths_matrix: np.ndarray[typing.Tuple[_C, _L], np.dtype[np.integer[typing.Any]]]
    output_dataframe_mapping: pd.DataFrame
    features: typing.List[str]
    node_id_mapping: typing.Dict[str, int]
    tree_path_labels: typing.List[str]

    @staticmethod
    def _get_node_id_mapping(nodes: typing.Dict[str, NodeType]):
        leaf_nodes = list(
            filter(lambda k: nodes[k].node_type == LeafNode.NODE_TYPE, nodes.keys())
        )
        return (
            {
                # We want to keep leaf nodes at the start of the mapping so we can compress the outputs
                # Dont have to differentiate between leaf_node_id and node_id.
                k: i
                for i, k in enumerate(
                    chain(
                        leaf_nodes,
                        filter(
                            lambda k: nodes[k].node_type != LeafNode.NODE_TYPE,
                            nodes.keys(),
                        ),
                    )
                )
            },
            leaf_nodes,
        )

    @staticmethod
    def _identify_independent_tree_roots(nodes: typing.Dict[str, NodeType]):
        root_nodes = set(nodes.keys())
        for n in nodes.values():
            if n.node_type == LeafNode.NODE_TYPE:
                continue
            root_nodes.difference_update(n.children)
        return root_nodes

    @staticmethod
    def _get_treelite_model_builder(
        root_nodes: typing.List[str],
        tree: "Tree",
        output_encoding: OutputEncoding = OutputEncoding.INDEX,
    ) -> "ModelBuilder":
        from treelite.model_builder import (
            Metadata,
            ModelBuilder,
            PostProcessorFunc,
            TreeAnnotation,
        )

        no_subtrees = len(root_nodes)
        no_outputs = (
            len(tree.tree_output.data)
            if output_encoding == OutputEncoding.ONE_HOT
            else 1
        )
        return ModelBuilder(
            threshold_type="float32",
            leaf_output_type="float32",
            metadata=Metadata(
                num_feature=len(tree.features),
                task_type="kMultiClf",
                average_tree_output=output_encoding == OutputEncoding.ONE_HOT,
                num_target=no_subtrees,  # We crete one target output per tree
                num_class=[no_outputs] * no_subtrees,
                leaf_vector_shape=(1, no_outputs),
            ),
            tree_annotation=TreeAnnotation(
                num_tree=no_subtrees,
                # Trees are independent and only contribute to their target
                target_id=list(range(no_subtrees)),
                class_id=([-1] * no_subtrees),
            ),
            postprocessor=PostProcessorFunc(name="identity"),
            base_scores=[0.0] * (no_subtrees * no_outputs),
        )

    @classmethod
    def _build_treelite_tree(
        cls,
        root_nodes: typing.List[str],
        tree: "Tree",
        node_id_mapping: typing.Dict[str, int],
        output_encoding: OutputEncoding = OutputEncoding.INDEX,
    ) -> "ModelBuilder":
        builder = cls._get_treelite_model_builder(root_nodes, tree, output_encoding)

        for root in cls._sort_root_nodes(node_id_mapping, root_nodes, tree):
            builder.start_tree()
            to_search = [root]
            seen = set()
            while to_search:
                n_key = to_search.pop()
                if n_key in seen:
                    raise ValueError(
                        f"Cannot build tree as it contains loops detected on node_id: {n_key}"
                    )
                seen.add(n_key)
                n = tree.nodes[n_key]
                n.build(
                    builder,
                    n_key,
                    node_id_mapping,
                    output_encoding=output_encoding,
                    tree=tree,
                )
                if n.node_type != LeafNode.NODE_TYPE:
                    to_search.extend(n.children)
            builder.end_tree()
        return builder

    @staticmethod
    def _sort_root_nodes(
        node_id_mapping: typing.Dict[str, int],
        root_nodes: typing.Set[str],
        tree: "Tree",
    ) -> typing.List[str]:
        # Might want to extend this later to take into account the leaf priority
        return sorted(root_nodes, key=node_id_mapping.get)

    @classmethod
    def build(cls, tree: "Tree") -> "typing.Self":
        if len(tree.nodes) <= 0:
            raise ValueError("Tree contains no nodes")
        node_id_mapping, leaf_nodes = cls._get_node_id_mapping(tree.nodes)
        root_nodes = cls._identify_independent_tree_roots(tree.nodes)
        if len(root_nodes) <= 0:
            # We know that the tree contains nodes so only way for no root is a loop
            raise ValueError("Cannot build tree as it contains loops.")
        builder = cls._build_treelite_tree(
            root_nodes=root_nodes,
            tree=tree,
            node_id_mapping=node_id_mapping,
            output_encoding=OutputEncoding.INDEX,
        )

        # Priority (TODO move to function)
        num_leaf_nodes = len(leaf_nodes)
        output_priority_mapping = np.repeat(-1, num_leaf_nodes).astype(np.int64)
        for leaf_node in leaf_nodes:
            leaf_node_idx = node_id_mapping[leaf_node]
            if tree.nodes[leaf_node].leaf_value == LeafNode.DEFAULT_LEAF_VALUE:
                # Default leafs should be given least priority
                output_priority_mapping[leaf_node_idx] = -2
                continue

            try:
                # Give greatest priority to largest leaf
                output_priority_mapping[leaf_node_idx] = (
                    num_leaf_nodes - tree.leaf_order.index(leaf_node)
                )
            except ValueError as e:
                raise ValueError(f"Could not find {leaf_node} in leaf order") from e
        # Output Mapping (TODO move to function)
        num_leaf_nodes = len(leaf_nodes)
        output_mapping = np.repeat(-1, num_leaf_nodes).astype(np.int64)
        for leaf_node_key in leaf_nodes:
            leaf_node_idx = node_id_mapping[leaf_node_key]
            output_mapping[leaf_node_idx] = tree.nodes[leaf_node_key].leaf_value
        # Output Dataframe (TODO move to function)
        try:
            output_df_map = pd.DataFrame(
                data=tree.tree_output.data,
                columns=tree.tree_output.columns,
                dtype=tree.tree_output.dtype,
                index=range(-1, len(tree.tree_output.data) - 1, 1),
            )
        except ValueError as e:
            raise ValueError("Could not create output dataframe") from e
        diff_set = set(output_mapping) - set(output_df_map.index)
        if diff_set:
            raise ValueError(
                f"Found output values with no matching items in output df: {diff_set}"
            )

        # Path Maps (TODO move to function)
        num_leaf_nodes = len(leaf_nodes)
        num_condition_nodes = len(node_id_mapping) - num_leaf_nodes
        paths_matrix = -np.ones((num_leaf_nodes, num_condition_nodes), dtype=np.int8)
        tree_path_labels = ["" for i in range(num_condition_nodes)]

        def dfs(node_key: str, path_vector: np.ndarray):
            node = tree.nodes[node_key]
            node_idx = node_id_mapping[node_key]
            if node.node_type == LeafNode.NODE_TYPE:
                if not (paths_matrix[node_idx] == -1).all():
                    raise ValueError(f"Found multiple paths to leaf node {node_key}")
                paths_matrix[node_idx] = path_vector
            else:
                tree_path_labels[node_idx - num_leaf_nodes] = node_key
                for child_idx, child in enumerate(node.children):
                    path_vector[node_idx - num_leaf_nodes] = child_idx
                    dfs(child, path_vector)
                path_vector[node_idx - num_leaf_nodes] = -1

        for root in root_nodes:
            path_vector = -np.ones((num_condition_nodes,), dtype=np.int8)
            dfs(root, path_vector)

        return CompiledTreeliteTree(
            model=builder.commit(),
            output_priority_mapping=output_priority_mapping,
            leaf_output_mapping=output_mapping,
            output_dataframe_mapping=output_df_map,
            node_id_mapping=node_id_mapping,
            features=tree.features,
            paths_matrix=paths_matrix,
            tree_path_labels=tree_path_labels,
        )

    def _get_inputs(self, function: typing.Callable):
        return {f: typing.Union[np.ndarray, pd.Series] for f in self.features}

    @creates_node(kwarg_input_generator="_get_inputs")
    def formatted_inputs(
        self, **kwargs: typing.Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        return np.stack([kwargs[f] for f in self.features], axis=-1)

    @creates_node(kwarg_input_generator="_get_inputs")
    def index(self, **kwargs: typing.Union[np.ndarray, pd.Series]) -> pd.Index:
        index_length = 0
        for f_name in self.features:
            f = kwargs[f_name]
            if isinstance(f, pd.Series):
                return f.index
            index_length = max(index_length, len(f))
        return pd.RangeIndex(0, index_length)

    @creates_node()
    def tree_results(
        self,
        formatted_inputs: np.ndarray[
            typing.Tuple[_B, _F], np.dtype[np.number[typing.Any]]
        ],
    ) -> np.ndarray[typing.Tuple[_B, _T], np.dtype[np.integer[typing.Any]]]:
        return treelite.gtil.predict(self.model, formatted_inputs)[:, :, 0].astype(int)

    @creates_node()
    def highest_priority_index(
        self,
        tree_results: np.ndarray[
            typing.Tuple[_B, _T], np.dtype[np.integer[typing.Any]]
        ],
    ) -> np.ndarray[typing.Tuple[_B], np.dtype[np.integer[typing.Any]]]:
        priorities = self.output_priority_mapping[tree_results]
        return priorities.argmax(axis=1)

    @creates_node()
    def prioritized_outputs(
        self,
        tree_results: np.ndarray[
            typing.Tuple[_B, _T], np.dtype[np.integer[typing.Any]]
        ],
        highest_priority_index: np.ndarray[
            typing.Tuple[_B], np.dtype[np.integer[typing.Any]]
        ],
    ) -> np.ndarray[typing.Tuple[_B], np.dtype[np.integer[typing.Any]]]:
        result_idx = tree_results[
            np.arange(len(highest_priority_index)), highest_priority_index
        ]
        return self.leaf_output_mapping[result_idx]

    @creates_node(is_namespaced=False)
    def get_output(
        self,
        index: pd.Index,
        prioritized_outputs: np.ndarray[
            typing.Tuple[_B], np.dtype[np.integer[typing.Any]]
        ],
    ) -> pd.DataFrame:
        return self.output_dataframe_mapping.loc[prioritized_outputs].set_index(
            index, drop=True
        )

    @creates_node()
    def all_tree_paths(
        self,
        tree_results: np.ndarray[
            typing.Tuple[_B, _T], np.dtype[np.integer[typing.Any]]
        ],
    ) -> np.ndarray[typing.Tuple[_B, _T, _C], np.dtype[np.integer[typing.Any]]]:
        return self.paths_matrix[tree_results]

    @creates_node()
    def prioritized_tree_paths(
        self,
        all_tree_paths: np.ndarray[
            typing.Tuple[_B, _T, _C], np.dtype[np.integer[typing.Any]]
        ],
        highest_priority_index: np.ndarray[
            typing.Tuple[_B], np.dtype[np.integer[typing.Any]]
        ],
    ) -> np.ndarray[typing.Tuple[_B, _C], np.dtype[np.integer[typing.Any]]]:
        return all_tree_paths[
            np.arange(len(highest_priority_index)), highest_priority_index
        ]

    @creates_node()
    def tree_path_keys(self) -> typing.List[str]:
        return self.tree_path_labels


class Tree(VariableNode):
    features: typing.List[str]
    nodes: typing.Dict[str, NodeType]
    leaf_order: typing.List[str] = Field(alias="leafOrder")
    metadata: TreeMetadata
    tree_output: TreeOutput = Field(alias="treeOutput")

    def compile(self):
        return CompiledTreeliteTree.build(self)
