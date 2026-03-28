from spockflow.nodes import creates_node
from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    # For Python < 3.11 compatibility
    class StrEnum(str, Enum):
        pass


import treelite
import typing as t
import typing_extensions as t_ext
import numpy as np
import pandas as pd
from itertools import chain
from dataclasses import dataclass
import re
from .compiled_node_types import CompiledNodeType, OutputEncoding, LeafNode
from logging import getLogger
from uuid import uuid4
from .edge import Edge, EdgeData

if t.TYPE_CHECKING:
    from treelite.model_builder import ModelBuilder
    from .tree import Tree


class PreprocFeatureMap(t.NamedTuple):
    old_name: str
    new_name: str


logger = getLogger(__name__)
_L = t.TypeVar("_L", bound=int)  # Leaf Nodes
_C = t.TypeVar("_C", bound=int)  # Condition nodes
_B = t.TypeVar("_B", bound=int)  # Bach
_T = t.TypeVar("_T", bound=int)  # Subtrees
_F = t.TypeVar("_F", bound=int)  # Features


@dataclass
class CompiledTreeliteTree:
    model: "treelite.Model"
    output_priority_mapping: np.ndarray[t.Tuple[_L], np.dtype[np.integer[t.Any]]]
    leaf_output_mapping: np.ndarray[t.Tuple[_L], np.dtype[np.integer[t.Any]]]
    output_dataframe_mapping: pd.DataFrame
    features: t.List[str]
    root_nodes: "np.ndarray[t.Tuple[_T], np.dtype[np.bytes_]]"
    # Useful to determine the path taken
    nodes: t.Dict[str, CompiledNodeType]
    node_id_mapping: t.Dict[str, int]
    node_condition_names: "np.ndarray[t.Tuple[_C], np.dtype[np.bytes_]]"
    node_children: t.Dict[str, t.List[str]]
    leaf_root_node_mapping: np.ndarray[t.Tuple[_L], np.dtype[np.integer[t.Any]]]
    paths_matrix: np.ndarray[t.Tuple[_C, _L], np.dtype[np.integer[t.Any]]]
    # Used for preprocessing
    preproc_nodes: t.Dict[str, PreprocFeatureMap]
    preproc_features: t.List[str]
    node_feature_names: t.Dict[str, str]
    # Variables
    variable_input_name: t.Optional[str] = None

    def _get_inputs(self, function: t.Callable):
        inputs = {f: t.Union[np.ndarray, pd.Series] for f in self.features}
        if self.variable_input_name is not None:
            inputs[self.variable_input_name] = t.Dict[str, t.Any]
        return inputs

    @creates_node(kwarg_input_generator="_get_inputs")
    def preprocessed_inputs(
        self, **kwargs: t.Union[np.ndarray, pd.Series, t.Dict[str, t.Any]]
    ) -> t.Dict[str, t.Union[np.ndarray, pd.Series]]:
        features = {}
        variables = kwargs.get(self.variable_input_name, {})
        for n_id, m in self.preproc_nodes.items():
            features[m.new_name] = self.nodes[n_id].preprocess_inputs(
                kwargs[m.old_name], variables=variables
            )
        return {**kwargs, **features}

    @creates_node()
    def formatted_inputs(
        self, preprocessed_inputs: t.Dict[str, t.Union[np.ndarray, pd.Series]]
    ) -> np.ndarray[t.Tuple[_B, _F], np.dtype[np.number[t.Any]]]:
        # TODO wondering if its best to convert to float here too?
        if len(self.preproc_features) <= 0:
            if len(self.features) <= 0:
                return np.zeros((0, 0), dtype=np.float32)
            else:
                return np.zeros(
                    (len(preprocessed_inputs[self.features[0]]), 0), dtype=np.float32
                )
        else:
            return np.stack(
                [preprocessed_inputs[f] for f in self.preproc_features], axis=-1
            )

    @creates_node(kwarg_input_generator="_get_inputs")
    def index(self, **kwargs: t.Union[np.ndarray, pd.Series]) -> pd.Index:
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
        formatted_inputs: np.ndarray[t.Tuple[_B, _F], np.dtype[np.number[t.Any]]],
    ) -> np.ndarray[t.Tuple[_B, _T], np.dtype[np.integer[t.Any]]]:
        return treelite.gtil.predict(self.model, formatted_inputs)[:, :, 0].astype(int)

    @creates_node()
    def highest_priority_index(
        self,
        tree_results: np.ndarray[t.Tuple[_B, _T], np.dtype[np.integer[t.Any]]],
    ) -> np.ndarray[t.Tuple[_B], np.dtype[np.integer[t.Any]]]:
        priorities = self.output_priority_mapping[tree_results]
        return priorities.argmax(axis=1)

    @creates_node()
    def prioritized_outputs(
        self,
        tree_results: np.ndarray[t.Tuple[_B, _T], np.dtype[np.integer[t.Any]]],
        highest_priority_index: np.ndarray[t.Tuple[_B], np.dtype[np.integer[t.Any]]],
    ) -> np.ndarray[t.Tuple[_B], np.dtype[np.integer[t.Any]]]:
        result_idx = tree_results[
            np.arange(len(highest_priority_index)), highest_priority_index
        ]
        return self.leaf_output_mapping[result_idx]

    @creates_node(is_namespaced=False)
    def get_output(
        self,
        index: pd.Index,
        prioritized_outputs: np.ndarray[t.Tuple[_B], np.dtype[np.integer[t.Any]]],
    ) -> pd.DataFrame:
        return self.output_dataframe_mapping.loc[prioritized_outputs].set_index(
            index, drop=True
        )

    @creates_node()
    def all_subtree_output(
        self,
        index: pd.Index,
        tree_results: np.ndarray[t.Tuple[_B, _T], np.dtype[np.integer[t.Any]]],
    ) -> pd.DataFrame:
        idx = self.leaf_output_mapping[tree_results.flatten()]
        original_index = np.asarray(index)
        n = len(self.root_nodes)
        level_0 = np.repeat(original_index, n)
        level_1 = self.root_nodes[np.tile(np.arange(n), len(original_index))]

        return self.output_dataframe_mapping.loc[idx].set_index(
            pd.MultiIndex.from_arrays(
                [level_0, level_1], names=[index.name, "subtree"]
            ),
            drop=True,
        )

    @creates_node()
    def all_tree_paths(
        self,
        tree_results: np.ndarray[t.Tuple[_B, _T], np.dtype[np.integer[t.Any]]],
    ) -> np.ndarray[t.Tuple[_B, _T, _C], np.dtype[np.integer[t.Any]]]:
        return self.paths_matrix[tree_results]

    @creates_node()
    def all_tree_paths_flat(
        self,
        index: pd.Index,
        all_tree_paths: np.ndarray[t.Tuple[_B, _T, _C], np.dtype[np.integer[t.Any]]],
    ) -> pd.Series:
        original_index = np.asarray(index)
        t = len(self.root_nodes)
        c = all_tree_paths.shape[2]
        level_b = np.repeat(original_index, t * c)
        level_t = self.root_nodes[
            np.tile(np.repeat(np.arange(t), c), len(original_index))
        ]
        level_c = self.node_condition_names[
            np.tile(np.tile(np.arange(c), t), len(original_index))
        ]
        return pd.Series(
            all_tree_paths.flat,
            # TODO might be good to try support multi-index for the origional index
            index=pd.MultiIndex.from_arrays(
                [level_b, level_t, level_c], names=[index.name, "subtree", "condition"]
            ),
            name="all_paths",
        )

    @creates_node()
    def prioritized_tree_paths(
        self,
        all_tree_paths: np.ndarray[t.Tuple[_B, _T, _C], np.dtype[np.integer[t.Any]]],
        highest_priority_index: np.ndarray[t.Tuple[_B], np.dtype[np.integer[t.Any]]],
    ) -> np.ndarray[t.Tuple[_B, _C], np.dtype[np.integer[t.Any]]]:
        return all_tree_paths[
            np.arange(len(highest_priority_index)), highest_priority_index
        ]

    @creates_node()
    def prioritized_tree_paths_flat(
        self,
        index: pd.Index,
        prioritized_tree_paths: np.ndarray[
            t.Tuple[_B, _C], np.dtype[np.integer[t.Any]]
        ],
        highest_priority_index: np.ndarray[t.Tuple[_B], np.dtype[np.integer[t.Any]]],
    ) -> pd.Series:
        original_index = np.asarray(index)
        c = prioritized_tree_paths.shape[1]
        level_b = np.repeat(original_index, c)
        # Not really needed but lets us know by the index which subtree was selected
        level_t = np.repeat(self.root_nodes[highest_priority_index], c)
        level_c = self.node_condition_names[np.tile(np.arange(c), len(original_index))]
        return pd.Series(
            prioritized_tree_paths.flat,
            index=pd.MultiIndex.from_arrays(
                [level_b, level_t, level_c], names=[index.name, "subtree", "condition"]
            ),
            name="prioritized_paths",
        )

    @creates_node()
    def compiled_tree(self) -> "t_ext.Self":
        # Useful to create custom outputs of node mappings
        return self
