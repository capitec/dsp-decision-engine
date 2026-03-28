import typing as t
import typing_extensions as t_ext
from logging import getLogger
from itertools import chain

# from functools import cmp_to_key
from uuid import uuid4

import numpy as np

from .compiled_tree import CompiledTreeliteTree, PreprocFeatureMap
from .edge import Edge, EdgeData, MultiSourceEdge
from .compiled_node_types import CompiledNodeType, OutputEncoding, LeafNode
from .node_types import NodeData

if t.TYPE_CHECKING:
    from treelite.model_builder import ModelBuilder
    from .tree import Tree, SubTree

logger = getLogger(__name__)


def _build_child_mapping(edges: t.List[Edge]) -> t.Dict[str, t.Dict[int, str]]:
    """Build child node mapping from edges array."""
    child_map = {}
    for edge in edges:
        source = edge.source
        target = edge.target
        source_indexes = edge.data.sourceIndex
        if not isinstance(source_indexes, list):
            source_indexes = [source_indexes]
        for source_index in source_indexes:
            if source not in child_map:
                child_map[source] = {}
            if source_index in child_map[source]:
                if child_map[source][source_index] == target:
                    logger.warning(
                        f"Duplicate edge found {source}:{source_index} -> {target}"
                    )
                else:
                    raise ValueError(
                        f"Conflicting edge found {source}:{source_index} -> {target} != {child_map[source][source_index]}"
                    )
            child_map[source][source_index] = target
    return child_map


def _build_simple_child_mapping(
    edges: t.List[t.Union[Edge, MultiSourceEdge]],
) -> t.Dict[str, t.List[str]]:
    """Build simple parent->children mapping from any edge type for subtree traversal."""
    child_map = {}
    for edge in edges:
        source = edge.source
        target = edge.target

        if source not in child_map:
            child_map[source] = []
        if target not in child_map[source]:
            child_map[source].append(target)
    return child_map


def _find_missing_edges(
    node_id: str,
    compiled_node: "CompiledNodeType",
    edges: t.List[Edge],
) -> t.Set[int]:
    """Find missing edge indices for a decision node."""
    # we build this here as after adding edges it will have to be rebuilt
    child_map = _build_child_mapping(edges)

    # Determine expected number of branches based on node type
    expected_branches = compiled_node.child_count

    # Find which indices are missing
    existing_indices = set(child_map.get(node_id, {}).keys())
    expected_indices = set(range(expected_branches))
    extra_edges = existing_indices - expected_indices
    if len(extra_edges):
        logger.warning(
            f"Found additional edges {extra_edges} from node {node_id} which will be ignored!"
        )
    return expected_indices - existing_indices


class CompiledNodeEdge(t.NamedTuple):
    nodes: t.Dict[str, "CompiledNodeType"]
    node_name_mapping: t.Dict[str, str]
    edges: t.List[Edge]


# def _duplicate_subtree(
#     root_id: str,
#     nodes: t.Dict[str, "NodeData"],
#     child_map: t.Dict[str, t.List[str]],
#     suffix: str,
#     existing_nodes: t.Dict[str, "NodeData"],
# ) -> t.Tuple[t.Dict[str, "NodeData"], t.List[Edge]]:
#     """Duplicate a subtree starting from root_id with new IDs using suffix.
#     Handles node ID conflicts by using generated_ prefix."""
#     duplicated_nodes = {}
#     duplicated_edges = []
#     node_mapping = {}  # old_id -> new_id

#     # BFS to collect all nodes in subtree
#     visited = set()
#     queue = [root_id]

#     while queue:
#         current_id = queue.pop(0)
#         if current_id in visited:
#             continue
#         visited.add(current_id)

#         # Duplicate the node
#         if current_id in nodes:
#             new_id = f"{current_id}_{suffix}"

#             # Handle conflicts by generating new UUID-based ID
#             if new_id in existing_nodes:
#                 new_id = f"generated_{str(uuid4())}"

#             node_mapping[current_id] = new_id
#             duplicated_nodes[new_id] = nodes[current_id].model_copy(deep=True)

#         # Add children to queue
#         if current_id in child_map:
#             for child_id in child_map[current_id]:
#                 if child_id not in visited:
#                     queue.append(child_id)

#     # Create edges within the duplicated subtree
#     for parent_id in visited:
#         if parent_id in child_map and parent_id in node_mapping:
#             for i, child_id in enumerate(child_map[parent_id]):
#                 if child_id in node_mapping:
#                     new_edge = Edge(
#                         id=str(uuid4()),
#                         source=node_mapping[parent_id],
#                         target=node_mapping[child_id],
#                         data=EdgeData(sourceIndex=i),
#                     )
#                     duplicated_edges.append(new_edge)

#     return duplicated_nodes, duplicated_edges


# def _duplicate_multi_source_edge_subtrees(
#     nodes: t.Dict[str, "NodeData"],
#     edges: t.List[MultiSourceEdge],
# ) -> t.Tuple[t.Dict[str, "NodeData"], t.List[Edge]]:
#     """For each multi-source edge, duplicate the target subtree for each source index."""
#     # Build simple child map for subtree traversal
#     child_map = _build_simple_child_mapping(edges)

#     new_edges: t.List[Edge] = []
#     new_nodes: t.Dict[str, "NodeData"] = {}

#     # Process each multi-source edge
#     for multi_edge in edges:
#         target_id = multi_edge.target
#         source_indices = multi_edge.data.sourceIndex

#         # If only one source index, convert to regular Edge and keep original nodes
#         if len(source_indices) == 1:
#             new_edge = Edge(
#                 id=str(uuid4()),
#                 source=multi_edge.source,
#                 target=target_id,
#                 data=EdgeData(sourceIndex=source_indices[0]),
#             )
#             new_edges.append(new_edge)
#             continue

#         # For multiple source indices, duplicate the target subtree for each
#         for i, source_index in enumerate(source_indices):
#             if i == 0:
#                 # First index uses original nodes (no suffix)
#                 new_edge = Edge(
#                     id=str(uuid4()),
#                     source=multi_edge.source,
#                     target=target_id,
#                     data=EdgeData(sourceIndex=source_index),
#                 )
#                 new_edges.append(new_edge)
#             else:
#                 # Subsequent indices get duplicated subtrees
#                 suffix = f"ms{i}_{source_index}"

#                 # Duplicate the subtree starting from target
#                 duplicated_nodes, duplicated_edges = _duplicate_subtree(
#                     target_id, nodes, child_map, suffix, new_nodes
#                 )

#                 # Add all duplicated nodes
#                 new_nodes.update(duplicated_nodes)

#                 # Add internal edges within duplicated subtree
#                 new_edges.extend(duplicated_edges)

#                 # Create the new single edge from source to duplicated target
#                 duplicated_target_id = f"{target_id}_{suffix}"
#                 if duplicated_target_id in duplicated_nodes:
#                     new_edge = Edge(
#                         id=str(uuid4()),
#                         source=multi_edge.source,
#                         target=duplicated_target_id,
#                         data=EdgeData(sourceIndex=source_index),
#                     )
#                     new_edges.append(new_edge)

#     return new_nodes, new_edges


# def _normalize_multi_source_edge_subtrees(
# Would be nice if we could get away with duping edges but we cannot
#         """
#         if ret != 0:
# >           raise TreeliteError(_LIB.TreeliteGetLastError().decode("utf-8"))
# E           treelite.core.TreeliteError: [11:21:58] /Users/runner/work/treelite/treelite/src/model_builder/model_builder.cc:179: Check failed: left_child_key != right_child_key (3 vs. 3) : Left and child nodes must be unique
#         """
#         if ret != 0:
# >           raise TreeliteError(_LIB.TreeliteGetLastError().decode("utf-8"))
# E           treelite.core.TreeliteError: [11:21:58] /Users/runner/work/treelite/treelite/src/model_builder/model_builder.cc:179: Check failed: left_child_key != right_child_key (3 vs. 3) : Left and child nodes must be unique
#     nodes: t.Dict[str, "NodeData"],
#     edges: t.List[MultiSourceEdge],
# ) -> t.Tuple[t.Dict[str, "NodeData"], t.Dict[str,str], t.List[Edge]]:
#     new_edges: t.List[Edge] = []
#     new_nodes: t.Dict[str, "NodeData"] = {**nodes}
#     # The node name mapping helps us with the path to remap duplicated paths back to the original names
#     node_name_mapping: t.Dict[str, str] = {n:n for n in nodes.keys()}

#     # subtrees_to_duplicate: t.List[t.Tuple[str, str]] = []

#     for edge in edges:
#         if not edge.data.sourceIndex:
#             continue
#         new_edges.append(
#             Edge.model_validate(
#                 {
#                     **edge.model_dump(exclude={"data"}),
#                     "data": {"sourceIndex": edge.data.sourceIndex[0]},
#                 }
#             )
#         )
#         for source_idx in edge.data.sourceIndex[1:]:
#             gen_prefix = f"g{uuid4()}_"
#             new_edges.append(
#                 Edge.model_validate(
#                     {
#                         **edge.model_dump(exclude={"data", "target"}),
#                         "target": edge.target,
#                         "data": {"sourceIndex": source_idx},
#                     }
#                 )
#             )
#     # node_edges = {n: [e for e in new_edges if e.source == n] for n in new_nodes.keys()}

#     # while subtrees_to_duplicate:
#     #     node_id, gen_prefix = subtrees_to_duplicate.pop(0)
#     #     new_nodes[f"{gen_prefix}{node_id}"] = new_nodes[node_id]
#     #     # Do the lookup so duplicates of duplicates get mapped too
#     #     node_name_mapping[f"{gen_prefix}{node_id}"] = node_name_mapping[node_id]

#     #     for e in node_edges[node_id]:
#     #         new_edges.append(
#     #             e.model_copy(
#     #                 update={
#     #                     "source": f"{gen_prefix}{node_id}",
#     #                     "target": f"{gen_prefix}{e.target}",
#     #                 }
#     #             )
#     #         )
#     #         subtrees_to_duplicate.append((e.target, gen_prefix))
#     return new_nodes, node_name_mapping, new_edges


def _normalize_multi_source_edge_subtrees(
    nodes: t.Dict[str, "NodeData"],
    edges: t.List[MultiSourceEdge],
) -> t.Tuple[t.Dict[str, "NodeData"], t.Dict[str, str], t.List[Edge]]:
    new_edges: t.List[Edge] = []
    new_nodes: t.Dict[str, "NodeData"] = {**nodes}
    # The node name mapping helps us with the path to remap duplicated paths back to the origional names
    node_name_mapping: t.Dict[str, str] = {n: n for n in nodes.keys()}

    subtrees_to_duplicate: t.List[t.Tuple[str, str]] = []

    def next_index():
        next_index._index += 1
        return next_index._index

    next_index._index = -1

    for edge in edges:
        if not edge.data.sourceIndex:
            continue
        new_edges.append(
            Edge.model_validate(
                {
                    **edge.model_dump(exclude={"data"}),
                    "data": {"sourceIndex": edge.data.sourceIndex[0]},
                }
            )
        )
        for source_idx in edge.data.sourceIndex[1:]:
            gen_postfix = f"#d{next_index()}"
            new_edges.append(
                Edge.model_validate(
                    {
                        **edge.model_dump(exclude={"data", "target"}),
                        "target": f"{edge.target}{gen_postfix}",
                        "data": {"sourceIndex": source_idx},
                    }
                )
            )
            subtrees_to_duplicate.append((edge.target, gen_postfix))
    node_edges = {n: [e for e in new_edges if e.source == n] for n in new_nodes.keys()}

    while subtrees_to_duplicate:
        node_id, gen_postfix = subtrees_to_duplicate.pop(0)
        new_node_id = f"{node_id}{gen_postfix}"
        new_nodes[new_node_id] = new_nodes[node_id]
        # Do the lookup so duplicates of duplicates get mapped too
        node_name_mapping[new_node_id] = node_name_mapping[node_id]

        new_node_edges = []
        for e in node_edges[node_id]:
            new_edge = e.model_copy(
                update={
                    "source": new_node_id,
                    "target": f"{e.target}{gen_postfix}",
                }
            )
            new_edges.append(new_edge)
            new_node_edges.append(new_edge)
            subtrees_to_duplicate.append((e.target, gen_postfix))
        node_edges[new_node_id] = new_node_edges
    return new_nodes, node_name_mapping, new_edges


def _compile_nodes(
    nodes: t.Dict[str, "NodeData"],
    edges: t.List[MultiSourceEdge],
    tree: "Tree",
    output_leaf_map: t.Dict[str, int],
) -> CompiledNodeEdge:
    """Create pseudo leaf nodes for missing edges."""
    # Find all missing edges first
    nodes, node_name_mapping, edges = _normalize_multi_source_edge_subtrees(
        nodes, edges
    )
    missing_edges_map = {}
    compiled_nodes = {}
    for node_id, node in nodes.items():
        compiled_node = node.compile(
            tree,
            output_leaf_map=output_leaf_map,
            node_id=node_id,
            node_name_mapping=node_name_mapping,
        )
        compiled_nodes[node_id] = compiled_node
        if not compiled_node.IS_LEAF:
            missing_indices = _find_missing_edges(node_id, compiled_node, edges)
            if missing_indices:
                missing_edges_map[node_id] = missing_indices

    # Create pseudo leaf nodes
    extra_nodes = {}
    extra_edges = []
    for parent_id, missing_indices in missing_edges_map.items():
        for index in missing_indices:
            pseudo_id = f"pseudo_{parent_id}_{index}"
            if pseudo_id in nodes:
                pseudo_id = str(uuid4())
            # Create a pseudo leaf node with default value
            extra_nodes[pseudo_id] = LeafNode(leaf_value=LeafNode.DEFAULT_LEAF_VALUE)
            output_leaf_map[pseudo_id] = LeafNode.DEFAULT_LEAF_VALUE
            extra_edges.append(
                Edge(
                    id=str(uuid4()),
                    source=parent_id,
                    target=pseudo_id,
                    data=EdgeData(sourceIndex=index),
                )
            )
    return compiled_nodes | extra_nodes, node_name_mapping, edges + extra_edges


@staticmethod
def _get_node_id_mapping(
    nodes: t.Dict[str, "CompiledNodeType"], node_weighting: t.Dict[str, int]
):
    leaf_nodes = list(filter(lambda k: nodes[k].IS_LEAF, nodes.keys()))
    # Sorting not really strictly needed but nice for paths as it puts the paths roughly in order
    other_nodes = sorted(
        filter(lambda k: not nodes[k].IS_LEAF, nodes.keys()),
        key=lambda k: node_weighting.get(k, float("inf")),
    )
    return (
        {
            # We want to keep leaf nodes at the start of the mapping so we can compress the outputs
            # Dont have to differentiate between leaf_node_id and node_id.
            k: i
            for i, k in enumerate(chain(leaf_nodes, other_nodes))
        },
        leaf_nodes,
    )


@staticmethod
def _identify_independent_tree_roots(
    nodes: t.Dict[str, "CompiledNodeType"], edges: t.List[Edge]
):
    root_nodes = set(nodes.keys())
    # Remove nodes that are targets of edges (they have parents)
    for edge in edges:
        target = edge.target
        root_nodes.discard(target)
    return root_nodes


@staticmethod
def _get_treelite_model_builder(
    root_nodes: t.List[str],
    num_outputs: int,
    num_features: int,
    output_encoding: OutputEncoding = OutputEncoding.INDEX,
) -> "ModelBuilder":
    from treelite.model_builder import (
        Metadata,
        ModelBuilder,
        PostProcessorFunc,
        TreeAnnotation,
    )

    no_subtrees = len(root_nodes)
    no_outputs = num_outputs if output_encoding == OutputEncoding.ONE_HOT else 1
    return ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
        metadata=Metadata(
            num_feature=num_features,
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


def _build_treelite_tree(
    nodes: t.Dict[str, CompiledNodeType],
    root_nodes: t.List[str],
    tree: "Tree",
    node_id_mapping: t.Dict[str, int],
    node_children: t.Dict[str, t.List[str]],
    node_feature_id_mapping: t.Dict[str, int],
    num_outputs: int,
    output_encoding: OutputEncoding = OutputEncoding.INDEX,
) -> "ModelBuilder":
    builder = _get_treelite_model_builder(
        root_nodes,
        num_outputs=num_outputs,
        num_features=len(set(node_feature_id_mapping.values())),
        output_encoding=output_encoding,
    )

    for root in root_nodes:
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
            n = nodes[n_key]
            n.build(
                builder,
                n_key,
                node_id_mapping,
                processed_feature_id=node_feature_id_mapping.get(
                    n_key
                ),  # Leafs dont have a feature so must use get
                node_children=node_children,
                output_encoding=output_encoding,
                tree=tree,
            )
            children = node_children.get(n_key, [])
            to_search.extend(children)
        builder.end_tree()
    return builder


# @staticmethod
# def _sort_root_nodes(
#     node_id_mapping: t.Dict[str, int],
#     root_nodes: t.Set[str],
#     tree: "Tree",
# ) -> t.List[str]:
#     # Sort by subtree priority if available
#     if tree.subtrees:
#         subtree_priority_map = {s["rootNodeId"]: s["priority"] for s in tree.subtrees}
#         return sorted(root_nodes, key=lambda x: subtree_priority_map.get(x, 999))
#     return sorted(root_nodes, key=node_id_mapping.get)

# @staticmethod
# def _find_leaf_subtree_priority(leaf_node_id: str, tree: "Tree") -> int:
#     """Find which subtree a leaf belongs to and return its priority."""
#     if not tree.subtrees:
#         return 0

#     # Find path from each subtree root to the leaf
#     for subtree in tree.subtrees:
#         root_id = subtree["rootNodeId"]
#         if CompiledTreeliteTree._is_leaf_in_subtree(leaf_node_id, root_id, tree):
#             # Higher priority number means lower actual priority
#             # Invert so lower priority number = higher priority
#             return len(tree.subtrees) - subtree["priority"]

#     # Default if not found in any subtree
#     return 0

# @staticmethod
# def get_subtree_leafs(root_id: str, tree: "Tree") -> bool:
#     """Return all the leaves of a subtree"""
#     leafs = set()
#     visited = set()
#     to_visit = [root_id]

#     while to_visit:
#         current = to_visit.pop()
#         if current in visited:
#             continue
#         visited.add(current)

#         # Get children using edge resolution
#         node = tree.nodes.get(current)
#         if node:
#             if node.IS_LEAF:
#                 leafs.add(current)
#             else:
#                 children = tree._resolve_node_children(node, current)
#                 to_visit.extend(children)
#     return leafs

# @staticmethod
# def _get_leaf_tree_mapping(tree: "Tree", root_nodes: t.List[str]):


@staticmethod
def determine_feature_mapping(
    nodes: t.Dict[str, CompiledNodeType],
    node_name_map: t.Dict[str, str],
    features: t.List[str],
):
    node_features = {}
    preproc_nodes: t.Dict[str, PreprocFeatureMap] = {}
    for n_id, n in nodes.items():
        if n.IS_LEAF:
            continue
        if n.preprocess_inputs is not None:
            n_id_orig = node_name_map[n_id]
            if n_id_orig in preproc_nodes:
                node_feature_name = preproc_nodes[n_id_orig].new_name
            else:
                node_feature_name = f"#preproc_{n_id_orig}"
                preproc_nodes[n_id_orig] = PreprocFeatureMap(
                    features[nodes[n_id_orig].split_feature_id], node_feature_name
                )
        else:
            # No pre-processing
            node_feature_name = features[n.split_feature_id]
        node_features[n_id] = node_feature_name
    new_feature_list = list(set(node_features.values()))
    return node_features, preproc_nodes, new_feature_list


# @staticmethod
# def determine_feature_mapping(
#     nodes: t.Dict[str,NodeType],
#     features: t.List[str],
# ):
#            # Priority (TODO move to function)
#     num_leaf_nodes = len(leaf_nodes)
#     output_priority_mapping = np.repeat(-1, num_leaf_nodes).astype(np.int64)
#     for leaf_node in leaf_nodes:
#         leaf_node_idx = node_id_mapping[leaf_node]
#         if tree.nodes[leaf_node].leaf_value == LeafNode.DEFAULT_LEAF_VALUE:
#             # Default leafs should be given least priority
#             output_priority_mapping[leaf_node_idx] = -2
#             continue

#         # Use subtree priorities from new format
#         if tree.subtrees:
#             # Find which subtree this leaf belongs to and use its priority
#             leaf_subtree_priority = _find_leaf_subtree_priority(leaf_node, tree)
#             output_priority_mapping[leaf_node_idx] = leaf_subtree_priority
#         else:
#             # Default priority based on leaf value
#             output_priority_mapping[leaf_node_idx] = tree.nodes[leaf_node].leaf_value


def _calculate_leaf_priority(
    nodes: t.Dict[str, CompiledNodeType],
    root_nodes: t.List[str],
    tree_priority: "t.List[SubTree]",
    leaf_nodes: "t.List[str]",
    node_children: "t.Dict[str, t.List[str]]",
):
    root_values = {st.rootNodeId: st.priority for st in tree_priority}
    # Python built in sort is guaranteed stable so if no root priority then will default to root order
    sorted_root_node = sorted(
        root_nodes, key=lambda x: root_values.get(x, float("inf"))
    )
    leaf_node_index = {l: i for i, l in enumerate(leaf_nodes)}
    num_leaves = len(leaf_nodes)
    leaf_node_order = -np.ones(num_leaves, dtype=np.int64)

    # Do a dfs through each subtree in order (iterative)
    # assign priority 0 to the left most leaf and increase to the rightmost leaf
    current_leaf_idx = 0
    # Traverse each root in priority order
    for root_id in sorted_root_node:
        # Use a stack for iterative DFS
        # We need to reverse children order when pushing to maintain left-to-right traversal
        stack = [root_id]
        while stack:
            node_id = stack.pop()
            # Get children of current node, or empty list if no children
            children = node_children.get(node_id, [])
            # If this is a leaf node (no children), assign priority
            if node_id in leaf_node_index:
                leaf_idx = leaf_node_index[node_id]
                leaf_node = t.cast(LeafNode, nodes[node_id])
                if leaf_node.leaf_value == LeafNode.DEFAULT_LEAF_VALUE:
                    leaf_node_order[leaf_idx] = (
                        num_leaves  # Put default values to the end
                    )
                else:
                    leaf_node_order[leaf_idx] = current_leaf_idx
                    current_leaf_idx += 1
            else:
                # If this is an internal node, add children to stack in reverse order
                # This ensures left-to-right traversal when popping from stack
                stack.extend(reversed(children))

    assert np.all(
        leaf_node_order != -1
    ), "Did not assign all leaf priorities. This means the produced node_children was missing leaves"
    # swap order because compiled tree expects Higher priority
    leaf_node_priority = len(leaf_node_order) - leaf_node_order - 1
    return leaf_node_priority


def _calculate_leaf_idx_to_output_mapping(
    nodes: t.Dict[str, CompiledNodeType],
    node_id_mapping: t.Dict[str, int],
    leaf_nodes: t.List[str],
):
    """The outputs of the tree are the index of the leaf node that was hit so we need a mapping to map that index to an item in the results table"""
    num_leaf_nodes = len(leaf_nodes)
    output_mapping = np.repeat(-1, num_leaf_nodes).astype(np.int64)
    for leaf_node_key in leaf_nodes:
        leaf_node_idx = node_id_mapping[leaf_node_key]
        output_mapping[leaf_node_idx] = t.cast(
            LeafNode, nodes[leaf_node_key]
        ).leaf_value
    return output_mapping


def _calculate_path_matrix(
    nodes: t.Dict[str, CompiledNodeType],
    root_nodes: t.Set[str],
    node_id_mapping: t.Dict[str, int],
    leaf_nodes: t.List[str],
    node_children: t.Dict[str, t.List[str]],
):
    """For each leaf node we calculate the index of the parent condition nodes taken to get there."""
    num_leaf_nodes = len(leaf_nodes)
    num_condition_nodes = len(node_id_mapping) - num_leaf_nodes
    # Note for many subtrees this may be inefficient
    # a better form may be one array to get the path in a subtree and one to determine which subtree the leaf belongs to
    # -np.ones((num_leaf_nodes, longest_subtree_len), dtype=np.int32), np.ones((num_leaf_nodes), dtype=np.int32)
    paths_matrix = -np.ones((num_leaf_nodes, num_condition_nodes), dtype=np.int32)
    leaf_root_node_mapping = -np.ones((num_leaf_nodes), dtype=np.int32)

    def dfs(node_key: str, path_vector: np.ndarray, root_node_idx: int):
        node = nodes[node_key]
        node_idx = node_id_mapping[node_key]
        if node.IS_LEAF:
            if leaf_root_node_mapping[node_idx] != -1:
                raise ValueError(f"Multiple paths point to same leaf {node_key}")
            paths_matrix[node_idx] = path_vector.copy()
            leaf_root_node_mapping[node_idx] = root_node_idx
        else:
            children = node_children.get(node_key, [])
            for child_idx, child in enumerate(children):
                new_path_vector = path_vector.copy()
                new_path_vector[node_idx - num_leaf_nodes] = child_idx
                dfs(child, new_path_vector, root_node_idx)

    for root in root_nodes:
        path_vector = -np.ones((num_condition_nodes,), dtype=np.int32)
        dfs(root, path_vector, node_id_mapping[root])
    return paths_matrix, leaf_root_node_mapping


def _get_node_weighting(
    node_roots: t.Set[str],
    node_children: t.Dict[str, t.List[str]],
    node_name_mapping: t.Dict[str, str],
) -> t.Dict[str, int]:
    # We only weight the origional nodes that way the sort will push the duplicated nodes to the end
    # Which is a requirement for the path mapping later
    to_explore = list({node_name_mapping.get(r, r) for r in node_roots})
    result = {k: 0 for k in to_explore}
    seen = set()
    while to_explore:
        parent_node = to_explore.pop()
        seen.add(parent_node)
        child_weight = result.get(parent_node, 0) + 1
        children = {
            c_mapped
            for c in node_children.get(parent_node, [])
            if (c_mapped := node_name_mapping.get(c, c)) not in seen
        }
        result.update({child: child_weight for child in children})
        to_explore.extend(children)
    return result


def compile(tree: "Tree") -> "CompiledTreeliteTree":
    if len(tree.nodes) <= 0:
        raise ValueError("Tree contains no nodes")
    output_df_map, output_leaf_map = tree.output_df_and_map()
    nodes, node_name_mapping, edges = _compile_nodes(
        tree.node_lookup, tree.edges, tree, output_leaf_map
    )
    tree_child_mapping = _build_child_mapping(edges)
    node_children = {
        # After filling implicit nodes there shouldnt be any issues here
        n_id: [n_vals[val_i] for val_i in range(nodes[n_id].child_count)]
        for n_id, n_vals in tree_child_mapping.items()
        if n_id in nodes
    }
    root_nodes = _identify_independent_tree_roots(nodes, edges)
    node_weights = _get_node_weighting(root_nodes, node_children, node_name_mapping)
    node_id_mapping, leaf_nodes = _get_node_id_mapping(nodes, node_weights)
    # Keep root nodes in order of node id to ensure no discrepancy in order between create vs root node order
    # (IE the builder will build them in order of lowest id first)
    root_nodes = sorted(root_nodes, key=lambda x: node_id_mapping[x])
    if len(root_nodes) <= 0:
        # We know that the tree contains nodes so only way for no root is a loop
        raise ValueError("Cannot build tree as it contains loops.")
    # tree = tree.model_copy(update={"nodes": nodes, "edges": edges})

    node_features, preproc_nodes, new_feature_list = determine_feature_mapping(
        nodes, node_name_mapping, tree.features
    )
    node_feature_id_mapping = {
        n_id: new_feature_list.index(ft) for n_id, ft in node_features.items()
    }

    builder = _build_treelite_tree(
        nodes=nodes,
        root_nodes=root_nodes,
        tree=tree,
        node_id_mapping=node_id_mapping,
        node_children=node_children,
        output_encoding=OutputEncoding.INDEX,
        num_outputs=len(output_df_map),
        node_feature_id_mapping=node_feature_id_mapping,
    )

    leaf_output_mapping = _calculate_leaf_idx_to_output_mapping(
        nodes=nodes, node_id_mapping=node_id_mapping, leaf_nodes=leaf_nodes
    )
    diff_set = set(leaf_output_mapping) - set(output_df_map.index)
    if diff_set:
        raise ValueError(
            f"Found output values with no matching items in output df: {diff_set}"
        )

    paths_matrix, leaf_root_node_mapping = _calculate_path_matrix(
        nodes=nodes,
        root_nodes=root_nodes,
        node_id_mapping=node_id_mapping,
        leaf_nodes=leaf_nodes,
        node_children=node_children,
    )
    # Paths matrix is in form num_leaf_nodes, num_condition_nodes at the moment
    # We want to get rid of the additionally created nodes
    # Lets say num_leaf_nodes = L, num_condition_nodes = N and num_original_condition nodes is O
    # path_matrix*origin_map = original_path_matrix
    # We want (L,N)*(N,O)->(L,O)
    path_origin_map = np.zeros(
        (paths_matrix.shape[1], len(set(node_name_mapping.values()) - set(leaf_nodes))),
        dtype=paths_matrix.dtype,
    )
    num_leaf_nodes = len(leaf_nodes)
    path_node_names = ["unknown"] * path_origin_map.shape[1]
    for k, i in node_id_mapping.items():
        if i < num_leaf_nodes:
            continue  # j is a leaf node
        i = i - num_leaf_nodes
        if k not in node_name_mapping:
            continue  # is probably a internal made up node for like range values
        j = node_id_mapping[node_name_mapping[k]]
        assert (
            j >= num_leaf_nodes
        ), "conditional node shouldn't be a duplicate of a leaf node"  # j is a leaf node (I dont think this should happen)
        j = j - num_leaf_nodes
        path_origin_map[i, j] = 1
        if path_node_names[j] != "unknown":
            assert (
                path_node_names[j] == node_name_mapping[k]
            ), "Path node name duplicate of multiple values"
        else:
            path_node_names[j] = node_name_mapping[k]
    # We have to do the +1, -1 so that we map the -1 (no action) to 0 before doing the normalization
    normalized_paths_matrix = ((paths_matrix + 1) @ path_origin_map) - 1

    leaf_node_set = set(leaf_nodes)
    reverse_node_id_lookup = {v: k for k, v in node_id_mapping.items()}
    node_condition_names = [
        reverse_node_id_lookup[i] for i in range(len(leaf_nodes), len(node_id_mapping))
    ]
    # Just some safety checks path_node_names can be removed when we are more confident
    assert all(
        v1 == v2 for v1, v2 in zip(node_condition_names, path_node_names)
    ), "I would think these should be the same"

    node_feature_names = {
        node_id: tree.features[node.split_feature_id]
        for node_id, node in nodes.items()
        if not node.IS_LEAF
    }

    compiled_tree = CompiledTreeliteTree(
        model=builder.commit(),
        root_nodes=np.array(root_nodes),
        output_priority_mapping=_calculate_leaf_priority(
            nodes=nodes,
            root_nodes=root_nodes,
            tree_priority=tree.subtrees,
            leaf_nodes=leaf_nodes,
            node_children=node_children,
        ),
        leaf_output_mapping=leaf_output_mapping,
        output_dataframe_mapping=output_df_map,
        # node_id_mapping=node_id_mapping,
        features=tree.features,
        paths_matrix=normalized_paths_matrix,
        leaf_root_node_mapping=leaf_root_node_mapping,
        nodes=nodes,
        node_id_mapping=node_id_mapping,
        node_condition_names=np.array(path_node_names, dtype=str),
        node_children=node_children,
        # For preproc
        preproc_nodes=preproc_nodes,
        preproc_features=new_feature_list,
        # For display
        node_feature_names=node_feature_names,
        variable_input_name=tree.variable_input_name if len(tree.variables) else None,
    )

    return compiled_tree
