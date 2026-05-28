import os
import pytest
import pandas as pd
import numpy as np
import treelite
from spockflow.components.treelite import Tree


@pytest.fixture
def datadir() -> str:
    return os.path.join(
        os.path.split(os.path.abspath(__file__))[0], "data", "treelite_config"
    )


@pytest.fixture
def basic_treelite_config(datadir: str) -> str:
    with open(os.path.join(datadir, "basic.json")) as fp:
        return fp.read()


@pytest.fixture
def multiple_trees_config(datadir: str) -> str:
    with open(os.path.join(datadir, "multiple_disconnected_trees.json")) as fp:
        return fp.read()


@pytest.fixture
def insufficient_features_config(datadir: str) -> str:
    with open(os.path.join(datadir, "insufficient_features.json")) as fp:
        return fp.read()


@pytest.fixture
def insufficient_outputs_config(datadir: str) -> str:
    with open(os.path.join(datadir, "insufficient_outputs.json")) as fp:
        return fp.read()


@pytest.fixture
def no_columns_config(datadir: str) -> str:
    with open(os.path.join(datadir, "no_columns.json")) as fp:
        return fp.read()


@pytest.fixture
def cyclic_tree_config(datadir: str) -> str:
    with open(os.path.join(datadir, "cyclic_tree.json")) as fp:
        return fp.read()


def test_parse_and_compile(basic_treelite_config: str):
    tree = Tree.model_validate_json(basic_treelite_config)

    assert len(tree.features) == 2
    assert set(tree.nodes.keys()) == {
        "0",
        "1",
        "2",
        "3a96d1da-b922-4c4d-bf24-0f7119b7eb72",
        "ab018e87-cb89-4bc4-8612-5cab02280f09",
    }

    compiled_tree = tree.compile()

    assert compiled_tree.output_priority_mapping.shape == (3,)
    assert compiled_tree.leaf_output_mapping.shape == (3,)
    assert all(
        str(dt) in ("object", "string") for dt in compiled_tree.output_dataframe_mapping.dtypes
    )
    leaf_priorities = [
        compiled_tree.output_priority_mapping[compiled_tree.node_id_mapping["1"]],
        compiled_tree.output_priority_mapping[
            compiled_tree.node_id_mapping["3a96d1da-b922-4c4d-bf24-0f7119b7eb72"]
        ],
        compiled_tree.output_priority_mapping[
            compiled_tree.node_id_mapping["ab018e87-cb89-4bc4-8612-5cab02280f09"]
        ],
    ]
    assert leaf_priorities[0] > leaf_priorities[1] > leaf_priorities[2]
    import numpy as np

    X = np.array(
        [
            [1, 0],
            [2, 0],
            [1, 1],
            [2, 1],
        ],
        dtype=np.float32,
    )
    predictions = treelite.gtil.predict(compiled_tree.model, X)
    # Shape is (Batch size, #num trees, 1 (We only ever output 1 value))
    assert predictions.shape == (4, 1, 1)
    priorities = compiled_tree.output_priority_mapping[predictions[:, :, 0].astype(int)]
    assert (
        priorities
        == np.array(
            [
                [leaf_priorities[0]],
                [leaf_priorities[2]],
                [leaf_priorities[0]],
                [leaf_priorities[1]],
            ]
        )
    ).all()
    outputs = compiled_tree.leaf_output_mapping[predictions[:, :, 0].astype(int)]
    assert (outputs == np.array([[0], [2], [0], [1]])).all()
    df_outs = compiled_tree.output_dataframe_mapping.loc[outputs.flatten()]
    assert (
        df_outs.description.values
        == np.array(
            [
                "Output of node 1",
                "Output of node ab018e87-cb89-4bc4-8612-5cab02280f09",
                "Output of node 1",
                "Output of node 3a96d1da-b922-4c4d-bf24-0f7119b7eb72",
            ]
        )
    ).all()


def test_multiple_disconnected_trees(multiple_trees_config: str):
    """Test handling of multiple independent trees."""
    tree = Tree.model_validate_json(multiple_trees_config)

    # Verify features and nodes
    assert len(tree.features) == 3
    assert set(tree.nodes.keys()) == {
        "tree1_root",
        "tree1_leaf1",
        "tree1_leaf2",
        "tree2_root",
        "tree2_leaf1",
        "tree2_leaf2",
    }

    compiled_tree = tree.compile()

    # Test predictions with sample data
    X = np.array(
        [
            [1.0, 0.0, 0.0],  # Should go to tree1_leaf1 and tree2_leaf1
            [2.0, 1.0, 0.0],  # Should go to tree1_leaf2 and tree2_leaf2
            [1.0, 2.0, 0.0],  # Should go to tree1_leaf1 and tree2_leaf2
        ],
        dtype=np.float32,
    )

    predictions = treelite.gtil.predict(compiled_tree.model, X)
    assert predictions.shape == (3, 2, 1)  # (batch_size, num_trees, 1)

    outputs = compiled_tree.leaf_output_mapping[predictions[:, :, 0].astype(int)]
    results = compiled_tree.output_dataframe_mapping.loc[outputs.flatten()].description
    expected_results = np.array(
        [
            "First tree left output",
            "Second tree right output",
            "First tree right output",
            "Second tree left output",
            "First tree left output",
            "Second tree left output",
        ]
    )
    assert (results == expected_results).all()


def test_insufficient_features(insufficient_features_config: str):
    """Test error handling when feature_id references non-existent feature."""
    tree = Tree.model_validate_json(insufficient_features_config)

    with pytest.raises(
        treelite.core.TreeliteError, match=r"split_index must be less than num_feature"
    ):
        tree.compile()


def test_insufficient_outputs(insufficient_outputs_config: str):
    """Test error handling when leaf value has no corresponding output."""
    tree = Tree.model_validate_json(insufficient_outputs_config)

    with pytest.raises(ValueError, match=r".*output values.*no matching items.*"):
        tree.compile()


def test_no_columns(no_columns_config: str):
    """Test handling of tree output without column definitions."""
    tree = Tree.model_validate_json(no_columns_config)
    with pytest.raises(ValueError, match=r".*output dataframe.*"):
        compiled_tree = tree.compile()


def test_cyclic_tree(cyclic_tree_config: str):
    """Test error handling for trees containing cycles."""
    tree = Tree.model_validate_json(cyclic_tree_config)

    with pytest.raises(ValueError, match=r".*contains loops.*"):
        tree.compile()


def test_leaf_priority_order(multiple_trees_config: str):
    """Test that leaf priorities are correctly assigned based on leaf_order."""
    tree = Tree.model_validate_json(multiple_trees_config)
    compiled_tree = tree.compile()

    # Get priorities for each leaf
    priorities = [
        compiled_tree.output_priority_mapping[compiled_tree.node_id_mapping[leaf]]
        for leaf in tree.leaf_order
    ]

    # Check that priorities are strictly decreasing
    assert all(priorities[i] > priorities[i + 1] for i in range(len(priorities) - 1))


def test_categorical_features(multiple_trees_config: str):
    """Test handling of categorical features."""
    tree = Tree.model_validate_json(multiple_trees_config)
    compiled_tree = tree.compile()

    # Test various categorical values
    X = np.array(
        [
            [1.0, 1.0, 0.0],  # Category in list
            [1.0, 5.0, 0.0],  # Category not in list
            [1.0, 0.0, 0.0],  # Category not in list
        ],
        dtype=np.float32,
    )

    predictions = treelite.gtil.predict(compiled_tree.model, X)
    outputs = compiled_tree.leaf_output_mapping[
        predictions[:, 1, 0].astype(int)
    ]  # Check second tree only

    # For tree2: category_list=[1,2], category_list_right_child=false
    # So categories 1,2 go left (value 2), others go right (value 3)
    expected_outputs = np.array([2, 3, 3])
    assert (outputs == expected_outputs).all()


def test_default_left_behavior(multiple_trees_config: str):
    """Test default_left behavior for missing/invalid values."""
    tree = Tree.model_validate_json(multiple_trees_config)
    compiled_tree = tree.compile()

    # Create data with NaN values
    X = np.array(
        [
            [np.nan, 1.0, 0.0],  # First tree: default_left=false
            [1.0, np.nan, 0.0],  # Second tree: default_left=true
        ],
        dtype=np.float32,
    )

    predictions = treelite.gtil.predict(compiled_tree.model, X)
    outputs = compiled_tree.leaf_output_mapping[predictions[:, :, 0].astype(int)]

    # First tree should go right (default_left=false)
    # Second tree should go left (default_left=true)
    expected_outputs = np.array(
        [
            [
                1,
                2,
            ],  # NaN in first feature -> right child of tree1, normal left path in tree2
            [
                0,
                2,
            ],  # Normal left path in tree1, NaN in second feature -> left child of tree2
        ]
    )
    assert (outputs == expected_outputs).all()


def test_create_and_run_hamilton_graph(basic_treelite_config: str):
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")
    assert set(tree_driver.graph.nodes.keys()) == {
        "tree",
        "tree.all_tree_paths",
        "tree.formatted_inputs",
        "tree.highest_priority_index",
        "tree.index",
        "tree.prioritized_outputs",
        "tree.prioritized_tree_paths",
        "tree.tree_results",
        "tree.tree_path_keys",
        "feature_0",
        "feature_1",
    }
    data = [
        [1, 0],
        [2, 0],
        [1, 1],
        [2, 1],
    ]
    formatted_inputs = tree_driver.raw_execute(
        inputs=pd.DataFrame(data=data, columns=tree.features),
        final_vars=["tree.formatted_inputs"],
    )["tree.formatted_inputs"]
    assert (formatted_inputs == np.array(data)).all()


def test_tree_results(basic_treelite_config: str):
    """Test the tree_results function which gets raw predictions from treelite."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    # Test with various input data
    input_data = pd.DataFrame(
        {"feature_0": [1.0, 2.0, 1.0, 2.0], "feature_1": [0.0, 0.0, 1.0, 1.0]}
    )

    results = tree_driver.raw_execute(
        inputs=input_data, final_vars=["tree.tree_results"]
    )["tree.tree_results"]

    # Based on the tree structure in basic.json:
    # feature_0 < 1.5 -> value 0
    # feature_0 >= 1.5 & feature_1 in [1,2,3,4] -> value 1
    # feature_0 >= 1.5 & feature_1 not in [1,2,3,4] -> value 2
    expected_results = np.array(
        [
            [0],  # feature_0=1, feature_1=0 -> left at first split
            [2],  # feature_0=2, feature_1=0 -> right at first split, right at second
            [0],  # feature_0=1, feature_1=1 -> left at first split
            [1],  # feature_0=2, feature_1=1 -> right at first split, left at second
        ]
    )

    assert (results == expected_results).all()


def test_highest_priority_index(basic_treelite_config: str):
    """Test the highest_priority_index function which determines which tree's output to use."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    input_data = pd.DataFrame(
        {"feature_0": [1.0, 2.0, 1.0, 2.0], "feature_1": [0.0, 0.0, 1.0, 1.0]}
    )

    results = tree_driver.raw_execute(
        inputs=input_data, final_vars=["tree.highest_priority_index"]
    )["tree.highest_priority_index"]

    # Each row should have the index of the highest priority result
    assert results.shape == (4,)
    assert (results >= 0).all()  # Indices should be non-negative
    assert (results < 1).all()  # Only one tree in basic config


def test_prioritized_outputs(basic_treelite_config: str):
    """Test the prioritized_outputs function which maps tree results to final output values."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    input_data = pd.DataFrame(
        {"feature_0": [1.0, 2.0, 1.0, 2.0], "feature_1": [0.0, 0.0, 1.0, 1.0]}
    )

    results = tree_driver.raw_execute(
        inputs=input_data, final_vars=["tree.prioritized_outputs"]
    )["tree.prioritized_outputs"]

    expected_outputs = np.array([0, 2, 0, 1])  # Based on the basic.json tree structure
    assert (results == expected_outputs).all()


def test_get_output(basic_treelite_config: str):
    """Test the get_output function which creates the final DataFrame output."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    # Test with DataFrame including custom index
    input_data = pd.DataFrame(
        {"feature_0": [1.0, 2.0, 1.0, 2.0], "feature_1": [0.0, 0.0, 1.0, 1.0]},
        index=["a", "b", "c", "d"],
    )

    results = tree_driver.raw_execute(inputs=input_data, final_vars=["tree"])["tree"]

    assert isinstance(results, pd.DataFrame)
    assert list(results.index) == ["a", "b", "c", "d"]
    assert set(results.columns) == {"name", "description", "value"}

    # Verify outputs match expected descriptions from basic.json
    assert results.loc["a", "description"] == "Output of node 1"
    assert (
        results.loc["b", "description"]
        == "Output of node ab018e87-cb89-4bc4-8612-5cab02280f09"
    )
    assert results.loc["c", "description"] == "Output of node 1"
    assert (
        results.loc["d", "description"]
        == "Output of node 3a96d1da-b922-4c4d-bf24-0f7119b7eb72"
    )


# THESE MIGHT CHANGE SOON SO NOT TOO WORRIED ABOUT TEST FOR NOW
# def test_all_tree_paths(basic_treelite_config: str):
#     """Test the all_tree_paths function which returns the path taken through each tree.

#     The path matrix has shape [batch_size, num_subtrees, num_condition_nodes] where:
#     0: Node not visited
#     1: Condition evaluated to true
#     -1: Condition evaluated to false
#     """
#     tree = Tree.model_validate_json(basic_treelite_config)
#     tree_driver = tree.get_driver({}, name="tree")

#     input_data = pd.DataFrame(
#         {"feature_0": [1.0, 2.0, 1.0, 2.0], "feature_1": [0.0, 0.0, 1.0, 1.0]}
#     )

#     results = tree_driver.raw_execute(
#         inputs=input_data, final_vars=["tree.all_tree_paths", "tree.tree_path_keys"]
#     )
#     paths = results["tree.all_tree_paths"]
#     path_labels = results["tree.tree_path_keys"]

#     # Verify shape: (n_samples, n_trees, n_condition_nodes)
#     assert len(paths.shape) == 3
#     assert paths.shape == (
#         4,
#         1,
#         2,
#     )  # 4 samples, 1 tree, 2 condition nodes (root and categorical test)
#     assert len(path_labels) == 2  # Should match number of condition nodes

#     # Verify path labels match node keys from basic.json
#     condition_labels = ["0", "2"]
#     assert set(path_labels) == set(
#         condition_labels
#     )  # Root node and categorical test node
#     idx_o = path_labels.index(condition_labels[0])
#     idx_1 = path_labels.index(condition_labels[1])
#     gt_paths = np.array(
#         [
#             # For sample 1: feature_0 = 1.0, feature_1 = 0.0
#             # Should go left at root (feature_0 < 1.5) and not hit categorical node
#             [1, 0],
#             # For sample 2: feature_0 = 2.0, feature_1 = 0.0
#             # Should go right at root (feature_0 >= 1.5) and right at categorical (0 not in [1,2,3,4])
#             [-1, -1],
#             # For sample 3: feature_0 = 1.0, feature_1 = 1.0
#             # Should go left at root and not hit categorical node
#             [1, 0],
#             # For sample 4: feature_0 = 2.0, feature_1 = 1.0
#             # Should go right at root and left at categorical (1 in [1,2,3,4])
#             [-1, 1],
#         ]
#     )[:, [idx_o, idx_1]]

#     assert (paths[:, 0] == gt_paths).all()


# def test_prioritized_tree_paths(basic_treelite_config: str):
#     """Test the prioritized_tree_paths function which returns the winning path for each sample.

#     Similar to all_tree_paths but only returns the path from the highest priority tree
#     for each sample.
#     """
#     tree = Tree.model_validate_json(basic_treelite_config)
#     tree_driver = tree.get_driver({}, name="tree")

#     input_data = pd.DataFrame(
#         {"feature_0": [1.0, 2.0, 1.0, 2.0], "feature_1": [0.0, 0.0, 1.0, 1.0]}
#     )

#     results = tree_driver.raw_execute(
#         inputs=input_data,
#         final_vars=[
#             "tree.prioritized_tree_paths",
#             "tree.all_tree_paths",
#             "tree.tree_path_keys",
#             "tree.highest_priority_index",
#         ],
#     )

#     paths = results["tree.prioritized_tree_paths"]
#     path_labels = results["tree.tree_path_keys"]
#     priority_indices = results["tree.highest_priority_index"]

#     # Verify shape: (n_samples, n_condition_nodes)
#     assert paths.shape == (4, 2)  # 4 samples, 2 condition nodes
#     assert len(path_labels) == 2
#     condition_labels = ["0", "2"]
#     assert set(path_labels) == set(
#         condition_labels
#     )  # Root node and categorical test node
#     idx_o = path_labels.index(condition_labels[0])
#     idx_1 = path_labels.index(condition_labels[1])

#     # Since we only have one tree in basic config, prioritized paths should match
#     # the first (and only) tree in all_tree_paths
#     all_paths = results["tree.all_tree_paths"]
#     assert (paths == all_paths[:, 0, :]).all()

#     # Verify paths match expected decision logic
#     expected_paths = np.array(
#         [
#             [1, 0],  # Left at root (true), don't reach categorical
#             [-1, -1],  # Right at root (false), right at categorical (false)
#             [1, 0],  # Left at root (true), don't reach categorical
#             [-1, 1],  # Right at root (false), left at categorical (true)
#         ]
#     )[:, [idx_o, idx_1]]
#     assert (paths == expected_paths).all()


def test_tree_path_keys(basic_treelite_config: str):
    """Test that tree_path_keys returns the correct node keys in order."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    results = tree_driver.raw_execute(
        inputs={"feature_0": [1.0], "feature_1": [0.0]},
        final_vars=["tree.tree_path_keys"],
    )
    path_labels = results["tree.tree_path_keys"]

    # Verify labels match node keys from basic.json
    assert path_labels == ["0", "2"]  # Root node and categorical test node

    # Verify these nodes exist in the tree and are condition nodes
    assert tree.nodes[path_labels[0]].node_type in [
        "numerical_test_node",
        "categorical_test_node",
    ]
    assert tree.nodes[path_labels[1]].node_type in [
        "numerical_test_node",
        "categorical_test_node",
    ]


def test_index_handling(basic_treelite_config: str):
    """Test index handling with different input types."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    # Test with pandas Series
    input_data = {
        "feature_0": pd.Series([1.0, 2.0], index=["a", "b"]),
        "feature_1": pd.Series([0.0, 1.0], index=["a", "b"]),
    }

    results = tree_driver.raw_execute(inputs=input_data, final_vars=["tree.index"])[
        "tree.index"
    ]

    assert isinstance(results, pd.Index)
    assert list(results) == ["a", "b"]

    # Test with numpy arrays
    input_data = {"feature_0": np.array([1.0, 2.0]), "feature_1": np.array([0.0, 1.0])}

    results = tree_driver.raw_execute(inputs=input_data, final_vars=["tree.index"])[
        "tree.index"
    ]

    assert isinstance(results, pd.RangeIndex)
    assert list(results) == [0, 1]


def test_formatted_inputs(basic_treelite_config: str):
    """Test the formatted_inputs function with different input types."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    # Test with pandas Series
    input_data = {
        "feature_0": pd.Series([1.0, 2.0]),
        "feature_1": pd.Series([0.0, 1.0]),
    }

    results = tree_driver.raw_execute(
        inputs=input_data, final_vars=["tree.formatted_inputs"]
    )["tree.formatted_inputs"]

    expected = np.array([[1.0, 0.0], [2.0, 1.0]])
    assert (results == expected).all()

    # Test with numpy arrays
    input_data = {"feature_0": np.array([1.0, 2.0]), "feature_1": np.array([0.0, 1.0])}

    results = tree_driver.raw_execute(
        inputs=input_data, final_vars=["tree.formatted_inputs"]
    )["tree.formatted_inputs"]

    assert (results == expected).all()


def test_end_to_end_workflow(basic_treelite_config: str):
    """Test the complete workflow from input to output."""
    tree = Tree.model_validate_json(basic_treelite_config)
    tree_driver = tree.get_driver({}, name="tree")

    # Test with mixed input types and custom index
    input_data = {
        "feature_0": pd.Series([1.0, 2.0, 1.0, 2.0], index=["a", "b", "c", "d"]),
        "feature_1": np.array([0.0, 0.0, 1.0, 1.0]),
    }

    results = tree_driver.raw_execute(
        inputs=input_data,
        final_vars=[
            "tree",
            "tree.formatted_inputs",
            "tree.tree_results",
            "tree.highest_priority_index",
            "tree.prioritized_outputs",
            "tree.all_tree_paths",
            "tree.prioritized_tree_paths",
        ],
    )

    # Verify all expected outputs are present
    assert set(results.keys()) == {
        "tree",
        "tree.formatted_inputs",
        "tree.tree_results",
        "tree.highest_priority_index",
        "tree.prioritized_outputs",
        "tree.all_tree_paths",
        "tree.prioritized_tree_paths",
    }

    # Verify final output has correct structure
    assert isinstance(results["tree"], pd.DataFrame)
    assert list(results["tree"].index) == ["a", "b", "c", "d"]
    assert set(results["tree"].columns) == {"name", "description", "value"}
