import os
import pytest
import pandas as pd
from spockflow.components.tree.v1.core import Tree, ConditionedNode, ChildTree


@pytest.fixture
def tree():
    return Tree()


def value(val):
    return pd.DataFrame({"value": [val]})


def values(vals):
    return pd.DataFrame({"value": vals})


def test_add_node_to_tree(tree):
    tree.condition(output=value(10), condition="A")
    tree.condition(output=value(20), condition="B")

    assert len(tree.root.nodes) == 2
    assert tree.root.nodes[0].value.loc[0, "value"] == 10
    assert tree.root.nodes[0].condition == "A"
    assert tree.root.nodes[1].value.loc[0, "value"] == 20
    assert tree.root.nodes[1].condition == "B"


def test_add_nested_node_to_tree(tree):
    @tree.condition(output=value(10))
    def test_error():
        pass

    with pytest.raises(ValueError):
        test_error.condition(output=value(5), condition="X")

    @tree.condition()
    def outer():
        pass

    outer.condition(output=value(5), condition="X")

    @outer.condition(output=value(6))
    def Y():
        pass

    assert len(tree.root.nodes) == 2
    assert tree.root.nodes[0].condition is test_error
    assert tree.root.nodes[0].value.loc[0, "value"] == 10
    assert len(tree.root.nodes[1].value.nodes) == 2
    assert tree.root.nodes[1].value.nodes[0].value.loc[0, "value"] == 5
    assert tree.root.nodes[1].value.nodes[0].condition == "X"
    assert tree.root.nodes[1].value.nodes[1].value.loc[0, "value"] == 6
    assert tree.root.nodes[1].value.nodes[1].condition is Y


def test_set_default_value(tree):
    tree.set_default(value(999))

    assert tree.root.default_value.loc[0, "value"] == 999


def test_include_subtree(tree):
    subtree = Tree()
    subtree.condition(output=value(100), condition="SubA")
    subtree.condition(output=value(200), condition="SubB")

    tree.include_subtree(subtree)

    assert len(tree.root.nodes) == 2
    assert tree.root.nodes[0].value.loc[0, "value"] == 100
    assert tree.root.nodes[0].condition == "SubA"
    assert tree.root.nodes[1].value.loc[0, "value"] == 200
    assert tree.root.nodes[1].condition == "SubB"

    tree.include_subtree(subtree, condition="C")

    assert len(tree.root.nodes) == 3
    assert tree.root.nodes[2].condition == "C"
    assert tree.root.nodes[2].value.nodes[0].value.loc[0, "value"] == 100
    assert tree.root.nodes[2].value.nodes[0].condition == "SubA"
    assert tree.root.nodes[2].value.nodes[1].value.loc[0, "value"] == 200
    assert tree.root.nodes[2].value.nodes[1].condition == "SubB"


def test_merge_subtrees(tree):
    subtree1 = Tree()
    subtree1.condition(output=value(100), condition="SubA")

    subtree2 = Tree()
    subtree2.condition(output=value(200), condition="SubB")

    tree.include_subtree(subtree1)
    tree.include_subtree(subtree2)

    assert len(tree.root.nodes) == 2
    assert tree.root.nodes[0].value.loc[0, "value"] == 100
    assert tree.root.nodes[0].condition == "SubA"
    assert tree.root.nodes[1].value.loc[0, "value"] == 200
    assert tree.root.nodes[1].condition == "SubB"


def test_annotations(tree):
    subtree1 = Tree()
    subtree1.condition(output=value(100), condition="SubA")

    subtree2 = Tree()
    subtree2.condition(output=value(200), condition="SubB")

    tree.include_subtree(subtree1)
    tree.include_subtree(subtree2)

    assert len(tree.root.nodes) == 2
    assert tree.root.nodes[0].value.loc[0, "value"] == 100
    assert tree.root.nodes[0].condition == "SubA"
    assert tree.root.nodes[1].value.loc[0, "value"] == 200
    assert tree.root.nodes[1].condition == "SubB"


def test_err_on_multiple_condition(tree):
    with pytest.raises(ValueError):

        @tree.condition(output=value(10), condition="A")
        def layer2():
            pass


def test_nested_subtree_with_annotations(tree):
    # Define annotated functions for tree construction
    @tree.condition()
    def A():
        pass

    @A.condition()
    def B():
        pass

    @B.condition()
    def C():
        pass

    @C.condition()
    def D():
        pass

    # Include a subtree before condition
    subtree = ChildTree()
    tree.condition(output=value(10), condition="DSubtree", child_tree=subtree)
    D.include_subtree(subtree)

    # Set default value and include subtree in the deepest layer
    @D.condition()
    def E():
        pass

    # Set default value for the deepest layer
    E.set_default(value(999))

    # Include another subtree into the deepest layer
    subtree = ChildTree()
    tree.condition(output=value(20), condition="F", child_tree=subtree)

    # Include the subtree into the current layer
    E.include_subtree(subtree)

    # Perform assertions to validate the tree structure
    order = [A, B, C, D]
    curr = tree.root.nodes
    for cond in order:
        assert len(curr) == 1
        assert curr[0].condition is cond
        curr = curr[0].value.nodes
    assert len(curr) == 2
    assert curr[0].value.loc[0, "value"] == 10
    assert curr[0].condition == "DSubtree"
    curr = curr[1]
    assert curr.condition is E
    assert curr.value.default_value.loc[0, "value"] == 999
    assert len(curr.value.nodes) == 1
    curr = curr.value.nodes[0]
    assert curr.value.loc[0, "value"] == 20
    assert curr.condition == "F"


def test_invalid_length_combinations(tree):
    tree.condition(
        output=values([1, 2]), condition="A"
    )  # Invalid: Length of value and condition mismatch

    subtree = Tree()
    subtree.condition(output=values([100]), condition="SubA")
    subtree.condition(output=values([100, 200, 300]), condition="SubB")
    with pytest.raises(ValueError):
        subtree.condition(output=values([100, 200, 300, 400]), condition="SubC")

    with pytest.raises(ValueError):
        tree.include_subtree(
            subtree
        )  # Invalid: Length of value and condition in subtree mismatch


def test_set_default_twice(tree):
    tree.set_default(value(999))

    with pytest.raises(ValueError):
        tree.set_default(value(888))  # Default value already set


def test_merge_subtrees_with_only_defaults(tree):
    subtree1 = Tree()
    subtree1.set_default(value(100))

    subtree2 = Tree()
    subtree2.set_default(value(200))

    tree.set_default(value(999))
    with pytest.raises(ValueError):
        tree.include_subtree(subtree1)

    with pytest.raises(ValueError):
        tree.include_subtree(subtree2)


def test_merge_subtrees_with_defaults(tree):
    subtree = Tree()
    subtree.condition(output=value(100), condition="SubA")
    subtree.condition(output=value(200), condition="SubB")
    subtree.set_default(output=value(300))

    subtree2 = Tree()
    subtree2.condition(output=value(1000), condition="SubD")
    subtree2.condition(output=value(2000), condition="SubE")
    subtree2.set_default(output=value(3000))

    tree.include_subtree(subtree)
    with pytest.raises(ValueError):
        tree.include_subtree(subtree2)


def test_circular_dep_tree(tree):

    # TODO this should raise an error
    with pytest.raises(ValueError):
        tree.include_subtree(tree, condition="A")
    # TODO this should work
    tree.include_subtree(tree.copy(), condition="A")

    subtree_1 = Tree()
    # TODO get the node output to work
    subtree_2 = subtree_1.condition(condition="A")

    with pytest.raises(ValueError):
        subtree_2.condition(condition="A", output=subtree_1)


def test_merge_with_tree_as_default_value(tree):
    tree0 = Tree()
    tree0.condition(output=value(100), condition="SubA")
    tree0.condition(output=value(200), condition="SubB")

    tree.condition(output=value(300), condition="SubC")
    tree.set_default(tree0)

    assert len(tree.root.nodes) == 3
    assert tree.root.nodes[0].value.loc[0, "value"] == 300
    assert tree.root.nodes[0].condition == "SubC"
    assert tree.root.nodes[1].value.loc[0, "value"] == 100
    assert tree.root.nodes[1].condition == "SubA"
    assert tree.root.nodes[2].value.loc[0, "value"] == 200
    assert tree.root.nodes[2].condition == "SubB"
