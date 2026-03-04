import typing as t
from pydantic import BaseModel, Field, Tag, RootModel, Discriminator, ValidationError, model_validator
from .v0.tree import Tree as V0Tree
from .v1.tree import Tree as V1Tree
from spockflow.nodes import VariableNode


class DeprecatedTree(BaseModel):
    @model_validator(mode='before')
    def raise_error(value):
        raise ValueError("This version of the tree has been deprecated")

class NodeValidationResult(BaseModel):
    """Validation result for a single node."""

    node_id: str
    node_type: t.Optional[str] = None
    errors: t.List[str] = Field(default_factory=list)
    warnings: t.List[str] = Field(default_factory=list)


class ValidationResults(BaseModel):
    """Comprehensive tree validation results focused on user-actionable feedback."""

    valid: bool = False
    errors: t.List[str] = Field(default_factory=list)
    warnings: t.List[str] = Field(default_factory=list)

    # Basic counts for UI display
    node_count: int = 0
    feature_count: int = 0
    subtree_count: t.Optional[int] = None
    format_version: t.Optional[int] = None

    # Node-specific issues
    node_results: t.List[NodeValidationResult] = Field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add a tree-level error."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a tree-level warning."""
        self.warnings.append(message)

    def add_node_error(
        self, node_id: str, message: str, node_type: t.Optional[str] = None
    ) -> None:
        """Add a node-specific error."""
        # Find or create node result
        node_result = next(
            (nr for nr in self.node_results if nr.node_id == node_id), None
        )
        if not node_result:
            node_result = NodeValidationResult(node_id=node_id, node_type=node_type)
            self.node_results.append(node_result)

        node_result.errors.append(message)
        self.valid = False

    def add_node_warning(
        self, node_id: str, message: str, node_type: t.Optional[str] = None
    ) -> None:
        """Add a node-specific warning."""
        # Find or create node result
        node_result = next(
            (nr for nr in self.node_results if nr.node_id == node_id), None
        )
        if not node_result:
            node_result = NodeValidationResult(node_id=node_id, node_type=node_type)
            self.node_results.append(node_result)

        node_result.warnings.append(message)


def obj_get(obj: any, k: str, default=None):
    if isinstance(obj, dict):
        return obj.get(k, default)
    return getattr(obj, k, default)


def get_tree_version(obj: any):
    format_version = obj_get(obj, "formatVersion")
    if format_version is None:
        format_version = obj_get(obj, "format_version")
    if format_version is not None:
        return f"v{format_version}-tree"
    else:
        nodes = obj_get(obj, "nodes")
        if isinstance(nodes, dict):
            return "v0-tree"
        return "v1-tree"


_Tree = t.Annotated[
    t.Union[
        t.Annotated[DeprecatedTree, Tag("v0-tree")],
        t.Annotated[V1Tree, Tag("v1-tree")],
    ],
    Discriminator(get_tree_version),
]


class Tree(RootModel, VariableNode):
    root: _Tree


    def upgrade(self):
        """Upgrade tree to latest version via upgrade chain"""
        upgraded_tree = self.root.upgrade()

        # If the upgrade returned a different tree, wrap it
        if upgraded_tree != self.root:
            return Tree(root=upgraded_tree)
        else:
            return self

# TODO need a better mechanism here
def validate_tree(
    obj: t.Any, return_results: bool = False
) -> t.Union[Tree, t.Tuple[Tree, ValidationResults]]:
    """
    Comprehensive tree validation that includes:
    1. Tree structure validation and auto-upgrade
    2. Tree-specific validations (via TreeV1.verify_tree)

    Args:
        obj: Tree configuration object (dict or Tree instance)
        return_results: If True, returns (tree, validation_results) tuple

    Returns:
        Tree instance if return_results=False, otherwise (Tree, ValidationResults) tuple
    """
    results = ValidationResults()

    try:
        # Step 1: Basic tree structure validation and upgrade
        validated_tree = Tree.model_validate(obj)

        # Check if tree needs upgrading
        original_version = validated_tree.root.format_version
        upgraded_tree = validated_tree.upgrade()
        final_version = upgraded_tree.root.format_version

        # Track upgrade info
        results.format_version = final_version

        if original_version != final_version:
            results.add_warning(
                f"Tree automatically upgraded from version {original_version} to {final_version}"
            )

        # Basic tree info for UI display
        results.node_count = len(upgraded_tree.root.nodes)
        results.feature_count = len(upgraded_tree.root.features)
        results.subtree_count = (
            len(upgraded_tree.root.subtrees) if upgraded_tree.root.subtrees else 0
        )

        if results.subtree_count == 0:
            results.add_warning("No subtrees defined in tree configuration")

        # Step 2: Tree-specific validations (V1 only)
        if hasattr(upgraded_tree.root, "verify_tree"):
            try:
                upgraded_tree.root.verify_tree(results)
            except Exception as e:
                results.add_error(f"Tree verification failed: {str(e)}")

        # Step 3: Compilation validation
        try:
            _ = upgraded_tree.compile()
        except Exception as e:
            results.add_error(f"Tree compilation failed: {str(e)}")

        # Determine overall validity - valid if no errors
        results.valid = len(results.errors) == 0 and not any(
            len(nr.errors) > 0 for nr in results.node_results
        )

        if return_results:
            return upgraded_tree, results
        else:
            return upgraded_tree

    except ValidationError as e:
        results.valid = False
        for error in e.errors():
            results.add_error(str(error))

        if return_results:
            # Return the original object as Tree if possible, otherwise None
            try:
                tree = Tree.model_validate(obj) if not isinstance(obj, Tree) else obj
            except Exception:
                tree = None
            return tree, results
        else:
            # Re-raise validation error if not returning results
            raise
