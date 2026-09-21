import typing as t
from pydantic import (
    BaseModel,
    Field,
    Tag,
    RootModel,
    Discriminator,
    model_validator,
)

# from .v0.tree import Tree as V0Tree
from .v1.tree import Tree as V1Tree
from .v2.tree import Tree as V2Tree
from .v3.tree import Tree as V3Tree


class DeprecatedTree(BaseModel):
    type: t.Literal["ui-tree"] = "ui-tree"

    @model_validator(mode="before")
    def raise_error(value):
        raise ValueError("This version of the tree has been deprecated")


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
        t.Annotated[V2Tree, Tag("v2-tree")],
        t.Annotated[V3Tree, Tag("v3-tree")],
    ],
    Discriminator(get_tree_version),
]


class Tree(RootModel):
    root: _Tree


    def upgrade(self):
        """Upgrade tree to latest version (v3) via upgrade chain."""
        upgraded_root = self.root

        # Keep upgrading until we reach v3
        while hasattr(upgraded_root, "upgrade") and upgraded_root.format_version < 3:
            upgraded_root = upgraded_root.upgrade()

        # If the upgrade changed the tree, return new wrapper
        if upgraded_root != self.root:
            return Tree(root=upgraded_root)
        else:
            return self

    @classmethod
    def latest_format_version(cls) -> int:
        """Helper to get the latest format version for UI display."""
        return 3

    @classmethod
    def default_tree(cls):
        """Create a default V3 tree with a single leaf node."""
        from .v3.nodes_ui import LeafNode, Position, PositionedNode
        from ..common.shared import TreeOutput

        return cls(
            root=V3Tree(
                type="v3-tree",
                metadata=None,
                edges=[],
                nodes=[
                    PositionedNode(
                        id="node-1",
                        position=Position(x=100.0, y=100.0),
                        data=LeafNode(type="leaf", result_idx=-1),
                    )
                ],
                subtrees=[],
                input_schema=None,
                format_version=3,
                output=TreeOutput(
                    data=[
                        {
                            "Action": "Alert",
                            "Description": "A rule suggests that there is an alert",
                        },
                        {
                            "Action": "Block",
                            "Description": "This is more serious and must be blocked",
                        },
                    ],
                    default={"Action": "Allow", "Description": "Default Action"},
                    dtypes=[("Action", "string"), ("Description", "string")],
                    type_defs={},
                ),
                parameters={},
                parameters_col="parameters",
            ),
        )
