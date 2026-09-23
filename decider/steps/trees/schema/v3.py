"""The v3 tree document: positioned nodes joined by edges, as the tree editor saves it."""
from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decider.steps.trees.schema.nodes import NodeData, arity
from decider.steps.trees.schema.tree import BaseTreeDocument, Node, Rule, Tree, add_node


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class PositionedNode(BaseModel):
    id: str
    position: Position = Field(default_factory=Position)
    data: NodeData


class MultiEdgeData(BaseModel):
    """The branch indices of the source node this edge leaves from."""

    sourceIndex: t.List[int]

    @model_validator(mode="before")
    @classmethod
    def _one_index_is_a_list(cls, value: t.Any) -> t.Any:
        if isinstance(value, dict) and not isinstance(value.get("sourceIndex", []), (list, tuple)):
            return {**value, "sourceIndex": [value["sourceIndex"]]}
        return value


class MultiSourceEdge(BaseModel):
    id: t.Optional[str] = None
    source: str
    target: str
    data: MultiEdgeData


class TreeMetadata(BaseModel):
    name: t.Optional[str] = None
    description: t.Optional[str] = None


class SubTree(BaseModel):
    id: t.Optional[str] = None
    name: t.Optional[str] = None


class V3TreeDocument(BaseTreeDocument):
    """A v3 tree: `nodes` say what each node tests, `edges` wire branch `sourceIndex` of `source` to `target`.

    A branch with no edge selects the default output row. The root is the
    node nothing points at (the first listed subtree's, when there are several).

    Example (age < 30 is "young", anything else the default "adult")::

        V3TreeDocument(
            nodes=[{"id": "root", "data": {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 30}}},
                   {"id": "young", "data": {"type": "leaf", "result_idx": 0}}],
            edges=[{"source": "root", "target": "young", "data": {"sourceIndex": 0}}],
            output={"data": [{"group": "young"}], "default": {"group": "adult"}, "dtypes": [("group", "String")]},
        ).to_tree()
    """

    model_config = ConfigDict(populate_by_name=True)

    type: t.Literal["v3-tree"] = "v3-tree"
    metadata: t.Optional[TreeMetadata] = None
    nodes: t.List[PositionedNode]
    edges: t.List[MultiSourceEdge] = Field(default_factory=list)
    subtrees: t.List[SubTree] = Field(default_factory=list)
    format_version: t.Literal[3] = Field(alias="formatVersion", default=3)

    @model_validator(mode="after")
    def _normalise(self) -> V3TreeDocument:
        if not self.nodes:
            raise ValueError("tree has no nodes")
        ids = [n.id for n in self.nodes]
        if dupes := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"tree has duplicate node id(s): {dupes}")
        for e in self.edges:
            for end in (e.source, e.target):
                if end not in ids:
                    raise ValueError(f"edge {e.id or ''} refers to unknown node {end!r}")
        self._tree = self._build()
        return self

    def _root_id(self) -> str:
        targets = {e.target for e in self.edges}
        roots = [n.id for n in self.nodes if n.id not in targets]
        if not roots:
            raise ValueError("tree has no root node (its nodes form a cycle)")
        subtree_order = {st.id: i for i, st in enumerate(self.subtrees) if st.id}
        return min(roots, key=lambda r: subtree_order.get(r, len(subtree_order)))

    def _build(self) -> Tree:
        data = {n.id: n.data for n in self.nodes}
        branches: dict[str, dict[int, str]] = {}
        for edge in self.edges:
            for i in edge.data.sourceIndex:
                branches.setdefault(edge.source, {})[i] = edge.target
        defaults = self._param_defaults()
        nodes: dict[str, Node] = {}
        root = self._root_id()
        todo = [root]
        while todo:
            nid = todo.pop()
            if nid in nodes:
                continue
            children = [branches.get(nid, {}).get(i) for i in range(arity(data[nid]))]
            add_node(nodes, nid, data[nid], children, defaults)
            todo += filter(None, children)
        return Tree(name=self.name, rules=(Rule(root=root),), nodes=nodes, output=self.output)
