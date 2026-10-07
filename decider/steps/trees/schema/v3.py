"""The v3 tree document: positioned nodes joined by edges, as the tree editor saves it."""
from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict, Field, model_validator

from decider.steps.trees.schema.flat import (
    CasesBranch,
    CasesIsInRule,
    CasesRangesRule,
    CasesStringMatchRule,
    CompositeRule,
    FlatRuleDocument,
    LeafRule,
    RuleRoot,
    RuleType,
    UnaryRule,
)
from decider.steps.trees.schema.nodes import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    CompositeNode,
    LeafNode,
    NodeData,
    UnaryNode,
    arity,
)
from decider.steps.trees.schema.tree import BaseTreeDocument, Node, Rule, Tree, add_node

__all__ = [
    "MultiEdgeData",
    "MultiSourceEdge",
    "Position",
    "PositionedNode",
    "SubTree",
    "TreeMetadata",
    "V3TreeDocument",
]


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

    # The tree editor saves its trees as "ui-tree"; the version is `formatVersion`, as decider_old read it.
    type: t.Literal["v3-tree", "ui-tree"] = "v3-tree"
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
        # Deleting a node in the editor can leave its outgoing edges behind; nothing walks them.
        self.edges = [e for e in self.edges if e.source in ids]
        for e in self.edges:
            if e.target not in ids:
                raise ValueError(f"edge {e.id or ''} refers to unknown node {e.target!r}")
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

    def to_flat_rules(self) -> FlatRuleDocument:
        """The same tree as one nested flat rule, from the root this tree runs; node ids are kept.

        A node with several parents is repeated under each, so the nesting can be larger than the graph.

        Example::

            V3TreeDocument(nodes=[{"id": "a", "data": {"type": "leaf"}}]).to_flat_rules()
        """
        data = {n.id: n.data for n in self.nodes}
        tree_nodes = self._tree.nodes

        def rule(nid: str) -> RuleType:
            node, d = data[nid], tree_nodes[nid]
            kids = [None if c is None else rule(c) for c in d.children]
            fields = {k: getattr(node, k) for k in type(node).model_fields if k not in ("id", "conditions")}
            if isinstance(node, LeafNode):
                return LeafRule(id=nid, **fields)
            if isinstance(node, UnaryNode):
                return UnaryRule(id=nid, then=kids[0], otherwise=kids[1], **fields)
            if isinstance(node, CompositeNode):
                return CompositeRule(id=nid, then=kids[0], otherwise=kids[1], conditions=node.conditions, **fields)
            cls = {CasesRanges: CasesRangesRule, CasesStringMatch: CasesStringMatchRule, CasesIsIn: CasesIsInRule}[type(node)]
            return cls(id=nid, conditions=[CasesBranch(when=c, then=i) for i, c in enumerate(node.conditions)],
                       otherwise=len(node.conditions), branches=[k or LeafRule(result_idx=-1) for k in kids], **fields)

        return FlatRuleDocument(name=self.name, output=self.output, parameters=self.parameters,
                                input_schema=self.input_schema, rule=RuleRoot(rule=rule(self._tree.rules[0].root)))
