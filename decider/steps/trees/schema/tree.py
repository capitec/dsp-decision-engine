"""The one tree model every accepted document format normalises into."""
from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from decider._serde import TStruct
from decider.steps.trees.schema.conditions import SchemaModel, get_field
from decider.steps.trees.schema.nodes import NodeData, arity
from decider.steps._values import ParamRef

__all__ = [
    "Node",
    "ParameterInfo",
    "Rule",
    "Tree",
    "TreeOutput",
]


class TreeOutput(BaseModel):
    """The rows a tree's leaves select: `data[result_idx]`, or `default` for -1.

    `dtypes` names each output column and its type, e.g. `[("band", "String")]`.
    A column a row leaves out is null there; no `default` (or `{}`) is a row of nulls.
    `{"type": "Custom", "type_id": "bands"}` names a categorical in `type_defs`, a string column.

    Example::

        TreeOutput(data=[{"band": "low"}], default={"band": "high"}, dtypes=[("band", "String")])
    """

    data: t.List[t.Dict[str, t.Any]] = Field(default_factory=list)
    default: t.Optional[t.Dict[str, t.Any]] = None
    dtypes: TStruct = Field(default_factory=list)
    type_defs: t.Dict[str, t.Any] = Field(default_factory=dict)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.dtypes) if isinstance(self.dtypes, dict) else tuple(name for name, _ in self.dtypes)

    def column_types(self) -> list[tuple[str, t.Any]]:
        """Each column with its type's name, e.g. `[("band", "String")]`."""
        out = []
        for column, dtype in self.dtypes.items() if isinstance(self.dtypes, dict) else self.dtypes:
            name = getattr(dtype, "type", dtype)
            if isinstance(name, str) and name.lower() == "custom":
                type_id = dtype.model_extra.get("type_id")
                kind = get_field(self.type_defs.get(type_id), "type")
                if kind != "categorical":
                    raise ValueError(f"output column {column!r}: custom type {type_id!r} is "
                                     f"{'not in type_defs' if kind is None else repr(kind)}; only a categorical is supported")
                name = "String"
            out.append((column, name))
        return out


class ParameterInfo(BaseModel):
    """A param declared in an older document's `parameters` block; its default fills matching refs."""

    type: t.Any = None
    default_value: t.Any = None


class Node(BaseModel):
    """One node: what it tests, and the node id each branch leads to (`None` is the default row)."""

    model_config = ConfigDict(frozen=True)

    data: NodeData
    children: t.Tuple[t.Optional[str], ...] = ()


class Rule(BaseModel):
    """One rule of a tree: the id of its root node, and its name for `mode="all"` output."""

    model_config = ConfigDict(frozen=True)

    root: str
    name: t.Optional[str] = None


class Tree(SchemaModel):
    """A validated tree, whichever document format it came from.

    `rules` are evaluated in order: with `mode="first_match"` the first rule
    that reaches a leaf other than the default wins; with `mode="all"` every
    rule's result is returned. A v3 tree document is a single rule.
    Get one from a document with `to_tree()`.

    Example::

        tree = load_document(document).to_tree()
        [tree.nodes[rule.root].data for rule in tree.rules]
    """

    name: str = "output"
    mode: t.Literal["first_match", "all"] = "first_match"
    rules: t.Tuple[Rule, ...]
    nodes: t.Dict[str, Node]
    output: TreeOutput = Field(default_factory=TreeOutput)

    @model_validator(mode="after")
    def _structure(self) -> Tree:
        for rule in self.rules:
            if rule.root not in self.nodes:
                raise ValueError(f"rule root {rule.root!r} is not a node")
        indegree = dict.fromkeys(self.nodes, 0)
        for nid, node in self.nodes.items():
            if len(node.children) != arity(node.data):
                raise ValueError(f"node {nid!r} has {len(node.children)} branches, expected {arity(node.data)}")
            for child in filter(None, node.children):
                if child not in self.nodes:
                    raise ValueError(f"node {nid!r} leads to unknown node {child!r}")
                indegree[child] += 1
        # Kahn's algorithm: whatever never reaches indegree 0 sits on a cycle.
        ready = [nid for nid, n in indegree.items() if n == 0]
        while ready:
            for child in filter(None, self.nodes[ready.pop()].children):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if cyclic := sorted(nid for nid, n in indegree.items() if n > 0):
            raise ValueError(f"tree has a cycle through node(s) {cyclic}")
        return self


class BaseTreeDocument(BaseModel):
    """What every tree document format shares: its name, output table, legacy `parameters` block and `to_tree()`.

    Example::

        load_document(document).to_tree()
    """

    name: str = "output"
    output: TreeOutput = Field(default_factory=TreeOutput)
    parameters: t.Dict[str, ParameterInfo] = Field(default_factory=dict)

    # Set by each format's after-validator, so a document is normalised once.
    _tree: Tree = PrivateAttr()

    def to_tree(self) -> Tree:
        """This document as the format-independent `Tree`."""
        return self._tree

    def _param_defaults(self) -> dict[str, t.Any]:
        return {k: p.default_value for k, p in self.parameters.items() if p.default_value is not None}


def with_defaults(obj: t.Any, defaults: dict[str, t.Any]) -> t.Any:
    """`obj` with every local `ParamRef` that has no default given one from `defaults`."""
    if isinstance(obj, ParamRef):
        if obj.default is None and not obj.shared and obj.param in defaults:
            return obj.model_copy(update={"default": defaults[obj.param]})
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_copy(update={k: with_defaults(v, defaults) for k, v in obj.__dict__.items()})
    if isinstance(obj, list):
        return [with_defaults(v, defaults) for v in obj]
    return obj


def add_node(
    nodes: dict[str, Node], nid: str, data: t.Any, children: t.Iterable[t.Optional[str]], defaults: dict[str, t.Any]
) -> None:
    node = Node(data=with_defaults(data.model_copy(update={"id": None}), defaults), children=tuple(children))
    if nodes.setdefault(nid, node) != node:
        raise ValueError(f"two different nodes share the id {nid!r}")
