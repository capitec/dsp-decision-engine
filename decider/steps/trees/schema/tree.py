"""The one tree model every accepted document format normalises into."""
from __future__ import annotations

import typing as t

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from decider.serializable.schema import TStruct
from decider.steps.trees.schema.conditions import SchemaModel
from decider.steps.trees.schema.nodes import NodeData, arity
from decider.steps.values import ParamRef


class TreeOutput(BaseModel):
    """The rows a tree's leaves select: `data[result_idx]`, or `default` for -1.

    `dtypes` names each output column and its type, e.g. `[("band", "String")]`.

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

    @model_validator(mode="after")
    def _rows_have_columns(self) -> TreeOutput:
        declared = set(self.columns)
        rows = [(f"data[{i}]", row) for i, row in enumerate(self.data)]
        if self.default is not None:
            rows.append(("default", self.default))
        for where, row in rows:
            if missing := declared - set(row):
                raise ValueError(f"TreeOutput.{where} is missing declared column(s) {sorted(missing)}")
        return self


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
