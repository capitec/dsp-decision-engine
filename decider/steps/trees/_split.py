"""A wide `mode: "all"` tree as several narrower ones, so no walker call returns hundreds of outputs."""
from __future__ import annotations

import typing as t

from decider.steps.trees.schema import Rule, Tree

# Compile time grows with the square of the outputs one call returns (numba and LLVM both), and a
# tree of 32 outputs compiles once for every other tree of the same output types.
MAX_OUTPUTS = 32


def split(tree: Tree, extra: int, limit: int | None = MAX_OUTPUTS) -> list[Tree]:
    """`tree` cut into trees of at most `limit` outputs (none: whole); `extra` is the output columns a rule adds besides the tree's own."""
    if limit is None:
        return [tree]
    per_rule = len(tree.output.columns) + extra
    size = max(1, limit // max(per_rule, 1))
    if tree.mode != "all" or len(tree.rules) <= size:
        return [tree]
    # A rule keeps the name its columns had in the whole tree.
    rules = [r if r.name else Rule(root=r.root, name=f"rule_{k}") for k, r in enumerate(tree.rules)]
    return [Tree(name=tree.name, mode="all", rules=tuple(chunk), output=tree.output, nodes=_reachable(tree, chunk))
            for chunk in (rules[i:i + size] for i in range(0, len(rules), size))]


def _reachable(tree: Tree, rules: t.Sequence[Rule]) -> dict:
    nodes: dict = {}
    todo = [r.root for r in rules]
    while todo:
        nid = todo.pop()
        if nid not in nodes:
            nodes[nid] = tree.nodes[nid]
            todo.extend(c for c in nodes[nid].children if c is not None)
    return nodes
