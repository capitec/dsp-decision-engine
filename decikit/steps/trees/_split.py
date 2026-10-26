"""A wide `mode: "all"` tree as several narrower ones, so no walker call returns hundreds of outputs."""
from __future__ import annotations

import typing as t

from decikit.steps.trees.schema import Rule, Tree

# The default width of one part, in output columns (rules times columns). Compile time grows with the square of
# the outputs one call returns, so a wide tree is cut into parts; each extra part costs a single record about
# 15-20 us, and every serving process compiles its parts again at start-up. `TreeConfig.part_outputs` sets it per tree.
DEFAULT_PART_OUTPUTS = 64


def split(tree: Tree, extra: int, limit: int | None = DEFAULT_PART_OUTPUTS) -> list[Tree]:
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
