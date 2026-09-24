"""A tree's paths, numbered the way the compiled walker numbers them, for `TreeConfig.trace_output`."""
from __future__ import annotations

import typing as t

from decider.steps.trees.schema import Tree

# ponytail: every distinct path is a string choice of the trace column; a DAG with more paths than this is refused.
MAX_PATHS = 100_000
SEP = ">"


def counts(tree: Tree, order: t.Sequence[str]) -> dict[str, int]:
    """How many paths lead from each node to an end (a leaf, or a missing branch); `order` is post-order."""
    n: dict[str, int] = {}
    for nid in order:
        kids = tree.nodes[nid].children
        n[nid] = sum(1 if c is None else n[c] for c in kids) if kids else 1
    total = sum(n[r.root] for r in tree.rules)
    if total > MAX_PATHS:
        raise ValueError(f"trace_output: the tree has {total} distinct paths, more than {MAX_PATHS}; shared "
                         "nodes multiply paths, so trace a smaller tree or use path_output (the leaf) instead")
    return n


def paths(tree: Tree, order: t.Sequence[str]) -> dict[str, list[str]]:
    """Each node's paths to an end as `>`-joined node ids, path `k` being the one the walker numbers `k`."""
    found: dict[str, list[str]] = {}
    for nid in order:
        kids = tree.nodes[nid].children
        found[nid] = [nid + SEP + p if c is not None else nid
                      for c in kids for p in (found[c] if c is not None else [""])] if kids else [nid]
    return found
