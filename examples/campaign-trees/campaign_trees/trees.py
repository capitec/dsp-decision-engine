"""Checks on a tree document before it is published, and the diff between two versions of it."""
from __future__ import annotations

from decikit import TreeConfig


def problems(tree: TreeConfig) -> list[str]:
    """What is wrong with a tree that still loads: nodes nothing reaches, and branches that lead nowhere.

    A branch without an edge ends in the default row, but a campaign tree must always end in a leaf that says
    something, so both count as problems.
    """
    listed = {node.id for node in tree.tree.nodes}
    reached = tree.nodes
    return ([f"node {n} is not reachable from the root" for n in sorted(listed - set(reached))]
            + [f"node {n} has a branch that leads nowhere" for n, node in reached.items() if None in node.children])


def changed_nodes(old: TreeConfig, new: TreeConfig) -> dict[str, list[str]]:
    """The node ids added, removed and changed between two versions of a tree (a threshold counts as a change).

    Ids are part of the document, so node-level history survives a new version: only the nodes listed here start a
    new history.
    """
    before, after = old.nodes, new.nodes
    return {"added": sorted(after.keys() - before.keys()), "removed": sorted(before.keys() - after.keys()),
            "changed": sorted(n for n in before.keys() & after.keys() if before[n] != after[n])}
