"""`TreeConfig.path_columns`: the tree unfolded so every route ends at a leaf of its own, whose row holds the route's columns."""
from __future__ import annotations

import typing as t

from decider.steps.trees._trace import MAX_PATHS
from decider.steps.trees.schema import LeafNode, Node, Rule, Tree, TreeOutput

__all__ = ["Route", "Visit"]

_DTYPES = {bool: "Boolean", int: "Int64", float: "Float64", str: "String"}


class Visit(t.NamedTuple):
    """One node a route passes: its id, what it tests, the branch taken, and where that branch leads (`None`: the default row)."""

    node_id: str
    node: t.Any
    branch: int
    child: t.Optional[str]


class Route(t.NamedTuple):
    """One way through a tree: the nodes passed, root first, and the output row it ends at (-1 for the default).

    Example::

        def depth(route: Route, tree: Tree) -> dict:
            return {"depth": len(route.visits), "hit": route.result_idx >= 0}
    """

    visits: t.Tuple[Visit, ...]
    result_idx: int


def with_route_columns(tree: Tree, fn: t.Callable[[Route, Tree], dict]) -> Tree:
    if tree.mode == "first_match" and len(tree.rules) > 1:
        raise ValueError("path_columns: when no rule of a first-match tree matches, the default row answers, "
                         "which no one route owns; use mode 'all' or a single rule")
    nodes: dict[str, t.Optional[Node]] = {}
    rows: list[dict] = []
    extras: list[dict] = []

    def fresh(base: str, own: bool = False) -> str:
        k, name = 1, base
        while name in nodes or (name in tree.nodes and not (own and name == base)):
            k += 1
            name = f"{base}~{k}"
        nodes[name] = None
        return name

    def end(nid: str, visits: list, result_idx: int) -> str:
        if len(rows) >= MAX_PATHS:
            raise ValueError(f"path_columns: the tree has more than {MAX_PATHS} routes; shared nodes multiply them")
        extras.append(fn(Route(tuple(visits), result_idx), tree))
        rows.append(dict(tree.output.default or {}) if result_idx < 0 else dict(tree.output.data[result_idx]))
        nodes[nid] = Node(data=LeafNode(result_idx=len(rows) - 1))
        return nid

    # ponytail: recursion depth follows tree depth, as the document's own validation does.
    def walk(nid: str, visits: list) -> str:
        node = tree.nodes[nid]
        rid = fresh(nid, own=True)
        if isinstance(node.data, LeafNode):
            return end(rid, visits, node.data.result_idx)
        kids = []
        for i, child in enumerate(node.children):
            step = [*visits, Visit(nid, node.data, i, child)]
            kids.append(end(fresh(f"{rid}.{i}"), step, -1) if child is None else walk(child, step))
        nodes[rid] = Node(data=node.data, children=tuple(kids))
        return rid

    rules = tuple(Rule(root=walk(r.root, []), name=r.name) for r in tree.rules)
    columns = list(extras[0]) if extras else []
    for extra in extras:
        if list(extra) != columns:
            raise ValueError(f"path_columns: every route must give the same columns, got {columns} and {list(extra)}")
    if clash := set(columns) & set(tree.output.columns):
        raise ValueError(f"path_columns: {sorted(clash)} are already output columns of the tree")
    dtypes = list(tree.output.dtypes.items()) if isinstance(tree.output.dtypes, dict) else list(tree.output.dtypes)
    dtypes += [(c, _dtype(c, [e[c] for e in extras])) for c in columns]
    output = TreeOutput(data=[{**row, **extra} for row, extra in zip(rows, extras)], default=tree.output.default,
                        dtypes=dtypes, type_defs=tree.output.type_defs)
    return Tree(name=tree.name, mode=tree.mode, rules=rules, nodes=nodes, output=output)


def _dtype(column: str, values: list) -> str:
    kinds = {type(v) for v in values if v is not None}
    if kinds == {int, float}:
        return "Float64"
    if len(kinds) != 1 or (kind := next(iter(kinds))) not in _DTYPES:
        raise ValueError(f"path_columns: column {column!r} needs one type of bool, int, float or str "
                         f"(or None), got {sorted(k.__name__ for k in kinds) or 'only None'}")
    return _DTYPES[kind]
