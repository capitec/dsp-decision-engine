"""Edits to tree documents: v3 nodes and edges, the output table, and flat rules. Every write is validated by `save_doc`."""
from __future__ import annotations

from decider._tooling.mcp._docs import read_doc, save_doc
from decider._tooling.mcp._tree_view import FLAT, branches, summary, tree_view

__all__ = ["tree_add_node", "tree_add_rule", "tree_remove_node", "tree_remove_rule", "tree_set_node",
           "tree_set_output", "tree_view"]

_OPS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le", "==": "eq", "!=": "ne", "between": "between", "isin": "in",
        "string_match": "match", "is_true": "true", "is_false": "false"}
_BAD_DATA = ('node data needs "type": "leaf" (with result_idx), "unary" (with condition), "composite" (with op and '
             'conditions) or "cases" (with op ranges|string_match|isin, feature and conditions)')


def _refuse(path: str, *why: str) -> dict:
    return {"saved": False, "path": path, "problems": list(why)}


def _open(path: str, kind: str):
    """`(doc, tree, None)` for a tree document of `kind` ("v3", "flat" or "any"), else `(None, None, refusal)`."""
    doc = read_doc(path)
    tree = doc.get("tree")
    if doc.get("type") != "tree" or not isinstance(tree, dict):
        return None, None, _refuse(path, "not a tree document ({\"type\": \"tree\", \"tree\": {...}}); use patch_config for other types")
    flat = tree.get("type") in FLAT
    if kind == "v3" and flat:
        return None, None, _refuse(path, f"this tree is {tree['type']} (rules nest their own branches), so it has no node edges; "
                                         "use tree_add_rule / tree_remove_rule to change rules, or patch_config for edits inside a rule")
    if kind == "flat" and not flat:
        return None, None, _refuse(path, "this is a v3 node/edge tree, not flat rules; use tree_add_node / tree_set_node / tree_remove_node")
    return doc, tree, None


def _indexes(edge: dict) -> list[int]:
    i = edge["data"]["sourceIndex"]
    return i if isinstance(i, list) else [i]


def _arity(data: dict) -> int:
    kind = data.get("type")
    if kind == "leaf":
        return 0
    if kind in ("unary", "composite"):
        return 2
    if kind == "cases" and isinstance(data.get("conditions"), list):
        return len(data["conditions"]) + 1
    raise ValueError(_BAD_DATA)


def _leaf_row_problem(tree: dict, data: dict) -> str | None:
    idx, rows = data.get("result_idx", -1), len(tree.get("output", {}).get("data", []))
    if data.get("type") == "leaf" and not -1 <= idx < rows:
        return (f"leaf result_idx {idx} but output.data has {rows} row(s); add the row first with "
                "tree_set_output(path, values), which returns its result_idx, or use -1 for the default row")
    return None


def _new_id(data: dict, taken: set[str]) -> str:
    kind = data["type"]
    if kind == "leaf":
        base = f"leaf_{data.get('result_idx', -1)}".replace("-1", "default")
    elif kind == "unary":
        c = data["condition"]
        base = f"{c['feature'] if isinstance(c['feature'], str) else 'expr'}_{_OPS.get(c['op'], c['op'])}"
    elif kind == "composite":
        base = f"{data['op']}_of_{len(data['conditions'])}"
    else:
        f = data["feature"]
        base = f"{f if isinstance(f, str) else 'expr'}_cases"
    nid, n = base, 1
    while nid in taken:
        n += 1
        nid = f"{base}_{n}"
    return nid


def _listing(data: dict) -> str:
    return ", ".join(f"{i} ({label})" for i, label in enumerate(branches(data))) or "none (a leaf has no branches)"


def tree_add_node(path: str, parent: str | None, branch: int, data: dict, id: str | None = None) -> dict:
    """Add a v3 node under `parent` at `branch` (the edge's sourceIndex); `parent=None` adds the root of an empty tree."""
    doc, tree, err = _open(path, "v3")
    if err:
        return err
    nodes = {n["id"]: n for n in tree["nodes"]}
    try:
        _arity(data)
    except ValueError as e:
        return _refuse(path, str(e))
    if bad := _leaf_row_problem(tree, data):
        return _refuse(path, bad)
    if id is not None and id in nodes:
        return _refuse(path, f"node id {id!r} is taken; pass another id, or leave id out for a generated one")
    if parent is None:
        if nodes:
            return _refuse(path, "the tree already has a root; pass parent=<node id> and branch=<index>. "
                                 "To replace the whole tree, remove its root with tree_remove_node first")
    elif parent not in nodes:
        return _refuse(path, f"no node {parent!r}; node ids: {sorted(nodes)[:30]}. Call tree_view to see them")
    else:
        labels = branches(nodes[parent]["data"])
        if not 0 <= branch < len(labels):
            return _refuse(path, f"node {parent!r} has no branch {branch}; its branches are {_listing(nodes[parent]['data'])}")
        for e in tree["edges"]:
            if e["source"] == parent and branch in _indexes(e):
                return _refuse(path, f"branch {branch} ({labels[branch]}) of {parent!r} already leads to {e['target']!r}; "
                                     f"call tree_remove_node(path, {e['target']!r}) first to replace it, or "
                                     f"tree_set_node(path, {e['target']!r}, data) to change that node in place")
    nid = id or _new_id(data, set(nodes))
    tree["nodes"].append({"id": nid, "data": data})
    if parent is not None:
        tree["edges"].append({"source": parent, "target": nid, "data": {"sourceIndex": branch}})
    return {**save_doc(path, doc), "id": nid}


def tree_set_node(path: str, node_id: str, data: dict) -> dict:
    """Replace a v3 node's data (its type and condition); its children stay on the same branch indexes."""
    doc, tree, err = _open(path, "v3")
    if err:
        return err
    node = next((n for n in tree["nodes"] if n["id"] == node_id), None)
    if node is None:
        return _refuse(path, f"no node {node_id!r}; node ids: {sorted(n['id'] for n in tree['nodes'])[:30]}")
    try:
        new = _arity(data)
    except ValueError as e:
        return _refuse(path, str(e))
    if bad := _leaf_row_problem(tree, data):
        return _refuse(path, bad)
    cut = [(i, e["target"]) for e in tree["edges"] if e["source"] == node_id for i in _indexes(e) if i >= new]
    if cut:
        return _refuse(path, f"the new data has {new} branch(es) but {node_id!r} has children on branch(es) "
                             f"{[f'{i} -> {t}' for i, t in cut]}; remove them first with tree_remove_node, or keep a type with more branches")
    old = _arity(node["data"])
    node["data"] = data
    result = save_doc(path, doc)
    if old != new and result["saved"]:
        result["note"] = (f"{node_id!r} went from {old} to {new} branches; children keep their branch index, "
                          f"and a cases node's last branch is always otherwise. Check with tree_view")
    return result


def tree_remove_node(path: str, node_id: str) -> dict:
    """Remove a v3 node, its incoming edges, and the descendants nothing else leads to; a node with another parent stays."""
    doc, tree, err = _open(path, "v3")
    if err:
        return err
    ids = {n["id"] for n in tree["nodes"]}
    if node_id not in ids:
        return _refuse(path, f"no node {node_id!r}; node ids: {sorted(ids)[:30]}")
    removed, grew = {node_id}, True
    while grew:
        grew = False
        for child in {e["target"] for e in tree["edges"] if e["source"] in removed} - removed:
            if all(e["source"] in removed for e in tree["edges"] if e["target"] == child):
                removed.add(child)
                grew = True
    if removed == ids:
        return _refuse(path, "that would leave the tree empty; change the root with tree_set_node, or replace the whole document with write_config")
    freed = [{"parent": e["source"], "branch": i} for e in tree["edges"]
             if e["target"] in removed and e["source"] not in removed for i in _indexes(e)]
    kept = sorted({e["target"] for e in tree["edges"] if e["source"] in removed and e["target"] not in removed})
    tree["nodes"] = [n for n in tree["nodes"] if n["id"] not in removed]
    tree["edges"] = [e for e in tree["edges"] if e["source"] not in removed and e["target"] not in removed]
    tree["subtrees"] = [s for s in tree.get("subtrees", []) if s.get("id") not in removed]
    result = save_doc(path, doc)
    if not result["saved"]:
        return result
    return {**result, "removed": sorted(removed), "kept_shared": kept,
            "now_default": freed, "note": "an emptied branch answers with the default row; add a node there with tree_add_node"}


def _columns(dtypes) -> list[str]:
    return list(dtypes) if isinstance(dtypes, dict) else [c for c, _ in dtypes]


def _guess(v) -> str | None:
    return next((n for t, n in ((bool, "Boolean"), (int, "Int64"), (float, "Float64"), (str, "String")) if isinstance(v, t)), None)


def tree_set_output(path: str, values: dict, index: int | None = None, dtypes: dict[str, str] | None = None) -> dict:
    """Append a result row to `output.data` (returns its `result_idx`) or replace row `index`; `index=-1` sets the default row.

    A column the output has no dtype for needs `dtypes`, e.g. `dtypes={"reason": "String"}`.
    """
    doc, tree, err = _open(path, "any")
    if err:
        return err
    out = tree.setdefault("output", {})
    rows = out.setdefault("data", [])
    if index is not None and not -1 <= index < len(rows):
        return _refuse(path, f"index {index} is out of range: output.data has {len(rows)} row(s) (0..{len(rows) - 1}, or -1 for the default); "
                             "leave index out to append")
    known = _columns(out.get("dtypes", []))
    dtypes = dtypes or {}
    new = [c for c in values if c not in known]
    if missing := [c for c in new if c not in dtypes]:
        guess = {c: _guess(values[c]) for c in missing}
        return _refuse(path, f"column(s) {missing} have no dtype in output.dtypes ({known}); call again with "
                             f"dtypes={guess} (use another type name if the guess is wrong)")
    for c in new:
        if isinstance(out.get("dtypes"), dict):
            out["dtypes"][c] = dtypes[c]
        else:
            out.setdefault("dtypes", []).append([c, dtypes[c]])
    if index == -1:
        out["default"] = values
    elif index is None:
        rows.append(values)
    else:
        rows[index] = values
    return {**save_doc(path, doc), "result_idx": len(rows) - 1 if index is None else index, "added_dtypes": new}


def tree_add_rule(path: str, rule: dict, at: int | None = None) -> dict:
    """Insert a flat rule `{"meta": {"name": ...}, "rule": {...}}` at position `at` (default the end) of a prioritized_flat_rule tree."""
    doc, tree, err = _open(path, "flat")
    if err:
        return err
    if tree["type"] != "prioritized_flat_rule":
        return _refuse(path, "a flat_rule document holds exactly one rule; to hold several, change tree.type to "
                             "prioritized_flat_rule and move the rule into a \"rules\" list (patch_config), then add rules")
    name = (rule.get("meta") or {}).get("name")
    rules = tree["rules"]
    if not name:
        return _refuse(path, 'the rule needs a name: {"meta": {"name": "..."}, "rule": {...}}')
    if name in [(r.get("meta") or {}).get("name") for r in rules]:
        return _refuse(path, f"a rule named {name!r} exists; pick another name, or tree_remove_rule(path, {name!r}) first to replace it")
    at = len(rules) if at is None else at
    if not 0 <= at <= len(rules):
        return _refuse(path, f"at={at} is out of range: there are {len(rules)} rule(s), so 0..{len(rules)}")
    rules.insert(at, rule)
    result = save_doc(path, doc)
    if not result["saved"]:
        return result
    mode = tree.get("mode", "first_match")
    notes = ["mode first_match: earlier rules win, so the position matters" if mode == "first_match" else "mode all: every rule runs"]
    if at < len(rules) - 1:
        notes.append("nodes without an id get ids from their rule's position, so the later rules' unnamed node ids shifted")
    return {**result, "index": at, "rules": len(rules), "mode": mode, "note": "; ".join(notes)}


def tree_remove_rule(path: str, name: str) -> dict:
    """Remove the flat rule named `name` from a prioritized_flat_rule tree; the other rules keep their order."""
    doc, tree, err = _open(path, "flat")
    if err:
        return err
    rules = tree.get("rules", [])
    names = [(r.get("meta") or {}).get("name") for r in rules]
    if name not in names:
        return _refuse(path, f"no rule named {name!r}; rules: {names[:30]}{' ...' if len(names) > 30 else ''}")
    i = names.index(name)
    del rules[i]
    return {**save_doc(path, doc), "removed_index": i, "rules": len(rules)}
