"""A readable outline of a tree document, and its structure as data, for the tree tools."""
from __future__ import annotations

import json

from decider import TreeConfig
from decider._tooling.mcp._docs import problems, read_doc

FLAT = ("flat_rule", "prioritized_flat_rule")
_BRANCHES = "Branch 0 of a condition node is taken when it holds, branch 1 otherwise."


def _val(v) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(map(_val, v)) + "]"
    if isinstance(v, dict) and "param" in v:
        default = "" if v.get("default") is None else f"={json.dumps(v['default'])}"
        return "{" + ("shared:" if v.get("shared") else "") + v["param"] + default + "}"
    return json.dumps(v)


def _feature(f) -> str:
    return f if isinstance(f, str) else f"({f['expression']})"


def condition(c: dict) -> str:
    op = c["op"]
    if op in ("and", "or", "not"):
        parts = [condition(x) for x in c["conditions"]]
        return f"NOT {parts[0]}" if op == "not" else "(" + f" {op.upper()} ".join(parts) + ")"
    f = _feature(c["feature"])
    if op == "between":
        return f"{f} between {_val(c.get('min'))} and {_val(c.get('max'))}"
    if op == "isin":
        return f"{f} in {_val(c['values'])}"
    if op == "string_match":
        flags = "" if c.get("case_sensitive", True) else " ignoring case"
        return f"{f} {c.get('match_type', 'exact')} {_val(c['patterns'])}{flags}"
    if op in ("is_true", "is_false"):
        return f"{f} is {op[3:]}"
    return f"{f} {op} {_val(c['threshold'])}"


def summary(d: dict) -> str:
    kind = d["type"]
    if kind == "leaf":
        return "leaf"
    if kind == "unary":
        return condition(d["condition"])
    if kind == "composite":
        return condition(d)
    return f"{_feature(d['feature'])} {d['op']} cases"


def _band(c: dict, upper_inclusive: bool) -> str:
    lo, hi = c.get("min"), c.get("max")
    left, right = ("(", "]") if upper_inclusive else ("[", ")")
    return f"{left}{'-inf' if lo is None else _val(lo)}, {'inf' if hi is None else _val(hi)}{right}"


def branches(d: dict) -> list[str]:
    """What each branch index of this node means, in index order."""
    kind = d["type"]
    if kind == "leaf":
        return []
    if kind in ("unary", "composite"):
        return ["true", "false"]
    if d["op"] == "ranges":
        cases = [_band(c, d.get("end_logic") == "upper_inclusive") for c in d["conditions"]]
    elif d["op"] == "isin":
        cases = [f"in {_val(c['values'])}" for c in d["conditions"]]
    else:
        cases = [f"{d.get('match_type', 'exact')} {_val(c['patterns'])}" for c in d["conditions"]]
    return cases + ["otherwise"]


def load_tree(path: str):
    """`(doc, TreeConfig, None)` for a tree document that builds, else `(doc, None, problems)`."""
    doc = read_doc(path)
    if doc.get("type") != "tree" or not isinstance(doc.get("tree"), dict):
        return doc, None, [f"{path} is not a tree document (expected {{\"type\": \"tree\", \"tree\": {{...}}}}); use read_config"]
    found = problems(doc)
    return doc, None if found else TreeConfig.load(doc), found


def _row(output, idx: int) -> str:
    if idx == -1:
        return f"default row {json.dumps(output.default or {})}"
    if 0 <= idx < len(output.data):
        return f"row {idx} {json.dumps(output.data[idx])}"
    return f"row {idx} MISSING (output.data has {len(output.data)} rows)"


def tree_view(path: str, node: str | None = None) -> dict:
    """A text outline of the tree (or the subtree under `node`, a node id or a rule name) plus its nodes as data."""
    doc, cfg, found = load_tree(path)
    if cfg is None:
        return {"path": path, "problems": found}
    tree = cfg.tree.to_tree()
    data = {nid: n.data.model_dump(mode="json", exclude_none=True) for nid, n in tree.nodes.items()}
    parents: dict[str, list[str]] = {}
    for nid, n in tree.nodes.items():
        for c in filter(None, n.children):
            parents.setdefault(c, []).append(nid)
    roots = {r.name: r.root for r in tree.rules}
    if node is None:
        starts = [(r.name, r.root) for r in tree.rules]
    elif node in roots:
        starts = [(node, roots[node])]
    elif node in tree.nodes:
        starts = [(None, node)]
    else:
        return {"path": path, "problems": [f"no node or rule {node!r}; node ids: {sorted(tree.nodes)[:30]}, rules: {list(roots)[:30]}"]}

    default = f"default row {json.dumps(tree.output.default or {})}"
    seen: set[str] = set()
    lines: list[str] = []

    def walk(nid: str, indent: str, tag: str) -> None:
        if nid in seen:
            lines.append(f"{indent}{tag}{nid}  (shown above)")
            return
        seen.add(nid)
        d = data[nid]
        if d["type"] == "leaf":
            lines.append(f"{indent}{tag}{nid}: leaf -> {_row(tree.output, d.get('result_idx', -1))}")
            return
        lines.append(f"{indent}{tag}{nid}: {summary(d)}")
        for i, (label, child) in enumerate(zip(branches(d), tree.nodes[nid].children)):
            if child is None:
                lines.append(f"{indent}  [{i} {label}] -> {default}")
            else:
                walk(child, indent + "  ", f"[{i} {label}] ")

    flat = doc["tree"].get("type") in FLAT
    for name, root in starts:
        if flat:
            lines.append(f"rule {name or root}:")
        walk(root, "  " if flat else "", "")
    nodes = [
        {"id": nid, "type": d["type"] + (":" + d["op"] if d["type"] == "cases" else ""), "condition": summary(d),
         "children": [{"branch": i, "label": label, "to": child}
                      for i, (label, child) in enumerate(zip(branches(d), tree.nodes[nid].children))],
         "parents": parents.get(nid, []),
         **({"result": _row(tree.output, d.get("result_idx", -1))} if d["type"] == "leaf" else {})}
        for nid, d in data.items() if nid in seen
    ]
    unreachable = [n["id"] for n in doc["tree"].get("nodes", []) if n["id"] not in tree.nodes]
    if flat:
        order = ("Rules are tried in the order listed; with mode first_match the first rule whose leaf is not the default wins, "
                 "so order matters." if tree.mode == "first_match" else
                 "mode all: every rule runs and writes <rule>.<column>.")
        head = f"{doc.get('name', tree.name)}: {doc['tree']['type']} with {len(tree.rules)} rule(s), mode {tree.mode}. {order} {_BRANCHES}"
    else:
        head = f"{doc.get('name', tree.name)}: v3 tree, {len(tree.nodes)} nodes. {_BRANCHES} An empty branch answers with the {default}."
    if unreachable:
        lines.append(f"unreachable (never run; they have no path from the root): {unreachable}")
    return {"path": path, "format": doc["tree"].get("type", "v3-tree"), "mode": tree.mode,
            "outline": "\n".join([head, *lines]), "nodes": nodes, "unreachable": unreachable,
            "rules": [{"name": r.name, "root": r.root} for r in tree.rules] if flat else [],
            "output": {"default": tree.output.default, "rows": len(tree.output.data)}, "problems": []}
