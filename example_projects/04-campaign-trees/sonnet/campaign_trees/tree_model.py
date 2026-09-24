"""Tree authoring, node identity (spec 04 §5.4.3) and path capture (§5.4.1) -- the layer
underneath `decider.steps.trees.TreeConfig`, which this project's pipeline uses for the
actual leaf/output computation (see `pipeline.py`).

**Why a layer exists at all.** `TreeConfig`'s `path_output` records only the id of the leaf
that answered (its own docstring: "the id of the leaf that answered") -- not the ordered
sequence of nodes visited. Spec 04 §5.4.1(a)-(d) requires the traversal itself, not just its
terminus, and requires it renderable later "without re-running the tree and without the
original run environment" (§5.4.1(d)). Neither is something `TreeConfig` gives us. This
module supplies both, from one authored tree, as two views that cannot drift from each
other's *arithmetic* (see below) even though they are two code paths for *traversal*:

  - `to_v3_document(tree, ...)` -- the JSON `TreeConfig.load()` accepts, for evaluation
    inside a `decider` pipeline (fast, vectorised, the thing that actually runs at 400M
    evaluations/cycle).
  - `walk(tree, record, ...)` -- a pure-Python traversal of the *same* structure, used both
    to capture the path at evaluation time and to re-render it later from stored feature
    values alone (§5.4.1(d), §9.1), with no decider dependency.

Both consume the same `term["expr"]` strings through `decider.steps.expr` (used by
`to_v3_document` as a tree `ComputedFeature` and by `walk` directly), so a two-feature
comparison is evaluated by the same arithmetic in both places -- only the *traversal engine*
differs. `tests/test_tree_model.py::test_walker_agrees_with_treeconfig_leaf` proves the two
traversal engines agree over a battery of synthetic records, which is the check this
divergence risk earns (see NOTES.md "Framework friction").

**Node identity (§5.4.3).** A tree is authored as `{temp_id: node}` with a declared root,
using an analyst's own (unstable, export-order-dependent) ids. `build_tree()` computes a
content-derived `node_key` for every node and rewrites the tree keyed by it -- never by the
temp id, which is discarded once built. `node_key` depends on exactly two things, per
§5.4.3 item 1 ("same conditions... same position in the tree's decision logic"):

  1. the node's own condition, canonicalised (op, feature/expression, threshold or set,
     AND/OR combination -- term order-independent, so reordering `and` clauses on a
     re-export does not change identity);
  2. its *position*: the set of (parent `node_key`, branch direction) pairs for every edge
     that reaches it. A node with two parents (the campaign-23 worked example's node 6,
     reachable from both node 3 and node 4 -- §5.3.3's own note) keeps ONE identity; a node
     downstream of a changed ancestor legitimately gets a new one, because its logical
     position *did* change, which is the point §5.4.3 item 1 makes about "position in the
     tree's decision logic" rather than raw graph distance from the root.

This makes identity independent of export order and of unrelated parts of the tree (§5.4.3
items 2-3: hashing, not the temp id's insertion order, and only nodes on the changed node's
own downstream paths are affected) without needing the modelling tool to guarantee anything
about how it numbers a re-fit (change scenario 2). It is a design choice, not the only
possible one -- see NOTES.md "What I built" for the trade-off (it does not detect two
conditions that are logically equivalent but textually different as "the same test": spec
04 §13 Q5 flags this as an open, reviewable question, not a solved one).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from decider.steps import expr as decider_expr

ParamThreshold = tuple  # ("param", name, default)


# ---------------------------------------------------------------------------
# Authoring: plain dicts, temp-id keyed. See module docstring for shapes.
# ---------------------------------------------------------------------------

def param_threshold(name: str, default: float) -> ParamThreshold:
    """An overlay-tunable threshold (§5.3.4 volume dial / cut-off shift): carried into the v3
    document as `{"param": name, "default": default}` and resolved the same way by `walk`."""
    return ("param", name, default)


def _is_param(v: Any) -> bool:
    return isinstance(v, tuple) and len(v) == 3 and v[0] == "param"


def _resolve(v: Any, overrides: dict[str, float]) -> Any:
    return overrides.get(v[1], v[2]) if _is_param(v) else v


# ---------------------------------------------------------------------------
# Canonicalisation, for node_key
# ---------------------------------------------------------------------------

def _canon_value(v: Any) -> str:
    return f"param:{v[1]}={v[2]}" if _is_param(v) else repr(v)


def _canon_term(term: dict) -> str:
    feature = term.get("expr") or term["feature"]
    op = term["op"]
    if op == "between":
        val = f"[{_canon_value(term.get('min'))},{_canon_value(term.get('max'))}]"
    elif op == "isin":
        val = "{" + ",".join(sorted(str(x) for x in term["values"])) + "}"
    elif op in ("is_true", "is_false"):
        val = ""
    else:
        val = _canon_value(term["threshold"])
    return f"{feature}{op}{val}"


def canon_node_self(node: dict) -> str:
    """The node's own content, independent of position -- what §5.4.3 item 1's "same
    conditions" half means. Exposed for the node identity map's human-readable diff."""
    if node["kind"] == "leaf":
        return "leaf:" + json.dumps(node["output"], sort_keys=True, default=str)
    terms_canon = sorted(_canon_term(t) for t in node["terms"])
    logic = node.get("logic") or ""
    return f"test:{logic}:" + "|".join(terms_canon)


# ---------------------------------------------------------------------------
# Building: temp-id tree -> node_key-keyed Tree
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tree:
    name: str
    root: str
    nodes: dict[str, dict]                    # node_key -> node (true/false children are node_keys)
    position_signature: dict[str, frozenset]   # node_key -> frozenset of (parent_node_key, direction)
    condition_text: dict[str, str]             # node_key -> canon_node_self(node), kept for the identity map


def _validate_structure(nodes: dict, root: str) -> None:
    if root not in nodes:
        raise ValueError(f"root {root!r} is not a node")
    seen: set[str] = set()
    stack = [root]
    while stack:
        tid = stack.pop()
        if tid in seen:
            continue
        seen.add(tid)
        node = nodes.get(tid)
        if node is None:
            raise ValueError(f"node {tid!r} is referenced but not defined")
        if node["kind"] == "test":
            for child in (node["true"], node["false"]):
                if child == tid:
                    raise ValueError(f"node {tid!r} is its own child (cycle)")
                stack.append(child)
    unreachable = set(nodes) - seen
    if unreachable:
        raise ValueError(f"unreachable node(s) (§5.9 item 2): {sorted(unreachable)}")
    # cycle check beyond immediate self-loop: a DFS with a recursion-safe visited set that
    # would never terminate on a real cycle is caught by Python's own recursion limit in
    # _compute_keys below; this pass additionally rejects the common "listed but never
    # reached from root" case with a clear message.


def _compute_keys(nodes: dict, root: str) -> tuple[dict[str, str], dict[str, frozenset]]:
    incoming: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for tid, n in nodes.items():
        if n["kind"] == "test":
            incoming[n["true"]].append((tid, 0))
            incoming[n["false"]].append((tid, 1))

    keys: dict[str, str] = {}
    sigs: dict[str, frozenset] = {}
    in_progress: set[str] = set()

    def key_of(tid: str) -> str:
        if tid in keys:
            return keys[tid]
        if tid in in_progress:
            raise ValueError(f"tree has a cycle through node {tid!r}")
        in_progress.add(tid)
        parents = incoming.get(tid, [])
        if not parents:
            sig = frozenset()
        else:
            sig = frozenset((key_of(p), d) for p, d in parents)
        digest = hashlib.sha256(f"{canon_node_self(nodes[tid])}::{sorted(sig)}".encode()).hexdigest()[:16]
        keys[tid] = digest
        sigs[tid] = sig
        in_progress.discard(tid)
        return digest

    for tid in nodes:
        key_of(tid)
    return keys, sigs


def build_tree(nodes: dict, root: str, name: str) -> Tree:
    """Validate `nodes` (temp-id keyed, per the module docstring) and rewrite it keyed by
    content-derived `node_key`. Raises `ValueError` naming the node and the problem for every
    structural defect (§5.9 items 1-2; §10 item 2)."""
    _validate_structure(nodes, root)
    keys, sigs = _compute_keys(nodes, root)
    rekeyed: dict[str, dict] = {}
    condition_text: dict[str, str] = {}
    position_signature: dict[str, frozenset] = {}
    for tid, n in nodes.items():
        k = keys[tid]
        condition_text[k] = canon_node_self(n)
        position_signature[k] = sigs[tid]
        if n["kind"] == "leaf":
            rekeyed[k] = {"kind": "leaf", "output": n["output"]}
        else:
            rekeyed[k] = {
                "kind": "test", "logic": n.get("logic"), "terms": n["terms"],
                "true": keys[n["true"]], "false": keys[n["false"]],
            }
    return Tree(name=name, root=keys[root], nodes=rekeyed,
                position_signature=position_signature, condition_text=condition_text)


# ---------------------------------------------------------------------------
# Validation beyond structure (§5.9, subset -- see NOTES.md "What I left out")
# ---------------------------------------------------------------------------

_NUMERIC_OPS = {">=", "<=", ">", "<", "==", "!="}


def _tighten_interval(lo: float, hi: float, term: dict) -> tuple[float, float]:
    op, thr = term["op"], term.get("threshold")
    thr = _resolve(thr, {}) if thr is not None else None
    if op == ">=":
        lo = max(lo, thr)
    elif op == ">":
        lo = max(lo, thr + 1e-9)
    elif op == "<=":
        hi = min(hi, thr)
    elif op == "<":
        hi = min(hi, thr - 1e-9)
    elif op == "between":
        if term.get("min") is not None:
            lo = max(lo, _resolve(term["min"], {}))
        if term.get("max") is not None:
            hi = min(hi, _resolve(term["max"], {}))
    return lo, hi


def _check_contradictions(tree: Tree, path_errors: list[str]) -> None:
    """§5.9 item 3, over the true-side of single-feature numeric AND-only tests (the
    tractable subset of a general constraint solver -- an OR node or a node testing two
    different features against each other is not interval-representable and is skipped;
    see NOTES.md "What I left out")."""

    def walk(node_key: str, bounds: dict[str, tuple[float, float]]) -> None:
        node = tree.nodes[node_key]
        if node["kind"] == "leaf":
            return
        for direction, child in ((0, node["true"]), (1, node["false"])):
            child_bounds = dict(bounds)
            if node.get("logic") in (None, "and") and direction == 0:
                for term in node["terms"]:
                    if term["op"] in _NUMERIC_OPS | {"between"} and "expr" not in term:
                        feat = term["feature"]
                        lo, hi = child_bounds.get(feat, (float("-inf"), float("inf")))
                        new_lo, new_hi = _tighten_interval(lo, hi, term)
                        if new_lo > new_hi:
                            path_errors.append(
                                f"node {child!r}: contradictory condition on {feat!r} "
                                f"(narrowed to [{new_lo}, {new_hi}])")
                        child_bounds[feat] = (new_lo, new_hi)
            walk(child, child_bounds)

    walk(tree.root, {})


def validate_tree(tree: Tree, *, feature_registry: dict[str, str], prohibited_features: set[str]) -> list[str]:
    """Returns every problem found (empty = passes). Never raises -- publication (§5.9)
    reports every problem at once, named by node, rather than failing on the first."""
    problems: list[str] = []
    for key, node in tree.nodes.items():
        if node["kind"] != "test":
            continue
        for term in node["terms"]:
            if "expr" in term:
                try:
                    decider_expr.parse(term["expr"])
                except Exception as e:  # noqa: BLE001 -- report, don't crash publication
                    problems.append(f"node {key!r}: bad expression {term['expr']!r}: {e}")
                continue
            feat = term["feature"]
            if feat not in feature_registry:
                problems.append(f"node {key!r}: references unknown feature {feat!r} (§5.9 item 4)")
            if feat in prohibited_features:
                problems.append(f"node {key!r}: references prohibited feature {feat!r} (§5.9 item 6)")
    _check_contradictions(tree, problems)
    return problems


# ---------------------------------------------------------------------------
# to_v3_document: Tree -> the JSON TreeConfig.load() accepts
# ---------------------------------------------------------------------------

def _param_or_literal(v: Any) -> Any:
    return {"param": v[1], "default": v[2]} if _is_param(v) else v


def _term_to_condition(term: dict) -> dict:
    feature = {"type": "computed", "expression": term["expr"]} if "expr" in term else term["feature"]
    op = term["op"]
    if op == "between":
        d = {"type": "unary", "op": "between", "feature": feature}
        if term.get("min") is not None:
            d["min"] = _param_or_literal(term["min"])
        if term.get("max") is not None:
            d["max"] = _param_or_literal(term["max"])
        return d
    if op == "isin":
        return {"type": "unary", "op": "isin", "feature": feature, "values": term["values"]}
    if op in ("is_true", "is_false"):
        return {"type": "unary", "op": op, "feature": feature}
    return {"type": "unary", "op": op, "feature": feature, "threshold": _param_or_literal(term["threshold"])}


def _node_to_condition(node: dict) -> dict:
    if len(node["terms"]) == 1 and not node.get("logic"):
        return _term_to_condition(node["terms"][0])
    return {"type": "composite", "op": node["logic"], "conditions": [_term_to_condition(t) for t in node["terms"]]}


_PY_TO_DTYPE = {bool: "Boolean", int: "Int64", float: "Float64", str: "String"}


def _infer_dtypes(rows: list[dict]) -> list[list[str]]:
    cols: dict[str, str] = {}
    for row in rows:
        for k, v in row.items():
            if k not in cols and v is not None:
                cols[k] = _PY_TO_DTYPE.get(type(v), "String")
    return [[k, v] for k, v in cols.items()]


def to_v3_document(tree: Tree, *, path_output: str = "leaf_key") -> dict:
    nodes_json: list[dict] = []
    edges_json: list[dict] = []
    output_rows: list[dict] = []
    row_index: dict[tuple, int] = {}

    def output_row_idx(output: dict) -> int:
        key = tuple(sorted(output.items()))
        if key not in row_index:
            row_index[key] = len(output_rows)
            output_rows.append(output)
        return row_index[key]

    for node_key, node in tree.nodes.items():
        if node["kind"] == "leaf":
            idx = output_row_idx(node["output"])
            nodes_json.append({"id": node_key, "data": {"type": "leaf", "result_idx": idx}})
        else:
            cond = _node_to_condition(node)
            data = {"type": "unary", "condition": cond} if cond["type"] == "unary" else cond
            nodes_json.append({"id": node_key, "data": data})
            edges_json.append({"source": node_key, "target": node["true"], "data": {"sourceIndex": [0]}})
            edges_json.append({"source": node_key, "target": node["false"], "data": {"sourceIndex": [1]}})

    return {
        "type": "tree", "name": tree.name, "path_output": path_output,
        "tree": {
            "nodes": nodes_json, "edges": edges_json,
            "output": {"data": output_rows, "dtypes": _infer_dtypes(output_rows)},
        },
    }


# ---------------------------------------------------------------------------
# walk: pure-Python traversal, for path capture (§5.4.1) and on-demand rendering (§5.4.1(d), §9.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TermEval:
    rendered: str
    held: bool | None  # None = unknown (null input), three-valued per TreeConfig's default null_handling


@dataclass(frozen=True)
class PathStep:
    node_key: str
    kind: str                    # "test" | "leaf"
    terms: tuple[TermEval, ...] = ()
    logic: str | None = None
    held: bool | None = None     # the node's own combined verdict (None only if the "otherwise" branch was taken)
    direction: int | None = None  # 0 (true) or 1 (false) -- which branch this step's node sent evaluation down
    output: dict | None = None   # leaf steps only


def _feature_value(term: dict, record: dict) -> Any:
    if "expr" in term:
        return decider_expr.parse(term["expr"]).evaluate(record)
    return record.get(term["feature"])


def _eval_term(term: dict, record: dict, overrides: dict[str, float]) -> TermEval:
    label = term.get("expr") or term["feature"]
    fval = _feature_value(term, record)
    op = term["op"]
    if fval is None:
        return TermEval(f"{label} = null", None)
    if op == "is_true":
        return TermEval(f"{label} = {fval}", bool(fval))
    if op == "is_false":
        return TermEval(f"{label} = {fval}", not bool(fval))
    if op == "isin":
        values = term["values"]
        return TermEval(f"{label} = {fval!r} (in {{{','.join(map(str, values))}}})", fval in values)
    if op == "between":
        lo = _resolve(term.get("min"), overrides) if term.get("min") is not None else None
        hi = _resolve(term.get("max"), overrides) if term.get("max") is not None else None
        held = (lo is None or fval >= lo) and (hi is None or fval <= hi)
        return TermEval(f"{label} = {fval} (between {lo} and {hi})", held)
    thr = _resolve(term["threshold"], overrides)
    cmp = {">=": fval >= thr, "<=": fval <= thr, ">": fval > thr, "<": fval < thr,
           "==": fval == thr, "!=": fval != thr}[op]
    return TermEval(f"{label} = {fval} ({op} {thr})", cmp)


def _combine(logic: str | None, helds: list[bool | None]) -> bool | None:
    if logic is None:
        return helds[0]
    if logic == "and":
        if any(h is False for h in helds):
            return False
        if any(h is None for h in helds):
            return None
        return True
    if logic == "or":
        if any(h is True for h in helds):
            return True
        if any(h is None for h in helds):
            return None
        return False
    raise ValueError(f"unknown logic {logic!r}")


def walk(tree: Tree, record: dict, overrides: dict[str, float] | None = None) -> list[PathStep]:
    """The ordered sequence of nodes visited for `record`, terminating in a leaf (§5.4.1(a)).
    `overrides` supplies the overlay stack's param values (§5.3.4) -- pass none for the
    published, unadjusted tree. Used both to capture a path live and, given only the stored
    path's tree version and the stored feature values, to re-render it later (§5.4.1(d)):
    `walk(tree, stored_record, stored_overrides)` reproduces the identical steps."""
    overrides = overrides or {}
    steps: list[PathStep] = []
    node_key = tree.root
    while True:
        node = tree.nodes[node_key]
        if node["kind"] == "leaf":
            steps.append(PathStep(node_key=node_key, kind="leaf", output=node["output"]))
            return steps
        term_evals = [_eval_term(t, record, overrides) for t in node["terms"]]
        held = _combine(node.get("logic"), [t.held for t in term_evals])
        direction = 0 if held is True else 1  # unknown ("otherwise") takes the false branch, per TreeConfig's default
        steps.append(PathStep(node_key=node_key, kind="test", terms=tuple(term_evals),
                               logic=node.get("logic"), held=held, direction=direction))
        node_key = node["true"] if direction == 0 else node["false"]


def render_path(steps: list[PathStep]) -> str:
    """A human-readable rendering (§5.4.1(d)), in the shape of the spec's own worked example."""
    lines = []
    for step in steps:
        if step.kind == "leaf":
            lines.append(f"leaf {step.node_key}  {step.output}")
            continue
        verdict = "HELD" if step.held else ("UNKNOWN->otherwise" if step.held is None else "NOT HELD")
        arrow = f"-> {'true' if step.direction == 0 else 'false'} branch"
        for i, t in enumerate(step.terms):
            tail = f"  {verdict}  {arrow}" if i == len(step.terms) - 1 else ""
            lines.append(f"node {step.node_key}  {t.rendered}{tail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node identity map, across two published versions (§5.4.3, §10 item 7)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeIdentityMap:
    carried_forward: tuple[str, ...]
    changed: tuple[dict, ...]     # {"old_node_key", "new_node_key", "old_condition", "new_condition"}
    added: tuple[str, ...]
    removed: tuple[str, ...]


def node_identity_map(old: Tree, new: Tree) -> NodeIdentityMap:
    """Classifies every node of `new` against `old` as carried forward, changed, added or
    removed (§5.4.3, acceptance §10 item 7). "Carried forward" falls straight out of
    `node_key` equality (that is the entire point of content-derived identity). "Changed" is
    a node whose *position* (parents' keys and directions) is unchanged but whose *own*
    condition text differs -- the position-signature match this function performs to
    distinguish "the volume dial moved node 6's threshold" (matched, §11 scenario 1) from
    "node 6 was deleted and an unrelated node was added elsewhere" (not matched)."""
    old_keys, new_keys = set(old.nodes), set(new.nodes)
    carried = old_keys & new_keys
    old_only, new_only = old_keys - new_keys, new_keys - old_keys

    by_position: dict[frozenset, list[str]] = defaultdict(list)
    for k in new_only:
        by_position[new.position_signature[k]].append(k)

    changed: list[dict] = []
    matched_new: set[str] = set()
    for k in sorted(old_only):
        candidates = [c for c in by_position.get(old.position_signature[k], []) if c not in matched_new]
        if candidates:
            new_k = candidates[0]
            matched_new.add(new_k)
            changed.append({
                "old_node_key": k, "new_node_key": new_k,
                "old_condition": old.condition_text[k], "new_condition": new.condition_text[new_k],
            })

    changed_old = {c["old_node_key"] for c in changed}
    changed_new = {c["new_node_key"] for c in changed}
    return NodeIdentityMap(
        carried_forward=tuple(sorted(carried)),
        changed=tuple(changed),
        added=tuple(sorted(new_only - changed_new)),
        removed=tuple(sorted(old_only - changed_old)),
    )
