"""decider2 Tree -> JDM.

The mapping target is `switchNode`, not `decisionTableNode`: a decider2 tree
is a graph of binary/N-way decision points, each with its own local test,
chained by edges — exactly JDM's `switchNode` + `hitPolicy: "first"` +
outgoing edges keyed by `sourceHandle`. This is a much closer structural
match than table -> table (jdm_to_decider2.py's direction), because neither
side forces the other's row/column shape onto it.

One `switchNode` per decider2 internal node:
  - `UnaryNode`         -> one statement (the condition, ZEL) + a default statement
  - `CompositeNode`     -> one statement joining `conditions` with `and`/`or`/`not`
  - `CasesRanges`       -> N statements (one per band, ZEL range syntax) + default
  - `CasesStringMatch`  -> N statements (`in [...]` over patterns) + default
  - `CasesIsIn`         -> N statements (`in [...]` over numeric values) + default
  - `LeafNode`          -> NOT a switchNode. See below.

What a `LeafNode` becomes, and why that is the one real seam: decider2's
leaves are an *index* into a shared `TreeOutput.data` table — two different
leaf nodes routinely share one `result_idx` (this tree's two "declined"
leaves both point at row 0), which is decider2's dedup for "many paths,
few distinct outcomes". JDM has no table-of-outcomes a switch branch can
point an edge at — a `switchNode`'s only job is to route, an `outputNode`
only exists once, at the very end. So each *distinct output row* used by a
leaf becomes one `expressionNode` (setting every output field from that
row's literal values) reached by every leaf edge that used to point at that
`result_idx`; the fan-in (several leaves -> one expressionNode) reproduces
the dedup, but as a shared node in the DAG rather than as an index into a
value table alongside the graph. Concretely: N distinct output rows really
in use -> N `expressionNode`s, however many leaves reference them.

What does NOT map (found while writing this, not assumed up front):
  - decider2's `param()` indirection. A `RangeCondition`/`Threshold` here
    can be a literal OR an `InputRef` (a named, separately-versioned,
    hot-swappable knob — doc 02 §4, doc 08 §4). JDM's cell/condition text
    has no such indirection: a threshold is baked into the ZEL string. This
    converter resolves every `InputRef` to its current value and emits a
    literal — round-tripping loses the *name* and the "change this without
    recompiling" property entirely; retuning the JDM copy means editing
    node content and republishing the document. Recorded per-conversion in
    `ConversionReport.lost_params`.
  - `CasesStringMatch`'s `match_type`/`case_sensitive`/`trim_whitespace`
    beyond exact+case-sensitive. ZEL's `matches`/`fuzzyMatch` exist but
    were not probed here (out of scope — decider2 itself only compiles
    `exact` to the kernel, doc `trees/__init__.py`'s migration notes), so
    this converter raises rather than silently emitting wrong-semantics ZEL.
  - Node ids: decider2 tree node ids are plain strings, kept as JDM node
    ids directly where they are already unique; JDM additionally requires
    every node to have a `name` (cosmetic, filled from the id).
  - `TreeOutput.default` (the `-1` sentinel row for "no leaf matched"): a
    decider2 tree cannot actually reach `-1` through normal traversal
    (every branch always has an explicit target, `_validate_structure`
    requires every source index used); the field exists for value-table
    completeness. There is nothing on the JDM side to receive it, so this
    converter ignores it — noted, not silently dropped, if it is set.
"""
from __future__ import annotations

import dataclasses
import typing as t

from decider2.trees.schema import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    CompositeNode,
    InputRef,
    LeafNode,
    RangeEndLogic,
    TLogicOp,
    Tree,
    TStringMatchType,
    UnaryNode,
)


@dataclasses.dataclass
class ConversionReport:
    nodes_converted: int = 0
    switch_nodes: int = 0
    expression_nodes: int = 0
    lost_params: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)


class TreeNotConvertible(Exception):
    pass


def _resolve(value, report: ConversionReport, *, where: str) -> float:
    if isinstance(value, InputRef):
        report.lost_params.append(f"{where}: InputRef({value.key!r}) resolved to a literal — name lost")
        raise TreeNotConvertible(
            f"{where}: threshold is an InputRef({value.key!r}) with no bound value supplied to "
            "the converter. Pass `param_values={...}` to resolve it, or accept the InputRef is "
            "not convertible (JDM has no equivalent indirection)."
        )
    return float(value)


def _feature_ref(feature) -> str:
    """A decider2 feature (a plain column name) as a ZEL identifier.

    NOT prefixed with `input.` — found empirically (a first isolated
    switchNode test came back `{}` for every input): the context a node
    inside a JDM graph evaluates its expressions against is the flat
    upstream object itself, not `{"input": <object>}`. The `credit-
    analysis.json` fixture confirms this independently — its
    `decisionTableNode` inputs use `"field": "company.turnover"`, never
    `"field": "input.company.turnover"`. `input` is the *name the editor
    gives the entry node*, not a wrapper key in the data.
    """
    root = feature.root if hasattr(feature, "root") else feature
    if not isinstance(root, str):
        raise TreeNotConvertible(f"computed feature {root!r} not supported by this converter")
    return root


def _zel_unary(condition, report: ConversionReport, param_values: dict) -> str:
    var = _feature_ref(condition.feature)
    op = condition.op
    if op in ("<", "<=", "==", ">", ">=", "!="):
        thr = _resolve(getattr(condition, "threshold"), report, where=f"{var} {op}")
        return f"{var} {op} {thr:g}"
    if op == "between":
        parts = []
        if condition.min is not None:
            parts.append(f"{var} >= {_resolve(condition.min, report, where=var):g}")
        if condition.max is not None:
            parts.append(f"{var} <= {_resolve(condition.max, report, where=var):g}")
        return " and ".join(parts)
    if op == "is_true":
        return f"{var} == true"
    if op == "is_false":
        return f"{var} == false"
    if op == "string_match":
        if condition.match_type != TStringMatchType.exact or not condition.case_sensitive:
            raise TreeNotConvertible(
                f"{var}: match_type={condition.match_type!r} case_sensitive={condition.case_sensitive} "
                "not probed against ZEN's matches()/fuzzyMatch() — refusing to guess semantics"
            )
        patterns = ", ".join(f'"{p}"' for p in condition.patterns)
        return f"{var} in [{patterns}]"
    raise TreeNotConvertible(f"unary op {op!r} has no ZEL translation written")


def _zel_range(cond, var: str, end_logic: RangeEndLogic, report: ConversionReport) -> str:
    lo_op = ">=" if end_logic is RangeEndLogic.lower_inclusive else ">"
    hi_op = "<" if end_logic is RangeEndLogic.lower_inclusive else "<="
    parts = []
    if cond.min is not None:
        parts.append(f"{var} {lo_op} {_resolve(cond.min, report, where=var):g}")
    if cond.max is not None:
        parts.append(f"{var} {hi_op} {_resolve(cond.max, report, where=var):g}")
    return " and ".join(parts)


def convert_tree(tree: Tree, *, param_values: dict | None = None) -> tuple[dict, ConversionReport]:
    """Returns (jdm_document, report). Raises TreeNotConvertible for a
    construct this converter does not yet handle (see module docstring)."""
    report = ConversionReport()
    node_map = tree.node_map()
    children = tree.children()

    jdm_nodes: list[dict] = []
    jdm_edges: list[dict] = []
    leaf_target_cache: dict[int, str] = {}  # result_idx -> expressionNode id (the dedup)
    edge_counter = [0]

    def new_edge(source: str, target: str, handle: str | None = None) -> None:
        edge_counter[0] += 1
        e = {"id": f"e{edge_counter[0]}", "sourceId": source, "targetId": target, "type": "edge"}
        if handle is not None:
            e["sourceHandle"] = handle
        jdm_edges.append(e)

    def leaf_target(leaf: LeafNode) -> str:
        idx = leaf.result_idx
        if idx in leaf_target_cache:
            return leaf_target_cache[idx]
        if idx == -1:
            raise TreeNotConvertible(
                "leaf result_idx=-1 (the default sentinel) reached during traversal — "
                "TreeOutput.default has no JDM target"
            )
        row = tree.output.data[idx]
        node_id = f"out_row_{idx}"
        expressions = [
            {"id": f"{node_id}_{col}", "key": col, "value": _literal(row[col])}
            for col in tree.output.columns
        ]
        jdm_nodes.append({
            "id": node_id, "type": "expressionNode", "name": f"row {idx}",
            "content": {"expressions": expressions},
        })
        report.expression_nodes += 1
        leaf_target_cache[idx] = node_id
        return node_id

    def _literal(value) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        return f"{value:g}" if isinstance(value, float) else str(value)

    def convert_node(node_id: str) -> str:
        """Returns the JDM node id this decider2 node became. Recurses into
        children, wiring edges as it goes."""
        pnode = node_map[node_id]
        data = pnode.data

        if isinstance(data, LeafNode):
            return leaf_target(data)

        kids = children.get(node_id, {})

        if isinstance(data, UnaryNode):
            cond_text = _zel_unary(data.condition, report, param_values or {})
            statements = [
                {"id": "s0", "condition": cond_text},
                {"id": "s1", "condition": "", "isDefault": True},
            ]
            jdm_id = node_id
            jdm_nodes.append({
                "id": jdm_id, "type": "switchNode", "name": node_id,
                "content": {"hitPolicy": "first", "statements": statements},
            })
            report.switch_nodes += 1
            for source_index, statement_id in ((0, "s0"), (1, "s1")):
                target = convert_node(kids[source_index])
                new_edge(jdm_id, target, handle=statement_id)
            return jdm_id

        if isinstance(data, CompositeNode):
            joiner = {TLogicOp.AND: " and ", TLogicOp.OR: " or "}.get(data.op)
            parts = [_zel_unary(c, report, param_values or {}) for c in data.conditions]
            if data.op is TLogicOp.NOT:
                cond_text = f"not ({parts[0]})"
            else:
                cond_text = joiner.join(f"({p})" for p in parts) if len(parts) > 1 else parts[0]
            statements = [{"id": "s0", "condition": cond_text}, {"id": "s1", "condition": "", "isDefault": True}]
            jdm_id = node_id
            jdm_nodes.append({
                "id": jdm_id, "type": "switchNode", "name": node_id,
                "content": {"hitPolicy": "first", "statements": statements},
            })
            report.switch_nodes += 1
            for source_index, statement_id in ((0, "s0"), (1, "s1")):
                target = convert_node(kids[source_index])
                new_edge(jdm_id, target, handle=statement_id)
            return jdm_id

        if isinstance(data, CasesRanges):
            var = _feature_ref(data.feature)
            statements = []
            for i, cond in enumerate(data.conditions):
                statements.append({"id": f"s{i}", "condition": _zel_range(cond, var, data.end_logic, report)})
            statements.append({"id": f"s{len(data.conditions)}", "condition": "", "isDefault": True})
            jdm_id = node_id
            jdm_nodes.append({
                "id": jdm_id, "type": "switchNode", "name": node_id,
                "content": {"hitPolicy": "first", "statements": statements},
            })
            report.switch_nodes += 1
            for i, st in enumerate(statements):
                target = convert_node(kids[i])
                new_edge(jdm_id, target, handle=st["id"])
            return jdm_id

        if isinstance(data, CasesStringMatch):
            if data.match_type != TStringMatchType.exact or not data.case_sensitive:
                raise TreeNotConvertible(f"{node_id}: non-exact/case-insensitive string match not probed")
            var = _feature_ref(data.feature)
            statements = []
            for i, cond in enumerate(data.conditions):
                patterns = ", ".join(f'"{p}"' for p in cond.patterns)
                statements.append({"id": f"s{i}", "condition": f"{var} in [{patterns}]"})
            statements.append({"id": f"s{len(data.conditions)}", "condition": "", "isDefault": True})
            jdm_id = node_id
            jdm_nodes.append({
                "id": jdm_id, "type": "switchNode", "name": node_id,
                "content": {"hitPolicy": "first", "statements": statements},
            })
            report.switch_nodes += 1
            for i, st in enumerate(statements):
                target = convert_node(kids[i])
                new_edge(jdm_id, target, handle=st["id"])
            return jdm_id

        if isinstance(data, CasesIsIn):
            var = _feature_ref(data.feature)
            statements = []
            for i, cond in enumerate(data.conditions):
                values = ", ".join(f"{_resolve(v, report, where=var):g}" for v in cond.values)
                statements.append({"id": f"s{i}", "condition": f"{var} in [{values}]"})
            statements.append({"id": f"s{len(data.conditions)}", "condition": "", "isDefault": True})
            jdm_id = node_id
            jdm_nodes.append({
                "id": jdm_id, "type": "switchNode", "name": node_id,
                "content": {"hitPolicy": "first", "statements": statements},
            })
            report.switch_nodes += 1
            for i, st in enumerate(statements):
                target = convert_node(kids[i])
                new_edge(jdm_id, target, handle=st["id"])
            return jdm_id

        raise TreeNotConvertible(f"node kind {type(data).__name__} has no converter written")

    input_id = "input"
    output_id = "output"
    jdm_nodes.append({"id": input_id, "type": "inputNode", "name": "Request"})
    root_jdm_id = convert_node(tree.root_id())
    new_edge(input_id, root_jdm_id)

    # Every distinct expressionNode (one per distinct output row actually
    # used) feeds the single outputNode.
    jdm_nodes.append({"id": output_id, "type": "outputNode", "name": "Response"})
    for exp_node_id in leaf_target_cache.values():
        new_edge(exp_node_id, output_id)

    report.nodes_converted = len(tree.nodes)
    doc = {"nodes": jdm_nodes, "edges": jdm_edges}
    return doc, report
