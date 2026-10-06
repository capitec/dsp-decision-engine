from __future__ import annotations

import ast
from importlib import import_module
from typing import Any, Iterator, Literal

from decider._engine.wiring.plan import Branch, Loop, ScatterGather, Sequence
from decider.lab.checks._base import Check, CheckContext
from decider.lab.checks._report import Finding, Severity, SkippedCheck
from decider.ir import BranchNode, CallNode, LoopNode
from decider.steps.tables import DecisionTableConfig
from decider.steps.trees import TreeConfig

__all__ = ["DeadOutput", "OpaqueFrameStep", "ShadowedWrite", "TableGap", "TableOverlap", "UnboundedLoop",
           "UnreachableBranch", "UnusedInput"]

HUGE_ITERATIONS = 100_000


class UnreachableBranch(Check):
    """A branch arm that can never run: a constant condition, a condition returning an index with no arm, a tree branch or a table row no input reaches.

    Trees and decision tables are solved with z3 (`pip install decider[prove]`); without it that part is
    listed in `report.skipped`.
    """

    type: Literal["unreachable_branch"] = "unreachable_branch"
    code = "DCD201"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        yield from _solved(ctx)
        for node in ctx.nodes:
            if not isinstance(node, BranchNode):
                continue
            src = ctx.source(node.condition)
            fn = _function(src)
            if fn is None:
                continue
            values, dynamic = _returned_constants(fn)
            is_bool = node.condition.outputs[0].annotation is bool
            picked = {(0 if v else 1) if is_bool else v for v in values if isinstance(v, (bool, int))}
            for index in sorted(i for i in picked if not 0 <= i < len(node.arms)):
                yield ctx.finding(node, f"{node.origin.path}: the condition returns {index}, but the branch has "
                                  f"{len(node.arms)} arms", "return an index below the number of arms, or add the arm",
                                  line=src.line(fn), index=index)
            if dynamic:
                continue
            for index, arm in enumerate(node.arms):
                if index not in picked:
                    yield ctx.finding(node, f"{node.origin.path}: arm {index} ({arm.origin.path}) never runs, "
                                      "the condition always returns a constant", "make the condition depend on the "
                                      "data, or drop the branch", line=src.line(fn), index=index)


class DeadOutput(Check):
    """A value written inside a branch, loop or each that nothing reads and that never leaves it."""

    type: Literal["dead_output"] = "dead_output"
    code = "DCD202"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for call, version in _unused_writes(ctx):
            if not _later(ctx, version):
                yield ctx.finding(call.node, f"{version.producer} writes {version.name!r}, which nothing reads "
                                  "and which does not leave the step that holds it",
                                  f"read it, add it to `modifies` or `carries`, emit it with "
                                  f".emit({version.name + '@' + version.producer!r}), or drop the write",
                                  name=version.name)


class ShadowedWrite(Check):
    """A value overwritten by a later step before anything reads it: the later write wins and the first is lost."""

    type: Literal["shadowed_write"] = "shadowed_write"
    code = "DCD203"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for call, version in _unused_writes(ctx):
            later = _later(ctx, version)
            if later:
                yield ctx.finding(call.node, f"{version.producer} writes {version.name!r}, but {later.producer} "
                                  "overwrites it before anything reads it",
                                  "read the first value in between, rename one of the two, or drop the first write",
                                  name=version.name, overwritten_by=later.producer)


class UnusedInput(Check):
    """A step argument the function body never uses."""

    type: Literal["unused_input"] = "unused_input"
    code = "DCD204"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            fn = _function(src := ctx.source(node))
            if fn is None or node.inputs is None:
                continue
            body = [n for stmt in fn.body for n in ast.walk(stmt) if isinstance(n, ast.Name)]
            names = {n.id for n in body}
            if names & {"locals", "vars"}:
                continue
            for inp in node.inputs:
                if inp.arg not in names:
                    yield ctx.finding(node, f"{node.origin.path}: argument {inp.arg!r} is never used",
                                      "remove the argument; if it only orders this step after its producer, keep it "
                                      "and suppress this check", line=src.line(fn), argument=inp.arg)


class UnboundedLoop(Check):
    """A loop with a huge `max_iterations`, or a condition that reads none of the carried names."""

    type: Literal["unbounded_loop"] = "unbounded_loop"
    code = "DCD205"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.nodes:
            if not isinstance(node, LoopNode):
                continue
            path = node.origin.path
            if node.max_iterations > HUGE_ITERATIONS:
                yield ctx.finding(node, f"{path}: max_iterations is {node.max_iterations}, so one row may run this "
                                  "long", f"set it to the most iterations a row can legitimately need",
                                  max_iterations=node.max_iterations)
            reads = {i.name for i in node.condition.inputs or ()}
            if node.condition.inputs is not None and not reads & set(node.carries):
                yield ctx.finding(node, f"{path}: the condition reads none of the carried names "
                                  f"{list(node.carries)}, so the iterations never change it",
                                  "make the condition depend on a carried value", carries=list(node.carries))


class OpaqueFrameStep(Check):
    """A frame step with unknown lineage: it blocks lineage checks and fusion, and a single record can't run it."""

    type: Literal["opaque_frame_step"] = "opaque_frame_step"
    code = "DCD206"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.nodes:
            if isinstance(node, CallNode) and node.kind == "frame" and (node.inputs is None or node.outputs is None):
                yield ctx.finding(node, f"{node.origin.path} is a frame step that doesn't declare what it reads and "
                                  "writes, so lineage checks and fusion stop here and a single record can't run it",
                                  "declare its inputs and outputs, or write it as a row step")


class TableOverlap(Check):
    """Two decision table rows that can both match one input; the earlier row wins. Reported by `unreachable_branch`, which solves the table."""

    type: Literal["table_overlap"] = "table_overlap"
    code = "DCD207"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        return iter(())


class TableGap(Check):
    """An input no decision table row matches, so it takes `default` (or null). Reported by `unreachable_branch`, which solves the table."""

    type: Literal["table_gap"] = "table_gap"
    code = "DCD208"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        return iter(())


def _function(src: Any) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if src is None:
        return None
    fn = src.tree.body[0] if src.tree.body else None
    return fn if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def _returned_constants(fn: ast.AST) -> tuple[set[Any], bool]:
    values: set[Any] = set()
    dynamic = False
    todo = list(fn.body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (bool, int)):
                values.add(node.value.value)
            else:
                dynamic = True
        todo.extend(ast.iter_child_nodes(node))
    return values, dynamic


def _used(ctx: CheckContext) -> set[int]:
    used = {v.id for v in ctx.plan.outputs.values()}
    for call in ctx.plan.calls:
        used.update(v.id for v in call.reads or ())
    todo = [ctx.plan.root]
    while todo:
        r = todo.pop()
        if isinstance(r, Sequence):
            todo.extend(r.children)
        elif isinstance(r, Branch):
            todo.extend(r.arms)
            for m in r.merges:
                used.update(v.id for v in (m.prior, *m.arms) if v is not None)
        elif isinstance(r, Loop):
            todo.append(r.body)
            used.update(v.id for c in r.carries for v in (c.initial, c.last))
        elif isinstance(r, ScatterGather):
            todo.append(r.body)
            used.update(v.id for v in (r.column, *r.new_fields))
    return used


def _unused_writes(ctx: CheckContext) -> Iterator[tuple[Any, Any]]:
    """Each `(call, version)` that something wrote and nothing reads, outside condition calls and unknown-lineage frame steps."""
    plan = ctx.plan
    routing = {id(n.condition) for n in ctx.nodes if isinstance(n, (BranchNode, LoopNode))}
    used = _used(ctx)
    for call in plan.calls:
        if id(call.node) in routing or call.node.outputs is None:
            continue
        for v in call.writes:
            if v.id not in used:
                yield call, v


def _later(ctx: CheckContext, version: Any) -> Any:
    chain = ctx.plan.chains[version.name]
    parent = version.producer.rpartition("/")[0]
    return next((v for v in chain[chain.index(version) + 1:] if v.producer.rpartition("/")[0] == parent), None)


def _skip(ctx: CheckContext, code: str, check: str, reason: str) -> None:
    ctx.skipped.append(SkippedCheck(check=check, code=code, reason=f"{ctx.node.origin.path}: {reason}"))


def _solver(ctx: CheckContext) -> Any:
    try:
        _smt = import_module("decider.lab.diff._smt")
    except ImportError:
        _skip(ctx, "DCD201", "unreachable_branch", "install decider[prove] to check this step's branches")
        return None
    return _smt


def _solved(ctx: CheckContext) -> Iterator[Finding]:
    by_path = {}
    for n in ctx.nodes:
        by_path.setdefault(n.origin.path, n)
    for path, step in ctx.steps.items():
        if path in by_path and isinstance(step, (TreeConfig, DecisionTableConfig)):
            ctx.node = by_path[path]
            yield from (tree_findings if isinstance(step, TreeConfig) else table_findings)(step, ctx)
    ctx.node = None


def tree_findings(step: Any, ctx: CheckContext) -> Iterator[Finding]:
    """DCD201 for a tree: branches no input can take."""
    smt = _solver(ctx)
    if smt is None:
        return
    try:
        arms = smt.unreachable_arms(step)
    except smt.Unencodable as e:
        _skip(ctx, "DCD201", "unreachable_branch", f"not solved, {e}")
        return
    for nid, k, child in arms:
        yield ctx.finding(ctx.node, f"{ctx.node.origin.path}: branch {k} of node {nid!r} (to {child!r}) can never "
                          "be taken", "change the conditions above it, or remove the branch",
                          code="DCD201", check="unreachable_branch", severity=Severity.WARNING, tree_node=nid, branch=k)


def table_findings(step: Any, ctx: CheckContext) -> Iterator[Finding]:
    """DCD201, DCD207 and DCD208 for a decision table."""
    smt = _solver(ctx)
    if smt is None:
        return
    try:
        result = smt.table_structure(step)
    except smt.Unencodable as e:
        _skip(ctx, "DCD201", "unreachable_branch", f"not solved, {e}")
        return
    path = ctx.node.origin.path
    for r in result["dead"]:
        yield ctx.finding(ctx.node, f"{path}: row {r} ({path}#{r}) can never win; the rows above it match "
                          "everything it would", "reorder or remove the row", code="DCD201",
                          check="unreachable_branch", severity=Severity.WARNING, row=r)
    if result["overlaps"] is None:
        _skip(ctx, "DCD207", "table_overlap", "too many rows to compare in pairs")
    for i, j, row in result["overlaps"] or ():
        yield ctx.finding(ctx.node, f"{path}: rows {i} and {j} both match {row}; row {i} wins",
                          "narrow one of the rows if the overlap is not intended", code="DCD207",
                          check="table_overlap", severity=Severity.INFO, rows=[i, j], example=row)
    if result["gap"] is not None:
        where = "its default" if step.default is not None else "null outputs, the table has no default"
        yield ctx.finding(ctx.node, f"{path}: no row matches {result['gap']}; it takes {where}",
                          "add a row for it, or set `default` on purpose", code="DCD208", check="table_gap",
                          severity=Severity.INFO if step.default is not None else Severity.WARNING,
                          example=result["gap"])
