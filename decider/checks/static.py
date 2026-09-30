from __future__ import annotations

import ast
from typing import Any, Iterator

from decider.check import Finding, Severity
from decider.checks._common import call_nodes, function_source, step_ref
from decider.engine.ir.context import to_ir
from decider.engine.ir.nodes import CallNode

_WALL_CLOCK: dict[str, set[str]] = {
    "time": {"time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
             "process_time", "process_time_ns", "thread_time", "thread_time_ns", "clock"},
    "datetime": {"now", "utcnow", "today"},
    "random": {"random", "randint", "randrange", "uniform", "choice", "choices", "shuffle",
               "sample", "gauss", "randbytes"},
    "uuid": {"uuid1", "uuid4"},
    "secrets": {"token_bytes", "token_hex", "token_urlsafe", "randbelow", "choice"},
    "os": {"urandom"},
}

_MUTABLE_BUILTINS = {"list", "dict", "set"}


def wall_clock(step: Any) -> list[Finding]:
    """Detect wall-clock and randomness reads that make a flow non-deterministic.

    Scans each step's source for calls like `time.time()`, `datetime.now()` or
    `random.random()`. A finding is a warning; whether it blocks a release is
    the client's policy.
    """
    return list(_flagged(step, "wall_clock", _wall_clock_findings))


def impure(step: Any) -> list[Finding]:
    """Detect shared-state mutation where the source shows it.

    Scans each step's source for `global`/`nonlocal` statements (writes to
    state other calls share) and mutable default arguments (state carried
    across calls). Heuristic: reading a module constant is not flagged, only
    writes and mutable defaults are.
    """
    return list(_flagged(step, "impure", _impure_findings))


def numeric(step: Any) -> list[Finding]:
    """Flag numeric risk patterns: overflow/underflow and floating-point sensitivity.

    Heuristic, not a proof of a defect, and deliberately narrow to stay quiet:
    exponentiation (overflow/underflow), integer scaling by a constant (int64
    overflow — int64 wraps silently), true division by an integer constant
    (cents lose precision when cast to float), and `==`/`!=` against a float
    constant (fragile on floats). Confirm each against threshold/regime cases
    with `decider.checks.equivalence`.
    """
    return list(_flagged(step, "numeric", _numeric_findings))


def _flagged(step: Any, name: str, visit) -> Iterator[Finding]:
    ir = to_ir(step)
    flow_id = ir.origin.id
    for node in call_nodes(ir):
        src = function_source(node.fn)
        if src is None:
            continue
        text, start = src
        yield from visit(node, ast.parse(text), start, flow_id, name)


def _finding(node: CallNode, start: int, flow_id: str | None, line: int, name: str,
             message: str, detail: dict) -> Finding:
    return Finding(check=name, severity=Severity.WARNING, message=message,
                   step=step_ref(node, flow_id), line=start + line - 1, detail=detail)


def _wall_clock_findings(node: CallNode, tree: ast.AST, start: int, flow_id: str | None,
                         name: str) -> Iterator[Finding]:
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        module = getattr(call.func.value, "id", None)
        if module in _WALL_CLOCK and call.func.attr in _WALL_CLOCK[module]:
            text = f"{module}.{call.func.attr}()"
            yield _finding(node, start, flow_id, call.lineno, name, f"{node.origin.path} reads {text}; "
                           "a wall-clock read makes the flow non-deterministic", {"call": text})


def _impure_findings(node: CallNode, tree: ast.AST, start: int, flow_id: str | None,
                     name: str) -> Iterator[Finding]:
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Global):
            for n in stmt.names:
                yield _finding(node, start, flow_id, stmt.lineno, name,
                               f"{node.origin.path} writes the global {n!r}",
                               {"kind": "global", "name": n})
        elif isinstance(stmt, ast.Nonlocal):
            for n in stmt.names:
                yield _finding(node, start, flow_id, stmt.lineno, name,
                               f"{node.origin.path} writes the closure cell {n!r}",
                               {"kind": "nonlocal", "name": n})
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = fn.args.args[-len(fn.args.defaults):]
        for arg, default in zip(args, fn.args.defaults):
            if _mutable_default(default):
                yield _finding(node, start, flow_id, default.lineno, name,
                               f"{node.origin.path} has a mutable default for {arg.arg!r}; "
                               "its state persists across calls",
                               {"kind": "mutable_default", "arg": arg.arg})


def _mutable_default(default: ast.expr) -> bool:
    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
        return True
    return (isinstance(default, ast.Call) and isinstance(default.func, ast.Name)
            and default.func.id in _MUTABLE_BUILTINS)


def _numeric_findings(node: CallNode, tree: ast.AST, start: int, flow_id: str | None,
                      name: str) -> Iterator[Finding]:
    for expr in ast.walk(tree):
        if isinstance(expr, ast.BinOp):
            if isinstance(expr.op, ast.Pow):
                yield _finding(node, start, flow_id, expr.lineno, name,
                               f"{node.origin.path}: {ast.unparse(expr)} may overflow or underflow",
                               {"expression": ast.unparse(expr), "risk": "overflow_underflow"})
            elif isinstance(expr.op, ast.Mult) and (_is_int_const(expr.left) or _is_int_const(expr.right)):
                yield _finding(node, start, flow_id, expr.lineno, name,
                               f"{node.origin.path}: {ast.unparse(expr)} scales an integer; "
                               "int64 overflows silently",
                               {"expression": ast.unparse(expr), "risk": "int64_overflow"})
            elif isinstance(expr.op, ast.Div) and (_is_int_const(expr.left) or _is_int_const(expr.right)):
                yield _finding(node, start, flow_id, expr.lineno, name,
                               f"{node.origin.path}: {ast.unparse(expr)} is true division; "
                               "integer values lose precision as float",
                               {"expression": ast.unparse(expr), "risk": "float_precision"})
        elif (isinstance(expr, ast.Compare) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in expr.ops)
              and _has_float_const(expr)):
            yield _finding(node, start, flow_id, expr.lineno, name,
                           f"{node.origin.path}: {ast.unparse(expr)} compares for equality with a float; "
                           "fragile on floats",
                           {"expression": ast.unparse(expr), "risk": "float_equality"})


def _is_int_const(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int)


def _has_float_const(expr: ast.Compare) -> bool:
    operands = [expr.left, *expr.comparators]
    return any(isinstance(o, ast.Constant) and isinstance(o.value, float) for o in operands)
