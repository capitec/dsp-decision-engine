from __future__ import annotations

import ast
from typing import Iterator, Literal

from decider.checks._base import Check, CheckContext
from decider.checks._report import Finding

__all__ = ["Numeric"]


class Numeric(Check):
    """Numeric risk patterns: overflow, underflow and floating-point sensitivity.

    Deliberately narrow: exponentiation, integer scaling by a constant (int64 wraps
    silently), true division with an integer constant, and `==` / `!=` against a float constant.
    """

    type: Literal["numeric"] = "numeric"
    # Temporary catch-all code; the per-risk numeric checks replace it and retire DCD100.
    code = "DCD100"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            src = ctx.source(node)
            for expr in ast.walk(src.tree) if src else ():
                found = _risk(expr)
                if found:
                    message, fix, risk = found
                    text = ast.unparse(expr)
                    yield ctx.finding(node, f"{node.origin.path}: {text} {message}", fix, line=src.line(expr),
                                      expression=text, risk=risk)


def _risk(expr: ast.AST) -> tuple[str, str, str] | None:
    if isinstance(expr, ast.BinOp):
        scaled = _is_int_const(expr.left) or _is_int_const(expr.right)
        if isinstance(expr.op, ast.Pow):
            return "may overflow or underflow", "bound the exponent or compute in a wider type", "overflow_underflow"
        if isinstance(expr.op, ast.Mult) and scaled:
            return ("scales an integer; int64 overflows silently", "check the operand range, or compute in float",
                    "int64_overflow")
        if isinstance(expr.op, ast.Div) and scaled:
            return ("is true division; integer values lose precision as float",
                    "divide in integer arithmetic or document the precision", "float_precision")
    elif isinstance(expr, ast.Compare) and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in expr.ops):
        if any(isinstance(o, ast.Constant) and isinstance(o.value, float) for o in [expr.left, *expr.comparators]):
            return "compares for equality with a float; fragile on floats", "compare with a tolerance", "float_equality"
    return None


def _is_int_const(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int)
