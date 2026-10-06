from __future__ import annotations

import ast
from typing import Iterator, Literal

from decider.lab.checks._base import Check, CheckContext
from decider.lab.checks._builtin.numeric._scope import DEC, FLOAT, INT, kind_of, scopes
from decider.lab.checks._report import Finding, Severity

__all__ = ["BareRound", "DivisionByVariable", "FloatMoney", "Int64Overflow", "IntTrueDivision"]

_DIVIDE = (ast.Div, ast.FloorDiv, ast.Mod)


def _both_const(e: ast.BinOp) -> bool:
    return isinstance(e.left, ast.Constant) and isinstance(e.right, ast.Constant)


class Int64Overflow(Check):
    """Integer `*`, `**`, `<<` and a running `+=` in a loop wrap silently past int64; an error when a Decimal is involved."""

    type: Literal["int64_overflow"] = "int64_overflow"
    code = "DCD101"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            for e in scope.walk():
                if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Mult, ast.Pow, ast.LShift)) and not _both_const(e):
                    op, operands = e.op, e
                elif isinstance(e, ast.AugAssign) and isinstance(e.op, ast.Add) and _in_loop(scope, e):
                    op, operands = e.op, ast.BinOp(e.target, e.op, e.value)
                else:
                    continue
                kinds = (scope.type_of(operands.left), scope.type_of(operands.right))
                decimal = DEC in kinds and FLOAT not in kinds
                if not decimal and kinds != (INT, INT):
                    continue
                text = ast.unparse(e)
                verb = "a running total" if isinstance(e, ast.AugAssign) else "a product"
                yield ctx.finding(node, f"{node.origin.path}: {text} is {verb} in int64, "
                                  + ("which raises OverflowError in Python and wraps silently in a compiled step, on a Decimal"
                                     if decimal else "which wraps silently"),
                                  "bound the operands" + ("" if decimal else ", or compute in float64 (exact to 2**53)"),
                                  line=src.line(e),
                                  severity=Severity.ERROR if decimal and self.severity is Severity.WARNING else None,
                                  expression=text, money=decimal)


def _in_loop(scope, e: ast.AST) -> bool:
    while e in scope.parents:
        e = scope.parents[e]
        if isinstance(e, (ast.For, ast.While)):
            return True
    return False


class FloatMoney(Check):
    """A float returned into a Decimal output, and `int(x * 100)` truncation."""

    type: Literal["float_money"] = "float_money"
    code = "DCD102"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            path = node.origin.path
            decimal_out = any(kind_of(o.annotation) == DEC for o in node.outputs or ())
            for e in scope.walk():
                if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "int" and e.args:
                    inner = e.args[0]
                    if (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mult)
                            and any(isinstance(s, ast.Constant) and s.value in (100, 100.0) for s in (inner.left, inner.right))):
                        text = ast.unparse(e)
                        yield ctx.finding(node, f"{path}: {text} truncates toward zero, so a cent is lost on .xx9",
                                          "round with decider.ops.decimal.round before the cast", line=src.line(e),
                                          expression=text, kind="truncation")
                elif decimal_out and isinstance(e, ast.Return) and e.value is not None and scope.type_of(e.value) == FLOAT:
                    text = ast.unparse(e.value)
                    yield ctx.finding(node, f"{path}: returns float {text} into a Decimal output",
                                      "return a Decimal: multiply a Decimal by the float, or build one with Currency(major, minor)",
                                      line=src.line(e), expression=text, kind="float_to_decimal")


class BareRound(Check):
    """`round(x, n)` and `np.round` differ between CPython and numba on .xx5 ties; an error on a Decimal output."""

    type: Literal["bare_round"] = "bare_round"
    code = "DCD103"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            money = scope.money_output()
            builtin_round = getattr(node.fn, "__globals__", {}).get("round", round) is round
            for e in scope.walk():
                if not isinstance(e, ast.Call):
                    continue
                f = e.func
                name = f.id if isinstance(f, ast.Name) and f.id == "round" and builtin_round else (
                    f"{f.value.id}.{f.attr}" if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id in ("np", "numpy") and f.attr in ("round", "around") else None)
                if name:
                    text = ast.unparse(e)
                    yield ctx.finding(node, f"{node.origin.path}: {text} rounds differently in Python and compiled modes on .xx5",
                                      "use decider.ops.decimal.round(x, n, Rounding.HALF_UP)", line=src.line(e),
                                      severity=Severity.ERROR if money and self.severity is Severity.WARNING else None,
                                      expression=text, money_output=money)


class DivisionByVariable(Check):
    """`/`, `//` or `%` by a value nothing proves non-zero: no guard, no `max(b, eps)`, no `gt=0` param."""

    type: Literal["division_by_variable"] = "division_by_variable"
    code = "DCD104"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            for e in scope.walk():
                if isinstance(e, ast.BinOp) and isinstance(e.op, _DIVIDE):
                    divisor, at = e.right, e
                elif isinstance(e, ast.AugAssign) and isinstance(e.op, _DIVIDE):
                    divisor, at = e.value, e
                else:
                    continue
                if scope.nonzero(divisor) or scope.guarded(at, divisor):
                    continue
                text = ast.unparse(e)
                yield ctx.finding(node, f"{node.origin.path}: {text} divides by {ast.unparse(divisor)}, which may be zero",
                                  "guard it (`if b != 0`), use max(b, eps), or bound the param with gt=0",
                                  line=src.line(e), expression=text)


class IntTrueDivision(Check):
    """True division of two ints turns them into floats; above 2**53 the quotient loses precision."""

    type: Literal["int_true_division"] = "int_true_division"
    code = "DCD105"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            for e in scope.walk():
                if (isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div) and not _both_const(e)
                        and scope.type_of(e.left) == INT and scope.type_of(e.right) == INT):
                    text = ast.unparse(e)
                    yield ctx.finding(node, f"{node.origin.path}: {text} is true division of two ints; the result is a float",
                                      "use // for an int result, or document the precision", line=src.line(e), expression=text)
