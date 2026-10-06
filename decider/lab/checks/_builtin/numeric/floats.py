from __future__ import annotations

import ast
from typing import Iterator, Literal

from decider.lab.checks._base import Check, CheckContext
from decider.lab.checks._builtin.numeric._scope import DEC, FLOAT, INT, kind_of, scopes
from decider.lab.checks._report import Finding, Severity
from decider.ir import CallNode, iter_nodes

__all__ = ["AccumulatorDtype", "FloatEquality", "NanPropagation", "PowExpOverflow", "UnsafeCast"]

_BOUNDING = {"min", "clip", "minimum"}
_NARROW = {"int8", "int16", "int32", "uint8", "uint16", "uint32", "float16", "float32"}
_POSITIVE_ONLY = {"sqrt": "non-negative", "log": "positive", "log2": "positive", "log10": "positive", "log1p": "above -1"}


def _name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _const(e: ast.AST) -> bool:
    return isinstance(e, ast.Constant) or (isinstance(e, ast.UnaryOp) and isinstance(e.operand, ast.Constant))


def _bounded(e: ast.AST) -> bool:
    return _const(e) or (isinstance(e, ast.Call) and _name(e) in _BOUNDING)


class PowExpOverflow(Check):
    """`**`, `exp` and `pow` with a non-constant exponent overflow to inf or underflow to 0."""

    type: Literal["pow_exp_overflow"] = "pow_exp_overflow"
    code = "DCD106"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            for e in scope.walk():
                if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Pow):
                    exponent = e.right
                    if scope.type_of(e.left) == INT and scope.type_of(exponent) == INT:
                        continue  # DCD101 owns int powers
                elif isinstance(e, ast.Call) and _name(e) in ("exp", "expm1") and e.args:
                    exponent = e.args[0]
                elif isinstance(e, ast.Call) and _name(e) in ("pow", "power") and len(e.args) == 2:
                    exponent = e.args[1]
                else:
                    continue
                if _bounded(exponent):
                    continue
                text = ast.unparse(e)
                yield ctx.finding(node, f"{node.origin.path}: {text} has an unbounded exponent; it can overflow to inf or underflow to 0",
                                  "clamp the exponent, e.g. min(x, 700)", line=src.line(e), expression=text)


class FloatEquality(Check):
    """`==` / `!=` against a float constant, or between two float values, is fragile."""

    type: Literal["float_equality"] = "float_equality"
    code = "DCD107"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            for e in scope.walk():
                if not isinstance(e, ast.Compare):
                    continue
                sides = [e.left, *e.comparators]
                for op, a, b in zip(e.ops, sides, sides[1:]):
                    if isinstance(op, (ast.Eq, ast.NotEq)) and (
                            any(isinstance(s, ast.Constant) and isinstance(s.value, float) for s in (a, b))
                            or (isinstance(a, ast.Name) and isinstance(b, ast.Name)
                                and scope.type_of(a) == scope.type_of(b) == FLOAT)):
                        text = ast.unparse(e)
                        yield ctx.finding(node, f"{node.origin.path}: {text} compares floats for equality; fragile on floats",
                                          "compare with a tolerance", line=src.line(e), expression=text)
                        break


class NanPropagation(Check):
    """A float input compared without a NaN guard (NaN compares False), and `sqrt` / `log` of a value that may be out of domain."""

    type: Literal["nan_propagation"] = "nan_propagation"
    code = "DCD108"
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            path = node.origin.path
            guarded = {n.id for c in scope.walk() if isinstance(c, ast.Call) and _name(c) in ("isnan", "isfinite", "isinf")
                       for n in ast.walk(c) if isinstance(n, ast.Name)}
            reported: set[str] = set()
            for e in scope.walk():
                if isinstance(e, ast.Compare) and any(isinstance(o, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for o in e.ops):
                    for n in ast.walk(e):
                        if (isinstance(n, ast.Name) and n.id in scope.inputs and scope.types.get(n.id) == FLOAT
                                and n.id not in guarded | reported):
                            reported.add(n.id)
                            yield ctx.finding(node, f"{path}: {n.id} is compared without a NaN guard; a NaN takes the false side silently",
                                              "check math.isnan, or fill the null upstream", line=src.line(e), name=n.id)
                elif isinstance(e, ast.Call) and _name(e) in _POSITIVE_ONLY and e.args:
                    arg = e.args[0]
                    if scope.nonzero(arg) or _nonneg(arg) and _name(e) == "sqrt" or scope.guarded(e, arg):
                        continue
                    text = ast.unparse(e)
                    yield ctx.finding(node, f"{path}: {text} needs a {_POSITIVE_ONLY[_name(e)]} argument and nothing guarantees it",
                                      "guard the argument, or clamp it with max(x, eps)", line=src.line(e), expression=text)


def _nonneg(e: ast.AST) -> bool:
    if isinstance(e, ast.Constant):
        return isinstance(e.value, (int, float)) and e.value >= 0
    if isinstance(e, ast.BinOp):
        if isinstance(e.op, ast.Pow):
            return isinstance(e.right, ast.Constant) and e.right.value in (2, 4, 6)
        if isinstance(e.op, ast.Mult) and ast.unparse(e.left) == ast.unparse(e.right):
            return True
        return isinstance(e.op, (ast.Add, ast.Mult)) and _nonneg(e.left) and _nonneg(e.right)
    return isinstance(e, ast.Call) and _name(e) == "abs"


class AccumulatorDtype(Check):
    """A loop carry that grows by addition or multiplication in int64 (wraps), or any carry or accumulator in float32."""

    type: Literal["accumulator_dtype"] = "accumulator_dtype"
    code = "DCD109"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        carried: dict[int, set[str]] = {}
        for n in ctx.nodes:
            for name in getattr(n, "carries", ()):
                for call in _calls(n.body):
                    if any(o.name == name for o in call.outputs or ()):
                        carried.setdefault(id(call), set()).add(name)
            acc = getattr(n, "accumulate", None)
            if acc is not None:
                carried.setdefault(id(acc), set()).update(o.name for o in acc.outputs or ())
        for node, src, scope in scopes(ctx):
            names = carried.get(id(node), set())
            for o in node.outputs or ():
                if o.name not in names:
                    continue
                dtype = getattr(o.annotation, "__name__", "")
                if dtype == "float32":
                    yield ctx.finding(node, f"{node.origin.path}: accumulator {o.name} is float32; sums lose precision after ~2**24",
                                      "accumulate in float64", name=o.name, dtype="float32")
                elif kind_of(o.annotation) == INT and _grows(scope):
                    yield ctx.finding(node, f"{node.origin.path}: accumulator {o.name} is int64; a growing total wraps silently",
                                      "accumulate in float64 (exact to 2**53)", name=o.name, dtype="int64")


def _calls(node) -> Iterator[CallNode]:
    return (n for n in iter_nodes(node) if isinstance(n, CallNode))


def _grows(scope) -> bool:
    # ponytail: a return like `total + step` counts as growth; a counter `i + 1` doesn't. A flow-sensitive range check would be exact.
    for stmt in scope.walk():
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            for e in ast.walk(stmt.value):
                if isinstance(e, ast.BinOp) and (isinstance(e.op, ast.Mult)
                                                 or isinstance(e.op, ast.Add) and not any(_const(s) for s in (e.left, e.right))):
                    return True
    return False


class UnsafeCast(Check):
    """`float(decimal)` past 2**53 of its int64 and narrowing casts such as `np.int32` can lose value."""

    type: Literal["unsafe_cast"] = "unsafe_cast"
    code = "DCD110"

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node, src, scope in scopes(ctx):
            for e in scope.walk():
                if not isinstance(e, ast.Call) or not e.args:
                    continue
                name, arg = _name(e), e.args[0]
                if name in _NARROW:
                    why, fix = f"{name} is a narrowing cast", "keep int64 / float64"
                elif name == "float" and scope.type_of(arg) == DEC:
                    why, fix = ("float(decimal) is inexact once its int64 passes 2**53 (about 900 billion at four places)",
                                "keep the amount a Decimal, and compare or divide it as one")
                else:
                    continue
                text = ast.unparse(e)
                yield ctx.finding(node, f"{node.origin.path}: {text}: {why}", fix, line=src.line(e), expression=text)
