from __future__ import annotations

import ast
import math
import typing
from typing import Any, Iterator, Literal

import polars as pl

from decider.checks._base import Check, CheckContext
from decider.checks._report import Finding, Severity
from decider.ir import CallNode, NullPolicy

__all__ = ["ConstantOutput", "Int64Range", "MissingRequired", "NanOutput"]

_INT64 = 2**63
_FLOAT_EXACT = 2**53


class MissingRequired(Check):
    """Required inputs that are absent from the frame or null in it, reported at the first step that reads them."""

    type: Literal["missing_required"] = "missing_required"
    code = "DCD504"
    needs_data = True
    severity: Severity = Severity.ERROR

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        readers: dict[str, object] = {}
        for call in ctx.plan.calls:
            for v in call.reads or ():
                if v.producer is None:
                    readers.setdefault(v.name, call.node)
        for inp in ctx.plan.inputs:
            if inp.null_policy is not NullPolicy.REQUIRED:
                continue
            node = readers.get(inp.name, ctx.ir)
            if inp.name not in ctx.frame.columns:
                yield ctx.finding(node, f"required input column {inp.name!r} is not in the data and has no default",
                                  fix="add the column, or give the parameter a default", column=inp.name)
            elif nulls := ctx.frame[inp.name].null_count():
                yield ctx.finding(node, f"required input column {inp.name!r} has {nulls} null row(s) and no default to fill them",
                                  fix="fill the nulls, or declare a default with `missing_as`", column=inp.name, nulls=nulls)


class NanOutput(Check):
    """NaN or infinity in an output column when the flow runs over the data."""

    type: Literal["nan_output"] = "nan_output"
    code = "DCD502"
    needs_data = True
    executes = True

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        if (run := ctx.outputs) is None:
            return
        out, producers = run
        for name, node in producers.items():
            if name not in out.columns or not out.schema[name].is_float():
                continue
            nans, infs = out[name].is_nan().sum(), out[name].is_infinite().sum()
            if nans or infs:
                yield ctx.finding(node, f"output {name!r} has {nans} NaN and {infs} infinite row(s) on this data",
                                  fix="guard the division, log or sqrt that produces it, or fill the input that feeds it",
                                  column=name, nans=nans, infs=infs)


class ConstantOutput(Check):
    """An output column with one value on every row of the data.

    Only outputs are checked; a branch arm that no row takes is not detected.
    """

    type: Literal["constant_output"] = "constant_output"
    code = "DCD506"
    needs_data = True
    executes = True
    severity: Severity = Severity.INFO

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        if (run := ctx.outputs) is None or run[0].height < 2:
            return
        out, producers = run
        for name, node in producers.items():
            if name in out.columns and out[name].n_unique() == 1:
                yield ctx.finding(node, f"output {name!r} is {out[name][0]!r} on all {out.height} rows of this data",
                                  fix="check the data covers the cases the step distinguishes, or drop the output",
                                  column=name, value=out[name][0])


# An expression's value is `(lo, hi, is_int)`, or `None` when unknown, which is never flagged.


class Int64Range(Check):
    """Observed column ranges pushed through `+ - *` in step source; reports results beyond int64.

    Integer results past 2^63 wrap silently; a float accumulator `+=` of a term past 2^53 drops units.
    Only expressions over input columns, int and float constants and earlier assignments are
    evaluated; anything else (calls, branches' assignments, loops) is unknown and never flagged.

    Example::

        checks.run(pipeline, ["int64_range"], data=frame)
    """

    type: Literal["int64_range"] = "int64_range"
    code = "DCD503"
    needs_data = True

    def run(self, ctx: CheckContext) -> Iterator[Finding]:
        for node in ctx.calls():
            src = ctx.source(node)
            if src is None:
                continue
            env: dict[str, Any] = {}
            for i in node.inputs or ():
                if i.name in ctx.frame.columns:
                    env[i.arg] = _observed(ctx.frame[i.name], _is_int(i.annotation))
            fn = next((n for n in ast.walk(src.tree) if isinstance(n, ast.FunctionDef)), None)
            if fn is None:
                continue
            for expr, lo, hi, limit in _overflows(fn.body, env):
                text = ast.unparse(expr)
                kind = "int64" if limit == _INT64 else "float64 exact-integer"
                yield ctx.finding(node, f"{node.origin.path}: {text} reaches [{lo}, {hi}] on this data, beyond the {kind} range",
                                  fix="compute in a wider type, rescale the inputs, or bound them",
                                  line=src.line(expr), expression=text, lo=lo, hi=hi)


def _is_int(annotation: Any) -> bool:
    return typing.get_args(annotation)[0] is int if typing.get_origin(annotation) is typing.Annotated else annotation is int


def _observed(col: pl.Series, is_int: bool) -> Any:
    if not (col.dtype.is_integer() or col.dtype.is_float()) or col.drop_nulls().len() == 0:
        return None
    lo, hi = col.min(), col.max()
    if isinstance(lo, float) and not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return lo, hi, is_int and col.dtype.is_integer()


def _overflows(body: list[ast.stmt], env: dict[str, Any], nested: bool = False) -> Iterator[tuple[ast.expr, Any, Any, int]]:
    for stmt in body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            yield from _eval_flag(stmt.value, env)
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                known = isinstance(t, ast.Name) and not nested
                for name in ast.walk(t):
                    if isinstance(name, ast.Name):
                        env[name.id] = _eval(stmt.value, env) if known else None
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            yield from _eval_flag(stmt.value, env)
            term = _eval(stmt.value, env)
            if isinstance(stmt.op, ast.Add) and term and not term[2] and max(abs(term[0]), abs(term[1])) > _FLOAT_EXACT:
                yield stmt.value, term[0], term[1], _FLOAT_EXACT
            env[stmt.target.id] = None
        elif isinstance(stmt, (ast.Return, ast.Expr)) and stmt.value is not None:
            yield from _eval_flag(stmt.value, env)
        elif isinstance(stmt, (ast.If, ast.For, ast.While)):
            # ponytail: assignments inside branches and loops make the name unknown instead of a joined range
            for name in ast.walk(stmt):
                if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                    env[name.id] = None
            for block in (stmt.body, stmt.orelse):
                yield from _overflows(block, env, nested=True)


def _eval_flag(expr: ast.expr, env: dict[str, Any]) -> Iterator[tuple[ast.expr, Any, Any, int]]:
    for sub in ast.walk(expr):
        if isinstance(sub, ast.BinOp):
            r = _eval(sub, env)
            if r and r[2] and max(abs(r[0]), abs(r[1])) >= _INT64:
                yield sub, r[0], r[1], _INT64


def _eval(expr: ast.expr, env: dict[str, Any]) -> Any:
    if isinstance(expr, ast.Constant) and type(expr.value) in (int, float):
        return expr.value, expr.value, isinstance(expr.value, int)
    if isinstance(expr, ast.Name):
        return env.get(expr.id)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        r = _eval(expr.operand, env)
        return r and (-r[1], -r[0], r[2])
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub, ast.Mult)):
        a, b = _eval(expr.left, env), _eval(expr.right, env)
        if a is None or b is None:
            return None
        kind = a[2] and b[2]
        if isinstance(expr.op, ast.Add):
            return a[0] + b[0], a[1] + b[1], kind
        if isinstance(expr.op, ast.Sub):
            return a[0] - b[1], a[1] - b[0], kind
        products = [x * y for x in a[:2] for y in b[:2]]
        return min(products), max(products), kind
    return None
