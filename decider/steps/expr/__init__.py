"""A closed arithmetic expression grammar for computed tree features."""
from __future__ import annotations

import ast
import operator
import typing as t
from dataclasses import dataclass

__all__ = ["Expr", "ExprError", "parse"]

MAX_DEPTH = 200

BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
UNARYOPS = {ast.USub: operator.neg, ast.Not: operator.not_}
CMPOPS = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
# name -> (function, min args, max args or None)
FUNCS: dict[str, tuple[t.Callable, int, int | None]] = {
    "min": (min, 2, None), "max": (max, 2, None), "abs": (abs, 1, 1),
}

_FRIENDLY = {
    ast.Attribute: "attribute access",
    ast.Subscript: "subscript",
    ast.Lambda: "lambda",
    ast.ListComp: "list comprehension",
    ast.SetComp: "set comprehension",
    ast.DictComp: "dict comprehension",
    ast.GeneratorExp: "generator expression",
    ast.NamedExpr: "walrus assignment (:=)",
    ast.JoinedStr: "f-string",
    ast.Starred: "star-args (*args)",
    ast.List: "list literal",
    ast.Tuple: "tuple literal",
    ast.Dict: "dict literal",
    ast.Set: "set literal",
    ast.IfExp: "conditional expression (a ternary 'x if c else y')",
    ast.Await: "await",
}


class ExprError(ValueError):
    """The expression is outside the admitted grammar."""


@dataclass(frozen=True)
class Expr:
    """A parsed, validated expression. Build one with `parse`.

    `node` is the validated `ast` tree; the compiler walks it directly.
    """

    source: str
    node: ast.expr

    def dependencies(self) -> set[str]:
        """Every free name the expression reads.

        >>> parse("min(a, b) - 2").dependencies() == {"a", "b"}
        True
        """
        called = {id(n.func) for n in ast.walk(self.node) if isinstance(n, ast.Call)}
        return {n.id for n in ast.walk(self.node) if isinstance(n, ast.Name) and id(n) not in called}

    def evaluate(self, names: t.Mapping[str, t.Any]) -> t.Any:
        """Evaluate against `names`; comparisons and `and`/`or`/`not` give bools.

        >>> parse("x - y > 10").evaluate({"x": 25.0, "y": 10.0})
        True
        """
        return _eval(self.node, names)


def parse(source: str) -> Expr:
    """Parse and validate `source`, or raise `ExprError` naming the offending construct.

    Admitted: names, int/float literals, `+ - * / // % **`, unary `-`,
    one comparison per `< <= > >= == !=`, `and`/`or`/`not`, parentheses,
    and calls to `min`, `max` and `abs`. Nothing is ever evaluated here.

    >>> parse("income - expenses").dependencies() == {"income", "expenses"}
    True
    """
    if not isinstance(source, str) or not source.strip():
        raise ExprError(f"expression is empty or not a string: {source!r}")
    try:
        body = ast.parse(source, mode="eval").body
    except SyntaxError as e:
        raise ExprError(f"{source!r} is not a valid expression: {e.msg} (line {e.lineno}, column {e.offset or 0})") from e
    _check(body, source, 0)
    return Expr(source, body)


def _reject(node: ast.AST, source: str, reason: str) -> ExprError:
    where = f"line {getattr(node, 'lineno', 1)}, column {getattr(node, 'col_offset', 0) + 1} of {source!r}"
    return ExprError(f"{reason} at {where}")


def _check(node: ast.AST, source: str, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise _reject(node, source, f"expression nests more than {MAX_DEPTH} levels deep; split it into a step")
    children: t.Iterable[ast.AST] = ()
    if isinstance(node, ast.Name):
        if len(node.id) >= 4 and node.id.startswith("__") and node.id.endswith("__"):
            raise _reject(node, source, f"dunder name {node.id!r} is not admitted")
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise _reject(node, source, f"boolean literal {node.value!r} is not admitted (compare a name instead, e.g. 'flag == 1')")
        if not isinstance(node.value, (int, float)):
            raise _reject(node, source, f"{type(node.value).__name__} literal {node.value!r} is not admitted (only numbers are)")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in BINOPS:
            raise _reject(node, source, f"operator {type(node.op).__name__!r} is not admitted (only + - * / // % ** are)")
        children = (node.left, node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in UNARYOPS:
            raise _reject(node, source, f"unary operator {type(node.op).__name__!r} is not admitted (only - and 'not' are)")
        children = (node.operand,)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise _reject(node, source, "a chained comparison (e.g. 'a < b < c') is not admitted; write 'a < b and b < c'")
        if type(node.ops[0]) not in CMPOPS:
            raise _reject(node, source, f"comparison {type(node.ops[0]).__name__!r} is not admitted (only < <= > >= == != are)")
        children = (node.left, node.comparators[0])
    elif isinstance(node, ast.BoolOp):
        children = node.values
    elif isinstance(node, ast.Call):
        _check_call(node, source)
        children = node.args
    else:
        what = _FRIENDLY.get(type(node), f"a {type(node).__name__} construct")
        raise _reject(node, source, f"{what} is not admitted in an expression")
    for child in children:
        _check(child, source, depth + 1)


def _check_call(node: ast.Call, source: str) -> None:
    if not isinstance(node.func, ast.Name):
        raise _reject(node, source, "only a call to a plain whitelisted function name is admitted")
    name = node.func.id
    if name not in FUNCS:
        raise _reject(node, source, f"call to {name!r} is not admitted; only {', '.join(sorted(FUNCS))} are whitelisted")
    if node.keywords:
        raise _reject(node, source, f"keyword arguments to {name!r}() are not admitted")
    for arg in node.args:
        if isinstance(arg, ast.Starred):
            raise _reject(arg, source, "star-args (*args) are not admitted")
    _, lo, hi = FUNCS[name]
    n = len(node.args)
    if n < lo or (hi is not None and n > hi):
        arity = f"exactly {lo}" if lo == hi else f"at least {lo}"
        raise _reject(node, source, f"{name}() takes {arity} argument(s); got {n}")


def _eval(node: ast.AST, names: t.Mapping[str, t.Any]) -> t.Any:
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return BINOPS[type(node.op)](_eval(node.left, names), _eval(node.right, names))
    if isinstance(node, ast.UnaryOp):
        return UNARYOPS[type(node.op)](_eval(node.operand, names))
    if isinstance(node, ast.Compare):
        return CMPOPS[type(node.ops[0])](_eval(node.left, names), _eval(node.comparators[0], names))
    if isinstance(node, ast.BoolOp):
        values = (bool(_eval(v, names)) for v in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    fn = FUNCS[node.func.id][0]
    return fn(*(_eval(a, names) for a in node.args))
