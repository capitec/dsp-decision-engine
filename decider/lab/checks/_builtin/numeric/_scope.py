from __future__ import annotations

import ast
import numbers
import typing
from functools import reduce
from typing import Any, Iterator

from decider.lab.checks._base import CheckContext
from decider.ir import CallNode, decimal_info

INT, FLOAT, DEC = "int", "float", "decimal"
_FLOAT_FUNCS = {"sqrt", "exp", "log", "log2", "log10", "log1p", "sin", "cos", "tan", "pow", "fabs"}
_INT_FUNCS = {"floor", "ceil", "trunc"}


def kind_of(annotation: Any) -> str | None:
    """`INT`, `FLOAT`, `DEC` (a Decimal or Currency) or `None` (unknown) for a declared type; `bool` is unknown."""
    if decimal_info(annotation) is not None:
        return DEC
    if typing.get_origin(annotation) is typing.Annotated:
        annotation = annotation.__origin__
    if not isinstance(annotation, type) or issubclass(annotation, bool):
        return None
    if issubclass(annotation, numbers.Integral):
        return INT
    return FLOAT if issubclass(annotation, numbers.Real) else None


def join(a: str | None, b: str | None) -> str | None:
    return a if a == b else FLOAT if {a, b} == {INT, FLOAT} else DEC if {a, b} == {DEC, INT} else None


def _positive(items: Any) -> bool:
    for m in items or ():
        if _positive(getattr(m, "metadata", None)):
            return True
        gt, ge = getattr(m, "gt", None), getattr(m, "ge", None)
        if (gt is not None and gt >= 0) or (ge is not None and ge > 0):
            return True
    return False


def _meta(annotation: Any) -> tuple:
    return getattr(annotation, "__metadata__", ())


class Scope:
    """What the types and bounds say about the names in one step function's body."""

    def __init__(self, node: CallNode, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.node, self.fn = node, fn
        self.types: dict[str, str | None] = {}
        self.positive: set[str] = set()
        self.inputs: set[str] = set()
        for i in node.inputs or ():
            self._declare(i.arg, i.annotation, _meta(i.annotation))
            self.inputs.add(i.arg)
        for p in node.params:
            self._declare(p.arg, p.annotation, [p.field_info, *_meta(p.annotation)])
        for name, value in node.consts:
            if not isinstance(value, bool):
                self.types[name] = INT if isinstance(value, int) else FLOAT if isinstance(value, float) else None
        self.declared = set(self.types)
        self.parents = {c: p for p in ast.walk(fn) for c in ast.iter_child_nodes(p)}
        assigns = [(t.id, s.value) for s in self.walk() if isinstance(s, ast.Assign)
                   for t in s.targets if isinstance(t, ast.Name)]
        assigns += [(s.target.id, s.value) for s in self.walk()
                    if isinstance(s, ast.AnnAssign) and s.value and isinstance(s.target, ast.Name)]
        assigns += [(s.target.id, ast.Constant(1.0) if isinstance(s.op, ast.Div) else s.value) for s in self.walk()
                    if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name)]
        for _ in range(3):
            seen: dict[str, list[str | None]] = {}
            for name, value in assigns:
                if name not in self.declared:
                    seen.setdefault(name, []).append(self.type_of(value))
            for name, ts in seen.items():
                self.types[name] = reduce(join, ts)

    def _declare(self, name: str, annotation: Any, bounds: Any) -> None:
        self.types[name] = kind_of(annotation)
        if _positive(bounds):
            self.positive.add(name)

    def walk(self) -> Iterator[ast.AST]:
        for stmt in self.fn.body:
            yield from ast.walk(stmt)

    def type_of(self, e: ast.AST) -> str | None:
        if isinstance(e, ast.Constant):
            return None if isinstance(e.value, bool) else INT if isinstance(e.value, int) else FLOAT if isinstance(e.value, float) else None
        if isinstance(e, ast.Name):
            return self.types.get(e.id)
        if isinstance(e, ast.UnaryOp):
            return None if isinstance(e.op, ast.Not) else self.type_of(e.operand)
        if isinstance(e, ast.IfExp):
            return join(self.type_of(e.body), self.type_of(e.orelse))
        if isinstance(e, ast.BinOp):
            left, right = self.type_of(e.left), self.type_of(e.right)
            if DEC in (left, right):
                # `Decimal * float` is a Decimal at the decimal's scale; `float / Decimal` is not defined.
                return DEC if not isinstance(e.op, (ast.FloorDiv, ast.Mod, ast.Pow)) else None
            if isinstance(e.op, ast.Div) or FLOAT in (left, right):
                return FLOAT
            if isinstance(e.op, ast.Pow) and isinstance(e.right, ast.UnaryOp):
                return FLOAT
            return INT if left == right == INT else None
        if isinstance(e, ast.Call):
            name = e.func.id if isinstance(e.func, ast.Name) else e.func.attr if isinstance(e.func, ast.Attribute) else ""
            args = [self.type_of(a) for a in e.args]
            if name in ("int", "len") or name in _INT_FUNCS or (name == "round" and len(args) == 1):
                return INT
            if name == "float" or name in _FLOAT_FUNCS:
                return FLOAT
            if name in ("abs", "round") and args:
                return args[0]
            if name in ("min", "max") and args:
                return reduce(join, args)
        return None

    def nonzero(self, e: ast.AST) -> bool:
        """Provably non-zero from a constant, a positive-bounded param, or `max(x, positive)`."""
        if isinstance(e, ast.Constant):
            return isinstance(e.value, (int, float)) and e.value != 0
        if isinstance(e, ast.Name):
            return e.id in self.positive
        if isinstance(e, ast.UnaryOp):
            return self.nonzero(e.operand)
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Mult, ast.Pow)):
            return self.nonzero(e.left) and self.nonzero(e.right)
        if isinstance(e, ast.Call) and isinstance(e.func, (ast.Name, ast.Attribute)):
            name = e.func.id if isinstance(e.func, ast.Name) else e.func.attr
            if name in ("max", "maximum"):
                return any(isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) and a.value > 0 for a in e.args)
        return False

    def guarded(self, at: ast.AST, value: ast.AST) -> bool:
        """Whether an enclosing branch, or an earlier early-exit or assert, tests `value`."""
        text = ast.unparse(value)

        def mentions(test: ast.AST) -> bool:
            return any(ast.unparse(n) == text for n in ast.walk(test))

        cur = at
        while cur in self.parents:
            parent = self.parents[cur]
            if isinstance(parent, (ast.If, ast.IfExp, ast.While)) and cur is not parent.test and mentions(parent.test):
                return True
            for seq in (getattr(parent, "body", None), getattr(parent, "orelse", None)):
                if isinstance(seq, list) and cur in seq:
                    for prev in seq[:seq.index(cur)]:
                        if isinstance(prev, ast.Assert) and mentions(prev.test):
                            return True
                        if isinstance(prev, ast.If) and mentions(prev.test) and isinstance(prev.body[-1], (ast.Return, ast.Raise)):
                            return True
            cur = parent
        return False

    def money_output(self) -> bool:
        """Whether an output is an amount of money: a Decimal or Currency."""
        return any(decimal_info(o.annotation) is not None for o in self.node.outputs or ())


def scopes(ctx: CheckContext) -> Iterator[tuple[CallNode, Any, Scope]]:
    """Each authored step function with its parsed source and type scope."""
    for node in ctx.calls():
        src = ctx.source(node)
        fn = src.tree.body[0] if src and src.tree.body else None
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node, src, Scope(node, fn)
