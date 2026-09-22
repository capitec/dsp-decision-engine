"""A closed expression grammar, compiled to numba source at build time.

The owner's ask (verbatim): "for decision trees it is important to allow
user expressions for some of the items. like x-y>10 it seems a pitty to now
have to write preprocessing code just to calculate x-y". Doc 08 §3.2 and doc
06 §O15 removed decider 1's `_ComputedFeature` entirely, citing "config may
not contain code" (doc 08 §1). That conflated two different things:

* a `{module_name, function_name}` **pointer** — `getattr` on an unregistered
  dotted path, resolved at runtime, with no declared interface and no
  schema. Doc 01 §5.4 records what this cost decider 1. It stays banned —
  nothing below reintroduces it.
* a **restricted expression** — `_ComputedFeature`
  (`decider/modules/rules/common/feature.py:59`) already had this shape:
  `ALLOWED_POLARS_FUNCTIONS` is a whitelist, and
  `extract_features_and_parameters` is static dependency extraction. It was
  never a pointer.

The real objection to `_ComputedFeature` was narrower and mechanical: it
called `simple_eval` to build a `polars.Expr` **at runtime, on every row
batch**. That is a live interpreter on the request path — the thing doc 05's
whole design exists to avoid — and it cannot produce anything a numba kernel
can call anyway (a `polars.Expr` is not numba-compilable source).

This module removes the runtime step instead of the feature: `parse()` walks
the expression with `ast`, validates it against the closed grammar below,
and returns an `Expr` whose `.emit()` renders numba-compilable **source
text** once, at build time. There is no interpreter left at all — every
`Expr` becomes a fragment of a real `.py` file that gets `njit`'d, cached
and imported like any other step. This is
strictly safer than decider 1, which ran `simpleeval` on every call.

**The grammar is closed, not open.** Every admitted `ast` node kind is one
class below, each of which validates itself, reports its own
`dependencies()` and emits its own source — the same "class owns its own
behaviour" pattern `decider2.trees.schema` uses for tree nodes and
conditions (a callable-keyed registry resolves which class handles which
`ast` node type; there is no `isinstance` chain doing that dispatch). Ask
the module a question it cannot answer — a node kind, an operator, or a call
target this module does not recognise — and `parse()` raises `ExprError`
naming the offending construct and where it is in the source, at PARSE
time, before any of it ever reaches source generation.

Admitted: bare names (dependencies — a feature or a param, resolved by
whatever `ExprContext` `.emit()` is given), numeric literals, the binary
operators `+ - * / // % **`, unary minus, the comparisons
`< <= > >= == !=` (one per `Compare` node — chained comparisons like
`a < b < c` are not admitted; write `a < b and b < c`), `and`/`or`/`not`,
parentheses (structural — `ast` does not retain them, so nothing to admit
explicitly), and calls to a three-function whitelist: `min`, `max`, `abs`
— all three compile in numba's nopython mode.

Rejected, each at parse, each naming the construct and its position: a call
to anything outside the whitelist, attribute access, subscripts, lambdas,
comprehensions and generator expressions, the walrus operator, f-strings,
star-args, list/tuple/dict/set literals, conditional expressions, any dunder
name, anything `ast.parse` itself refuses (e.g. `import`, which cannot even
appear inside a single `eval`-mode expression), and an expression nested
deeper than this module is willing to recurse into.

**No decision-relevant constant reaches emitted source here either** — same
rule as `decider2.trees.codegen` (doc 05 §4.2). A numeric literal inside an
expression does not become a Python literal in the generated `.py` file; it
becomes a name that resolves to a `param()`-declared kernel argument
(`ExprContext.constant`), so retuning a constant buried inside an expression
is a values change, not a recompile — exactly like any other threshold.
"""
from __future__ import annotations

import ast
import typing as t
from dataclasses import dataclass, field
from functools import reduce

from numba import njit

__all__ = ["ExprError", "ExprContext", "ExprCompileContext", "Expr", "parse"]

# `ast.parse` itself will refuse something absurd long before this, but a
# left-leaning chain of binary operators (`a + b + c + ...`) recurses once
# per term with no help from `ast.parse`'s own limits, and CPython's default
# recursion limit is 1000 — so this module enforces its own, smaller cap and
# names the expression rather than letting a bare `RecursionError` (or a
# segfault from a C-level parser doing the same, on a big enough input)
# surface with no context. 200 is comfortably above anything a "closed
# expression grammar for a rule's feature" should ever need.
_MAX_DEPTH = 200


class ExprError(ValueError):
    """`source` is outside the admitted grammar — raised at `parse()`, never
    at emit time (doc 06 §O15, doc 08 §1/§3.2: a closed, statically
    validated grammar, compiled at build time, is admitted; a runtime
    evaluator is not, and this exception is how the line is enforced)."""


class ExprContext(t.Protocol):
    """What `Expr.emit` needs from its caller — two names, not a class.

    `decider2.trees.schema`'s per-use adapter is the only implementation
    today (`_ExprEmitAdapter`, wrapping a tree's `EmitContext`), but nothing
    here imports `trees` — anything providing these two methods can consume
    an `Expr`, which is the whole point of keeping this module standalone.
    """

    def name(self, ident: str) -> str:
        """The numba-source token for the free name `ident` — a feature
        column, a param, whatever the caller's own vocabulary resolves it
        to. Called once per occurrence; a caller that de-duplicates repeat
        uses of the same name (as `decider2.trees.codegen.EmitContext`
        does) may return the same token every time."""
        ...

    def constant(self, value: "int | float") -> str:
        """The numba-source token for the numeric literal `value` — never
        the literal itself. Doc 05 §4.2's rule, extended to expressions: no
        decision-relevant constant is emitted into driver source, so a
        caller registers `value` as a retunable kernel argument (a
        `param()`) and returns the name it was given."""
        ...


class ExprCompileContext(t.Protocol):
    """What `Expr.compile` needs from its caller — the closure-building
    counterpart of `ExprContext` (above). Stage 2 of the codegen-elimination
    migration: `.emit()` still renders numba SOURCE TEXT and is kept only so
    nothing that already calls it breaks; `.compile()` is what
    `decider2.trees.schema._ExprEmitAdapter` now actually uses, and it never
    produces a string. `feats`/`thresholds` name the SAME two homogeneous
    tuples `decider2.trees.interpreter.walk_tree` takes — both are UniTuples
    (every entry is a plain `float64`), so numba indexes them with an
    ordinary RUNTIME int with no "must be a compile-time constant" caveat
    (unlike a heterogeneous tuple), which is what lets `.compile()` close
    over a plain Python `int` slot rather than needing one hand-written
    closure per slot position.
    """

    def name_index(self, ident: str) -> int:
        """`ident`'s slot in the `feats` tuple a compiled expression will be
        called with."""
        ...

    def constant_index(self, value: "int | float") -> int:
        """`value`'s slot in the `thresholds` tuple a compiled expression
        will be called with — registered as a retunable kernel argument
        (never baked in as a literal), the same rule `ExprContext.constant`
        documents for the text path."""
        ...


# ---------------------------------------------------------------------------
# Real, hand-written njit closures for every admitted operator/function —
# composed by `Expr.compile` below instead of spliced into source text.
# Exactly the grammar `_BINOPS`/`_UNARYOPS`/`_CMPOPS`/`_WHITELISTED_FUNCS`
# already admit; no operator reaches here that `parse()` didn't already
# validate.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _op_add(a, b):
    return a + b


@njit(cache=True)
def _op_sub(a, b):
    return a - b


@njit(cache=True)
def _op_mul(a, b):
    return a * b


@njit(cache=True)
def _op_div(a, b):
    return a / b


@njit(cache=True)
def _op_floordiv(a, b):
    return a // b


@njit(cache=True)
def _op_mod(a, b):
    return a % b


@njit(cache=True)
def _op_pow(a, b):
    return a ** b


_BINOP_FNS: "dict[str, t.Any]" = {
    "+": _op_add, "-": _op_sub, "*": _op_mul, "/": _op_div,
    "//": _op_floordiv, "%": _op_mod, "**": _op_pow,
}


@njit(cache=True)
def _op_neg(a):
    return -a


@njit(cache=True)
def _op_not(a):
    return 0.0 if a != 0.0 else 1.0


_UNARYOP_FNS: "dict[str, t.Any]" = {"-": _op_neg, "not ": _op_not}


@njit(cache=True)
def _cmp_lt(a, b):
    return 1.0 if a < b else 0.0


@njit(cache=True)
def _cmp_le(a, b):
    return 1.0 if a <= b else 0.0


@njit(cache=True)
def _cmp_gt(a, b):
    return 1.0 if a > b else 0.0


@njit(cache=True)
def _cmp_ge(a, b):
    return 1.0 if a >= b else 0.0


@njit(cache=True)
def _cmp_eq(a, b):
    return 1.0 if a == b else 0.0


@njit(cache=True)
def _cmp_ne(a, b):
    return 1.0 if a != b else 0.0


_CMPOP_FNS: "dict[str, t.Any]" = {
    "<": _cmp_lt, "<=": _cmp_le, ">": _cmp_gt, ">=": _cmp_ge, "==": _cmp_eq, "!=": _cmp_ne,
}


@njit(cache=True)
def _bool_and(a, b):
    return 1.0 if (a != 0.0 and b != 0.0) else 0.0


@njit(cache=True)
def _bool_or(a, b):
    return 1.0 if (a != 0.0 or b != 0.0) else 0.0


_BOOLOP_FNS: "dict[str, t.Any]" = {"and": _bool_and, "or": _bool_or}


@njit(cache=True)
def _fn_min2(a, b):
    return a if a < b else b


@njit(cache=True)
def _fn_max2(a, b):
    return a if a > b else b


@njit(cache=True)
def _fn_abs(a):
    return abs(a)


_CALL_FNS: "dict[str, t.Any]" = {"min": _fn_min2, "max": _fn_max2, "abs": _fn_abs}


# ---------------------------------------------------------------------------
# Dispatch — a registry keyed on the `ast` node type, not an isinstance chain
# ---------------------------------------------------------------------------

_REGISTRY: "dict[type[ast.AST], type[Expr]]" = {}


def _admits(ast_type: "type[ast.AST]"):
    """Register the decorated `Expr` subclass as the handler for `ast_type`.

    This is the one place an `ast` node type is associated with the class
    that understands it. `Expr.build` below looks the association up in
    `_REGISTRY`; it never asks `isinstance(node, ast.Whatever)`.
    """

    def decorate(cls: "type[Expr]") -> "type[Expr]":
        _REGISTRY[ast_type] = cls
        return cls

    return decorate


_FRIENDLY_NAMES: "dict[type[ast.AST], str]" = {
    ast.Attribute: "attribute access",
    ast.Subscript: "subscript",
    ast.Slice: "slice",
    ast.Lambda: "lambda",
    ast.ListComp: "list comprehension",
    ast.SetComp: "set comprehension",
    ast.DictComp: "dict comprehension",
    ast.GeneratorExp: "generator expression",
    ast.NamedExpr: "walrus assignment (:=)",
    ast.JoinedStr: "f-string",
    ast.FormattedValue: "f-string",
    ast.Starred: "star-args (*args)",
    ast.List: "list literal",
    ast.Tuple: "tuple literal",
    ast.Dict: "dict literal",
    ast.Set: "set literal",
    ast.IfExp: "conditional expression (a ternary 'x if c else y')",
    ast.Await: "await",
    ast.Yield: "yield",
    ast.YieldFrom: "yield from",
}


def _pos(node: ast.AST, source: str) -> str:
    lineno = getattr(node, "lineno", 1)
    col = getattr(node, "col_offset", 0) + 1
    return f"line {lineno}, column {col} of {source!r}"


def _reject(node: ast.AST, source: str, reason: str) -> "ExprError":
    return ExprError(f"{reason} at {_pos(node, source)}")


# ---------------------------------------------------------------------------
# Expr — the base every admitted node kind implements
# ---------------------------------------------------------------------------


class Expr:
    """One admitted `ast` node, validated, with its own dependencies and its
    own emission. Never instantiate a subclass directly — build a tree with
    `Expr.build` (internally) or, from the top, `parse()`."""

    @classmethod
    def build(cls, node: ast.AST, source: str, *, depth: int = 0) -> "Expr":
        """Resolve `node` to the `Expr` subclass registered for its type and
        let that subclass validate and build itself — the one dispatch
        point, by registry lookup rather than an isinstance chain."""
        if depth > _MAX_DEPTH:
            raise _reject(
                node, source,
                f"expression nests more than {_MAX_DEPTH} levels deep — split it "
                "into a named step in code instead of one expression",
            )
        handler = _REGISTRY.get(type(node))
        if handler is None:
            friendly = _FRIENDLY_NAMES.get(type(node), f"a {type(node).__name__} construct")
            raise _reject(node, source, f"{friendly} is not admitted in a decider2 expression")
        return handler.from_node(node, source, depth)

    @classmethod
    def from_node(cls, node: ast.AST, source: str, depth: int) -> "Expr":
        raise NotImplementedError

    def dependencies(self) -> "set[str]":
        """Every free name this expression reads — decider 1's
        `extract_names_and_parameters` (`decider/modules/rules/common/
        feature.py`), static, at parse time, never lost by the closed
        grammar replacing the interpreter it used to run on."""
        raise NotImplementedError

    def emit(self, ctx: ExprContext) -> str:
        """This expression's numba-source fragment, resolving every name
        and every numeric literal through `ctx` — never a bare literal,
        never an interpreter call.

        Kept only so an existing caller of `.emit()` (this module's own
        tests) still gets the same string back; `decider2.trees.schema` no
        longer calls it — see `.compile()`, which is what a tree's own
        kernel now uses."""
        raise NotImplementedError

    def compile(self, ctx: ExprCompileContext):
        """A real `@njit` closure — `fn(feats, thresholds) -> float64` —
        equivalent to `.emit()`'s string, built by composing this node's
        children's own closures instead of splicing their text together.
        `feats`/`thresholds` are the same two homogeneous tuples
        `decider2.trees.interpreter.walk_tree` takes; every name and every
        numeric literal is resolved through `ctx` to a slot in one of them
        (never a Python literal baked into the closure), same rule as
        `.emit()`."""
        raise NotImplementedError


def _is_dunder(ident: str) -> bool:
    return len(ident) >= 4 and ident.startswith("__") and ident.endswith("__")


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


@_admits(ast.Name)
@dataclass(frozen=True)
class _NameExpr(Expr):
    id: str

    @classmethod
    def from_node(cls, node: ast.Name, source: str, depth: int) -> "_NameExpr":
        if _is_dunder(node.id):
            raise _reject(node, source, f"dunder name {node.id!r} is not admitted")
        return cls(id=node.id)

    def dependencies(self) -> "set[str]":
        return {self.id}

    def emit(self, ctx: ExprContext) -> str:
        return ctx.name(self.id)

    def compile(self, ctx: ExprCompileContext):
        idx = ctx.name_index(self.id)

        @njit(cache=True)
        def fn(feats, thresholds):
            return feats[idx]

        return fn


@_admits(ast.Constant)
@dataclass(frozen=True)
class _NumExpr(Expr):
    value: "int | float"

    @classmethod
    def from_node(cls, node: ast.Constant, source: str, depth: int) -> "_NumExpr":
        value = node.value
        if isinstance(value, bool):
            raise _reject(
                node, source,
                f"boolean literal {value!r} is not admitted (only numeric literals are; "
                "use a name and compare it, e.g. 'flag == 1')",
            )
        if isinstance(value, (int, float)):
            return cls(value=value)
        kind = type(value).__name__
        raise _reject(node, source, f"{kind} literal {value!r} is not admitted (only numeric literals are)")

    def dependencies(self) -> "set[str]":
        return set()

    def emit(self, ctx: ExprContext) -> str:
        return ctx.constant(self.value)

    def compile(self, ctx: ExprCompileContext):
        idx = ctx.constant_index(self.value)

        @njit(cache=True)
        def fn(feats, thresholds):
            return thresholds[idx]

        return fn


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

_BINOPS: "dict[type[ast.operator], str]" = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
}


@_admits(ast.BinOp)
@dataclass(frozen=True)
class _BinOpExpr(Expr):
    op: str
    left: Expr
    right: Expr

    @classmethod
    def from_node(cls, node: ast.BinOp, source: str, depth: int) -> "_BinOpExpr":
        symbol = _BINOPS.get(type(node.op))
        if symbol is None:
            raise _reject(
                node, source,
                f"operator {type(node.op).__name__!r} is not admitted (only + - * / // % ** are)",
            )
        left = Expr.build(node.left, source, depth=depth + 1)
        right = Expr.build(node.right, source, depth=depth + 1)
        return cls(op=symbol, left=left, right=right)

    def dependencies(self) -> "set[str]":
        return self.left.dependencies() | self.right.dependencies()

    def emit(self, ctx: ExprContext) -> str:
        return f"({self.left.emit(ctx)} {self.op} {self.right.emit(ctx)})"

    def compile(self, ctx: ExprCompileContext):
        left_fn = self.left.compile(ctx)
        right_fn = self.right.compile(ctx)
        op_fn = _BINOP_FNS[self.op]

        @njit(cache=True)
        def fn(feats, thresholds):
            return op_fn(left_fn(feats, thresholds), right_fn(feats, thresholds))

        return fn


_UNARYOPS: "dict[type[ast.unaryop], str]" = {ast.USub: "-", ast.Not: "not "}


@_admits(ast.UnaryOp)
@dataclass(frozen=True)
class _UnaryOpExpr(Expr):
    op: str
    operand: Expr

    @classmethod
    def from_node(cls, node: ast.UnaryOp, source: str, depth: int) -> "_UnaryOpExpr":
        symbol = _UNARYOPS.get(type(node.op))
        if symbol is None:
            raise _reject(
                node, source,
                f"unary operator {type(node.op).__name__!r} is not admitted "
                "(only unary minus and 'not' are)",
            )
        operand = Expr.build(node.operand, source, depth=depth + 1)
        return cls(op=symbol, operand=operand)

    def dependencies(self) -> "set[str]":
        return self.operand.dependencies()

    def emit(self, ctx: ExprContext) -> str:
        return f"({self.op}{self.operand.emit(ctx)})"

    def compile(self, ctx: ExprCompileContext):
        operand_fn = self.operand.compile(ctx)
        op_fn = _UNARYOP_FNS[self.op]

        @njit(cache=True)
        def fn(feats, thresholds):
            return op_fn(operand_fn(feats, thresholds))

        return fn


_CMPOPS: "dict[type[ast.cmpop], str]" = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
}


@_admits(ast.Compare)
@dataclass(frozen=True)
class _CompareExpr(Expr):
    op: str
    left: Expr
    right: Expr

    @classmethod
    def from_node(cls, node: ast.Compare, source: str, depth: int) -> "_CompareExpr":
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise _reject(
                node, source,
                "a chained comparison (e.g. 'a < b < c') is not admitted; "
                "write it as 'a < b and b < c'",
            )
        symbol = _CMPOPS.get(type(node.ops[0]))
        if symbol is None:
            raise _reject(
                node, source,
                f"comparison {type(node.ops[0]).__name__!r} is not admitted "
                "(only < <= > >= == != are — not 'is'/'in')",
            )
        left = Expr.build(node.left, source, depth=depth + 1)
        right = Expr.build(node.comparators[0], source, depth=depth + 1)
        return cls(op=symbol, left=left, right=right)

    def dependencies(self) -> "set[str]":
        return self.left.dependencies() | self.right.dependencies()

    def emit(self, ctx: ExprContext) -> str:
        return f"({self.left.emit(ctx)} {self.op} {self.right.emit(ctx)})"

    def compile(self, ctx: ExprCompileContext):
        left_fn = self.left.compile(ctx)
        right_fn = self.right.compile(ctx)
        op_fn = _CMPOP_FNS[self.op]

        @njit(cache=True)
        def fn(feats, thresholds):
            return op_fn(left_fn(feats, thresholds), right_fn(feats, thresholds))

        return fn


@_admits(ast.BoolOp)
@dataclass(frozen=True)
class _BoolOpExpr(Expr):
    op: str
    values: "tuple[Expr, ...]"

    @classmethod
    def from_node(cls, node: ast.BoolOp, source: str, depth: int) -> "_BoolOpExpr":
        symbol = "and" if isinstance(node.op, ast.And) else "or"
        values = tuple(Expr.build(v, source, depth=depth + 1) for v in node.values)
        return cls(op=symbol, values=values)

    def dependencies(self) -> "set[str]":
        out: "set[str]" = set()
        for v in self.values:
            out |= v.dependencies()
        return out

    def emit(self, ctx: ExprContext) -> str:
        return "(" + f" {self.op} ".join(v.emit(ctx) for v in self.values) + ")"

    def compile(self, ctx: ExprCompileContext):
        op_fn = _BOOLOP_FNS[self.op]
        value_fns = tuple(v.compile(ctx) for v in self.values)

        def _pair(left_fn, right_fn):
            @njit(cache=True)
            def fn(feats, thresholds):
                return op_fn(left_fn(feats, thresholds), right_fn(feats, thresholds))

            return fn

        return reduce(_pair, value_fns)


# ---------------------------------------------------------------------------
# Calls — the three-function whitelist
# ---------------------------------------------------------------------------

# name -> (min args, max args or None for unbounded). All three compile in
# numba's nopython mode with plain scalar arguments.
_WHITELISTED_FUNCS: "dict[str, tuple[int, int | None]]" = {
    "min": (2, None),
    "max": (2, None),
    "abs": (1, 1),
}


@_admits(ast.Call)
@dataclass(frozen=True)
class _CallExpr(Expr):
    func: str
    args: "tuple[Expr, ...]"

    @classmethod
    def from_node(cls, node: ast.Call, source: str, depth: int) -> "_CallExpr":
        if not isinstance(node.func, ast.Name):
            raise _reject(
                node, source,
                "only a call to a plain whitelisted function name is admitted "
                "(no attribute, subscript or other dynamic call target)",
            )
        name = node.func.id
        if name not in _WHITELISTED_FUNCS:
            whitelist = ", ".join(sorted(_WHITELISTED_FUNCS))
            raise _reject(
                node, source,
                f"call to {name!r} is not admitted; only {whitelist} are whitelisted",
            )
        if node.keywords:
            raise _reject(node, source, f"keyword arguments to {name!r}() are not admitted")
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise _reject(a, source, "star-args (*args) are not admitted")
        lo, hi = _WHITELISTED_FUNCS[name]
        n = len(node.args)
        if n < lo or (hi is not None and n > hi):
            arity = f"exactly {lo}" if lo == hi else f"at least {lo}"
            raise _reject(node, source, f"{name}() takes {arity} argument(s); got {n}")
        args = tuple(Expr.build(a, source, depth=depth + 1) for a in node.args)
        return cls(func=name, args=args)

    def dependencies(self) -> "set[str]":
        out: "set[str]" = set()
        for a in self.args:
            out |= a.dependencies()
        return out

    def emit(self, ctx: ExprContext) -> str:
        return f"{self.func}({', '.join(a.emit(ctx) for a in self.args)})"

    def compile(self, ctx: ExprCompileContext):
        arg_fns = tuple(a.compile(ctx) for a in self.args)
        if self.func == "abs":
            (operand_fn,) = arg_fns
            abs_fn = _CALL_FNS["abs"]

            @njit(cache=True)
            def fn(feats, thresholds):
                return abs_fn(operand_fn(feats, thresholds))

            return fn

        # min/max: a 2-argument primitive (_fn_min2/_fn_max2), folded
        # pairwise over however many arguments this call actually has —
        # `_WHITELISTED_FUNCS["min"/"max"] = (2, None)` admits 2 or more.
        pair_fn = _CALL_FNS[self.func]

        def _pair(left_fn, right_fn):
            @njit(cache=True)
            def fn(feats, thresholds):
                return pair_fn(left_fn(feats, thresholds), right_fn(feats, thresholds))

            return fn

        return reduce(_pair, arg_fns)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse(source: str) -> Expr:
    """Parse and validate `source` against the closed grammar this module
    admits, returning a validated `Expr`. Raises `ExprError`, naming the
    offending construct and its position, for anything outside the grammar
    — including a plain `SyntaxError` from `ast.parse` itself (e.g. `import`,
    which cannot occur in a single expression at all).

    Nothing here evaluates `source`. This is the entire runtime cost of a
    computed feature going forward: one `ast.parse` and a walk, at load
    time, never per row.
    """
    if not isinstance(source, str) or not source.strip():
        raise ExprError(f"expression is empty or not a string: {source!r}")
    try:
        module = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ExprError(
            f"{source!r} is not a valid expression: {e.msg} "
            f"(line {e.lineno}, column {(e.offset or 0)})"
        ) from e
    return Expr.build(module.body, source)
