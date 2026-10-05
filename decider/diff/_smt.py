"""Tree and decision-table steps as z3 terms, to ask whether two versions of a step can disagree.

What is modelled, exactly:
- float features are IEEE float64 (NaN is a null, as the walker reads it; infinities and -0.0 behave as in
  Python); int features are int64 with a separate null flag; bool and string features have a null flag too.
- tree conditions follow the Python walker: a test on a null is unknown, NOT keeps it unknown, AND and OR use
  three-valued logic, and a node takes its otherwise branch on unknown. A string match on a null is plain
  true or false (`null_handling`).
- computed features: `+ - *`, unary `-` and `abs` over float64. Division, `//`, `%`, `**`, `min`, `max`,
  comparisons and `and`/`or` are not modelled.
- string matches: exact, starts_with, ends_with and contains, case-sensitive and untrimmed.
- tables: inline rows only; a null input never matches; the first matching row wins.
What is not modelled makes the step unencodable, never a guess: params, `null_handling="error"`,
`trace_output`, regex or case-insensitive matches, rows from params.
A counterexample is an input row for the step on its own; the flow around it may never produce that row.
"""
from __future__ import annotations

import ast
import operator
import re
import struct
import typing as t

import polars as pl
import z3

from decider.steps import ParamRef
from decider.steps.tables import DecisionTableConfig
from decider.steps.tables.schema import (
    AndExpression, BetweenExpression, BoundMode, EqExpression, InExpression, OrExpression, bounds, groups, leaves,
)
from decider.steps.trees import TreeConfig
from decider.steps.trees._encode import feature_kinds, kind_of
from decider.steps.trees.schema import (
    CasesIsIn, CasesRanges, CompositeNode, LeafNode, LogicOp, NullHandling, RangeEndLogic, StringMatchType,
    UnaryBetween, UnaryIsFalse, UnaryIsIn, UnaryIsTrue, UnaryNode, UnaryStringMatch,
)
from decider.steps.tables._encode import dtypes, kind, typed, variables

MODELLED = __doc__.split("\n\n", 1)[1].strip()
TIMEOUT_MS = 10_000
_F64 = z3.Float64()
_ORDER = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq,
          "!=": operator.ne}
_FP_ORDER = {"<": z3.fpLT, "<=": z3.fpLEQ, ">": z3.fpGT, ">=": z3.fpGEQ, "==": z3.fpEQ,
             "!=": lambda a, b: z3.Not(z3.fpEQ(a, b))}
_POLARS = {"float": pl.Float64, "int": pl.Int64, "bool": pl.Boolean, "str": pl.String}
# Each takes (pattern, string): z3's PrefixOf and SuffixOf put the affix first, Contains the string first.
_MATCH = {StringMatchType.exact: lambda p, s: s == p, StringMatchType.starts_with: z3.PrefixOf,
          StringMatchType.ends_with: z3.SuffixOf, StringMatchType.contains: lambda p, s: z3.Contains(s, p)}
Terms = t.Dict[str, t.Tuple[str, t.Any]]


class Unencodable(Exception):
    """A step uses something the encoding doesn't model, so nothing can be said about it."""


class Env:
    """The input variables and output values two versions of a step share."""

    def __init__(self) -> None:
        self.vars: dict[str, tuple[str, t.Any, t.Any]] = {}  # name -> (kind, null flag, value)
        self.facts: list[t.Any] = []
        self.values: dict[tuple, int] = {}

    def read(self, name: str, k: str) -> tuple[t.Any, t.Any]:
        if name not in self.vars:
            if k == "float":
                v = z3.FP(f"f:{name}", _F64)
                null = z3.fpIsNaN(v)
            else:
                v = {"int": z3.Int, "bool": z3.Bool, "str": z3.String}[k](f"{k}:{name}")
                null = z3.Bool(f"null:{name}")
                if k == "int":
                    # The walker's null int is INT64_MIN, so a real int lies strictly inside.
                    self.facts.append(z3.Or(null, z3.And(v > -(2**63), v < 2**63)))
            self.vars[name] = (k, null, v)
        if self.vars[name][0] != k:
            raise Unencodable(f"input {name!r} is read as {self.vars[name][0]} in one version and {k} in the other")
        return self.vars[name][1:]

    def value(self, k: str, v: t.Any) -> t.Any:
        """An output value as a small int, equal exactly when the Python values are."""
        if v is not None:
            v = {"float": lambda x: float(x) + 0.0, "int": int, "bool": bool}.get(k, str)(v)
        return z3.IntVal(self.values.setdefault((k, repr(v)), len(self.values)))


def _const(v: t.Any, k: str) -> t.Any:
    if isinstance(v, ParamRef):
        raise Unencodable("a threshold, value or pattern is a param")
    if k == "float":
        return z3.FPVal(float(v), _F64)
    if isinstance(v, bool) or float(v) != int(v) or abs(int(v)) >= 2**63:
        raise Unencodable(f"threshold {v!r} is not an int64 for an int feature")
    return z3.IntVal(int(v))


def _cmp(op: str, k: str, x: t.Any, c: t.Any) -> t.Any:
    return (_FP_ORDER if k == "float" else _ORDER)[op](x, c)


def _arith(node: ast.AST, env: Env, kinds: dict[str, str]) -> t.Any:
    if isinstance(node, ast.Name):
        if kinds.get(node.id) != "float":
            raise Unencodable(f"computed feature reads {node.id!r}, which is not a float")
        return env.read(node.id, "float")[1]
    if isinstance(node, ast.Constant):
        return z3.FPVal(float(node.value), _F64)
    ops = {ast.Add: z3.fpAdd, ast.Sub: z3.fpSub, ast.Mult: z3.fpMul}
    if isinstance(node, ast.BinOp) and type(node.op) in ops:
        return ops[type(node.op)](z3.RNE(), _arith(node.left, env, kinds), _arith(node.right, env, kinds))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return z3.fpNeg(_arith(node.operand, env, kinds))
    if isinstance(node, ast.Call) and node.func.id == "abs":
        return z3.fpAbs(_arith(node.args[0], env, kinds))
    raise Unencodable(f"the computed feature uses {ast.unparse(node)!r}, which is not modelled")


class _Tests:
    """A tree's conditions as `(holds, fails)` pairs: unknown is neither."""

    def __init__(self, env: Env, kinds: dict[str, str]):
        self.env, self.kinds = env, kinds

    def feature(self, f: t.Any) -> tuple[str, t.Any, t.Any]:
        if isinstance(f.root, str):
            k = self.kinds[f.root]
            return (k, *self.env.read(f.root, k))
        v = _arith(f.root.expr.node, self.env, self.kinds)
        return "float", z3.fpIsNaN(v), v

    @staticmethod
    def known(null: t.Any, ok: t.Any) -> tuple[t.Any, t.Any]:
        return z3.And(z3.Not(null), ok), z3.And(z3.Not(null), z3.Not(ok))

    def compare(self, f: t.Any, op: str, value: t.Any) -> tuple[t.Any, t.Any]:
        k, null, x = self.feature(f)
        return self.known(null, _cmp(op, k, x, _const(value, k)))

    def between(self, f: t.Any, b: t.Any, ops: tuple[str, str]) -> tuple[t.Any, t.Any]:
        k, null, x = self.feature(f)
        return self.known(null, z3.And(*[_cmp(op, k, x, _const(v, k))
                                         for op, v in zip(ops, (b.min, b.max)) if v is not None]))

    def isin(self, f: t.Any, values: t.Any) -> tuple[t.Any, t.Any]:
        if isinstance(values, ParamRef):
            raise Unencodable("isin values are a param")
        k, null, x = self.feature(f)
        return self.known(null, z3.Or(*[_cmp("==", k, x, _const(v, k)) for v in values]))

    def match(self, f: t.Any, patterns: t.Any, node: t.Any) -> tuple[t.Any, t.Any]:
        if node.match_type not in _MATCH or not node.case_sensitive or node.trim_whitespace:
            raise Unencodable(f"string match {node.match_type.value} (case-insensitive, trimmed or regex)")
        handling = getattr(node, "null_handling", None)
        if handling is NullHandling.error:
            raise Unencodable("string match with null_handling error")
        _, null, s = self.feature(f)
        hit = z3.Or(*[_MATCH[node.match_type](_text(p), s) for p in patterns])
        ok = z3.If(null, z3.BoolVal(handling is NullHandling.match), hit)
        return ok, z3.Not(ok)

    def holds(self, c: t.Any) -> tuple[t.Any, t.Any]:
        if isinstance(c, CompositeNode):
            subs = [self.holds(s) for s in c.conditions]
            if c.op is LogicOp.NOT:
                return subs[0][1], subs[0][0]
            ts, fs = [s[0] for s in subs], [s[1] for s in subs]
            return (z3.And(*ts), z3.Or(*fs)) if c.op is LogicOp.AND else (z3.Or(*ts), z3.And(*fs))
        if isinstance(c, UnaryStringMatch):
            return self.match(c.feature, c.patterns, c)
        if isinstance(c, UnaryBetween):
            return self.between(c.feature, c, (">=", "<="))
        if isinstance(c, UnaryIsIn):
            return self.isin(c.feature, c.values)
        if isinstance(c, (UnaryIsTrue, UnaryIsFalse)):
            k, null, x = self.feature(c.feature)
            truthy = x if k == "bool" else _cmp("!=", k, x, _const(0, k))
            return self.known(null, truthy if isinstance(c, UnaryIsTrue) else z3.Not(truthy))
        return self.compare(c.feature, c.op, c.threshold)

    def branches(self, data: t.Any) -> list[t.Any]:
        """The condition for each branch but the last, in the order the walker tries them."""
        if isinstance(data, UnaryNode):
            return [self.holds(data.condition)[0]]
        if isinstance(data, CompositeNode):
            return [self.holds(data)[0]]
        out = []
        for when in data.conditions:
            if isinstance(data, CasesRanges):
                lower = data.end_logic is RangeEndLogic.lower_inclusive
                out.append(self.between(data.feature, when, (">=", "<") if lower else (">", "<="))[0])
            elif isinstance(data, CasesIsIn):
                out.append(self.isin(data.feature, when.values)[0])
            else:
                out.append(self.match(data.feature, when.patterns, data)[0])
        return out


def _text(p: t.Any) -> t.Any:
    if isinstance(p, ParamRef):
        raise Unencodable("a pattern is a param")
    return z3.StringVal(p)


def _select(tree: t.Any, tests: _Tests, root: str, leaf: t.Callable[[str], t.Any], default: t.Any) -> t.Any:
    # The value the walk from `root` ends at: `leaf(id)` for a leaf, `default` where it reaches no leaf.
    memo: dict[str, t.Any] = {}

    def go(nid: t.Optional[str]) -> t.Any:
        if nid is None:
            return default
        if nid not in memo:
            node = tree.nodes[nid]
            if isinstance(node.data, LeafNode):
                memo[nid] = default if node.data.result_idx < 0 else leaf(nid)
            else:
                arms = [go(k) for k in node.children]
                term = arms[-1]
                for cond, arm in reversed(list(zip(tests.branches(node.data), arms))):
                    term = z3.If(cond, arm, term)
                memo[nid] = term
        return memo[nid]

    return go(root)


def _tree_terms(cfg: TreeConfig, env: Env) -> Terms:
    if cfg.null_handling != "otherwise" or cfg.trace_output is not None:
        raise Unencodable("null_handling error or trace_output")
    tree = cfg.tree.to_tree()
    if tree.required_params():
        raise Unencodable("the tree has params")
    kinds = feature_kinds(tree, cfg.feature_types)
    tests = _Tests(env, kinds)
    rows = [*tree.output.data, tree.output.default or {}]
    if any(not -1 <= n.data.result_idx < len(tree.output.data) for n in tree.nodes.values()
           if isinstance(n.data, LeafNode)):
        raise Unencodable("a leaf selects a row the output lacks")
    dt = tree.output.dtypes
    columns = [(c, kind_of(getattr(d, "type", d), c)) for c, d in (dt.items() if isinstance(dt, dict) else dt)]
    slots = ([("", tree.rules)] if tree.mode == "first_match"
             else [(f"{r.name or f'rule_{k}'}.", [r]) for k, r in enumerate(tree.rules)])
    out: Terms = {}
    for prefix, rules in slots:
        every = [(c, k, lambda nid, c=c: rows[tree.nodes[nid].data.result_idx].get(c), rows[-1].get(c))
                 for c, k in columns]
        if cfg.path_output is not None:
            every.append((cfg.path_output, "path", lambda nid: nid, None))
        for name, k, pick, other in every:
            term = env.value(k, other)
            for rule in reversed(rules):
                term = _select(tree, tests, rule.root, lambda nid, pick=pick, k=k: env.value(k, pick(nid)), term)
            out[prefix + name] = (k, term)
    return out


def _table_terms(cfg: DecisionTableConfig, env: Env) -> Terms:
    if getattr(cfg.rows, "data", None) is None:
        raise Unencodable("the rows come from params")
    dt = dtypes(cfg.columns)
    types = {c: kind(d, c) for c, d in dt.items()}
    data = typed(cfg.rows.data, dt)
    kinds = variables(cfg.expression, types)
    ladders = groups(cfg.expression, data)
    bands = {id(leaf): bounds(leaf, data, ladders) for leaf in leaves(cfg.expression)
             if isinstance(leaf, BetweenExpression)}

    def holds(e: t.Any, r: int) -> t.Any:
        if isinstance(e, AndExpression):
            return z3.And(*[holds(s, r) for s in e.expressions]) if e.expressions else z3.BoolVal(True)
        if isinstance(e, OrExpression):
            return z3.Or(*[holds(s, r) for s in e.expressions]) if e.expressions else z3.BoolVal(False)
        k = kinds[e.variable]
        null, x = env.read(e.variable, k)
        if isinstance(e, BetweenExpression):
            lo, hi = bands[id(e)][r]
            low, high = (">=", "<") if e.mode is BoundMode.lower_inclusive else (">", "<=")
            ok = [_cmp(op, k, x, _const(b, k)) for op, b in ((low, lo), (high, hi)) if b is not None]
        elif isinstance(e, EqExpression):
            v = data[r][e.value_column]
            ok = [] if v is None else [x == z3.StringVal(v) if k == "str" else x == z3.BoolVal(v) if k == "bool"
                                       else _cmp("==", k, x, _const(v, k))]
            if v is None:
                return z3.BoolVal(False)
        elif isinstance(e, InExpression):
            items = [v for v in data[r][e.values_column] or () if v is not None]
            ok = [z3.Or(*[x == z3.StringVal(v) if k == "str" else _cmp("==", k, x, _const(v, k)) for v in items])
                  if items else z3.BoolVal(False)]
        else:
            ok = [x]
        return z3.And(z3.Not(null), *ok)

    matched = [holds(cfg.expression, r) for r in range(len(data))]
    out: Terms = {}
    for column, default in zip(cfg.outputs, cfg.default or [None] * len(cfg.outputs)):
        k = types[column]
        term = env.value(k, default)
        for r in reversed(range(len(data))):
            term = z3.If(matched[r], env.value(k, data[r][column]), term)
        out[column] = (k, term)
    return out


def solve(env: Env, *constraints: t.Any) -> t.Optional[z3.ModelRef]:
    """A model satisfying `constraints` over `env`'s inputs, or None when there is none.

    Raises `Unencodable` when z3 can't decide in time.
    """
    s = z3.Solver()
    s.set(timeout=TIMEOUT_MS)
    s.add(*env.facts, *constraints)
    verdict = s.check()
    if verdict == z3.unknown:
        raise Unencodable(f"z3 could not decide within {TIMEOUT_MS // 1000}s")
    return s.model() if verdict == z3.sat else None


def _terms(step: t.Any, env: Env) -> Terms:
    try:
        if isinstance(step, TreeConfig):
            return _tree_terms(step, env)
        if isinstance(step, DecisionTableConfig):
            return _table_terms(step, env)
    except (ValueError, OverflowError) as e:
        raise Unencodable(str(e)) from e
    raise Unencodable(f"a {type(step).__name__} is not a tree or a decision table")


def _row(env: Env, m: z3.ModelRef) -> dict[str, t.Any]:
    row: dict[str, t.Any] = {}
    for name, (k, null, v) in env.vars.items():
        if z3.is_true(m.eval(null, model_completion=True)):
            row[name] = None
        elif k == "float":
            if z3.is_true(m.eval(z3.fpIsInf(v), model_completion=True)):
                row[name] = float("-inf") if z3.is_true(m.eval(z3.fpIsNegative(v), model_completion=True)) else float("inf")
            else:
                bits = m.eval(z3.fpToIEEEBV(v), model_completion=True).as_long()
                row[name] = struct.unpack("<d", struct.pack("<Q", bits))[0]
        elif k == "str":
            text = m.eval(v, model_completion=True).as_string()
            row[name] = re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda g: chr(int(g[1], 16)), text)
        else:
            val = m.eval(v, model_completion=True)
            row[name] = val.as_long() if k == "int" else z3.is_true(val)
    return row


def _reproduces(old: t.Any, new: t.Any, env: Env, row: dict[str, t.Any]) -> bool:
    # Runs both steps on the row, so a counterexample is only claimed when the real steps disagree on it.
    frame = pl.DataFrame({n: [row[n]] for n in row}, schema={n: _POLARS[env.vars[n][0]] for n in row})
    try:
        return not old.run(frame).equals(new.run(frame))
    except Exception:
        return False


def differ(old: t.Any, new: t.Any) -> t.Optional[dict[str, t.Any]]:
    """None when two tree or table steps give the same outputs for every input; else an input row where they differ.

    Raises `Unencodable` when either step uses something the encoding doesn't model.

    Example::

        differ(old_tree, new_tree)   # {"amount": 5.0} where only one of them says "high"
    """
    if type(old) is not type(new):
        raise Unencodable("one step is a tree and the other a table")
    env = Env()
    a, b = _terms(old, env), _terms(new, env)
    if {n: k for n, (k, _) in a.items()} != {n: k for n, (k, _) in b.items()}:
        raise Unencodable("the two versions write different output columns, or of different types")
    model = solve(env, z3.Or(*[a[n][1] != b[n][1] for n in a]))
    if model is None:
        return None
    row = _row(env, model)
    if not _reproduces(old, new, env, row):
        raise Unencodable("z3's counterexample did not reproduce when both steps were run")
    return row
