"""The Python tree walker: the compiled walker's twin, reporting every node it passes."""
from __future__ import annotations

import operator
import re
import typing as t

from decider.steps.trees.schema import (
    CasesIsIn,
    CasesRanges,
    CompositeNode,
    Feature,
    LeafNode,
    LogicOp,
    NullHandling,
    RangeEndLogic,
    StringMatchType,
    Tree,
    UnaryBetween,
    UnaryIsFalse,
    UnaryIsIn,
    UnaryIsTrue,
    UnaryNode,
    UnaryStringMatch,
)
from decider.steps.trees.walker import NULL_INT
from decider.steps.values import ParamRef

_COMPARE = {"<": operator.lt, "<=": operator.le, "==": operator.eq, ">": operator.gt, ">=": operator.ge,
            "!=": operator.ne}


_CAST = {"float": float, "int": int, "bool": bool}


def _text(name: str) -> t.Callable[[t.Any], str]:
    def cast(s: t.Any) -> str:
        if not isinstance(s, str):
            raise TypeError(f"{name!r} is a string input, got {type(s).__name__} {s!r}")
        return s

    return cast
_MATCH = {
    StringMatchType.exact: operator.eq,
    StringMatchType.starts_with: str.startswith,
    StringMatchType.ends_with: str.endswith,
    StringMatchType.contains: operator.contains,
    StringMatchType.regex: lambda s, p: re.search(p, s) is not None,
}


def _value(v: t.Any, cast: t.Callable) -> t.Any:
    # The walker gets a null float as NaN and a null int as NULL_INT, so they are nulls here too.
    if v is None or v != v:
        return None
    v = cast(v)
    return None if cast is int and v == NULL_INT else v


class _Row:
    """One row's feature values and params, read the way the compiled walker reads them.

    A test on a null is unknown (`None`): NOT keeps it unknown, AND and OR
    follow three-valued logic, and a node takes its otherwise branch on it.
    """

    __slots__ = ("values", "params", "kinds")

    def __init__(self, values: dict[str, t.Any], params: t.Any, kinds: dict[str, str]):
        self.values, self.params, self.kinds = values, params, kinds

    def feature(self, f: Feature) -> t.Any:
        if isinstance(f.root, str):
            return self.values[f.root]
        if any(self.values[name] is None for name in f.root.expr.dependencies()):
            return None
        x = f.root.expr.evaluate(self.values)
        # The compiled walker computes a null as NaN, so a NaN result is unknown too.
        return None if x != x else x

    def value(self, v: t.Any, f: Feature) -> t.Any:
        if isinstance(v, ParamRef):
            return getattr(self.params, v.param)
        return int(v) if isinstance(f.root, str) and self.kinds[f.root] == "int" else float(v)

    def between(self, f: Feature, bounds: t.Any, ops: tuple[t.Callable, t.Callable]) -> t.Optional[bool]:
        x = self.feature(f)
        if x is None:
            return None
        return ((bounds.min is None or ops[0](x, self.value(bounds.min, f)))
                and (bounds.max is None or ops[1](x, self.value(bounds.max, f))))

    def isin(self, f: Feature, values: t.Any) -> t.Optional[bool]:
        x = self.feature(f)
        if x is None:
            return None
        if isinstance(values, ParamRef):
            return x == self.value(values, f)
        return any(x == self.value(v, f) for v in values)

    def matches(self, f: Feature, patterns: t.Sequence[t.Any], node: t.Any) -> bool:
        s = self.feature(f)
        if s is None:
            return getattr(node, "null_handling", None) is NullHandling.match
        if node.trim_whitespace:
            s = s.strip()
        found = [getattr(self.params, p.param) if isinstance(p, ParamRef) else p for p in patterns]
        if not node.case_sensitive:
            s, found = s.lower(), [p.lower() for p in found]
        test = _MATCH[node.match_type]
        return any(test(s, p) for p in found)

    def holds(self, c: t.Any) -> t.Optional[bool]:
        if isinstance(c, CompositeNode):
            if c.op is LogicOp.NOT:
                r = self.holds(c.conditions[0])
                return None if r is None else not r
            decisive = c.op is LogicOp.OR
            unknown = False
            for sub in c.conditions:
                r = self.holds(sub)
                if r is decisive:
                    return decisive
                unknown = unknown or r is None
            return None if unknown else not decisive
        if isinstance(c, UnaryStringMatch):
            return self.matches(c.feature, c.patterns, c)
        if isinstance(c, UnaryBetween):
            return self.between(c.feature, c, (operator.ge, operator.le))
        if isinstance(c, UnaryIsIn):
            return self.isin(c.feature, c.values)
        x = self.feature(c.feature)
        if x is None:
            return None
        if isinstance(c, UnaryIsTrue):
            return bool(x)
        if isinstance(c, UnaryIsFalse):
            return not x
        return _COMPARE[c.op](x, self.value(c.threshold, c.feature))

    def branch(self, data: t.Any) -> int:
        if isinstance(data, UnaryNode):
            return 0 if self.holds(data.condition) else 1
        if isinstance(data, CompositeNode):
            return 0 if self.holds(data) else 1
        for i, when in enumerate(data.conditions):
            if isinstance(data, CasesRanges):
                lower = data.end_logic is RangeEndLogic.lower_inclusive
                hit = self.between(data.feature, when, (operator.ge, operator.lt) if lower else (operator.gt, operator.le))
            elif isinstance(data, CasesIsIn):
                hit = self.isin(data.feature, when.values)
            else:
                hit = self.matches(data.feature, when.patterns, data)
            if hit:
                return i
        return len(data.conditions)


def _leaf(tree: Tree, root: str, row: _Row, visit: t.Callable[[str], None]) -> int:
    nid: t.Optional[str] = root
    while nid is not None:
        visit(nid)
        node = tree.nodes[nid]
        if isinstance(node.data, LeafNode):
            return node.data.result_idx
        nid = node.children[row.branch(node.data)]
    return -1


def reference(tree: Tree, inputs: t.Sequence[t.Any], kinds: dict[str, str],
              columns: t.Sequence[tuple[int, str, t.Any]]) -> t.Callable:
    """`reference(row, params, consts, visit)`: the tree walked in Python, calling `visit(node_id)` per node.

    Returns one value per output column, in the order `columns` lists them.

    Example::

        walk = reference(tree, program.inputs, kinds, program.columns)
        walk((0.9,), params, (), print)    # prints the node ids passed, returns the outputs
    """
    names = [(i.name, _text(i.name) if kinds[i.name] == "str" else _CAST[kinds[i.name]]) for i in inputs]
    rows = [*tree.output.data, tree.output.default or {}]

    def walk(row: tuple, params: t.Any, consts: tuple, visit: t.Callable[[str], None]) -> tuple:
        r = _Row({n: _value(v, cast) for (n, cast), v in zip(names, row)}, params, kinds)
        leaves: dict[int, int] = {}
        out = []
        for slot, column, choices in columns:
            if slot not in leaves:
                if slot >= 0:
                    leaves[slot] = _leaf(tree, tree.rules[slot].root, r, visit)
                else:
                    leaves[slot] = next((leaf for rule in tree.rules
                                         if (leaf := _leaf(tree, rule.root, r, visit)) != -1), -1)
            v = rows[leaves[slot]].get(column)
            out.append(str(v) if choices is not None and v is not None else v)
        return tuple(out)

    return walk
