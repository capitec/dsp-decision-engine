"""A `Tree` as the arrays the compiled walker reads, plus the inputs, params and outputs of its row node."""
from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np

from decider.engine.ir.decls import Input, NullPolicy, Output, ParamDecl
from decider.steps.expr.postfix import to_postfix
from decider.steps.trees.ops import EQ, GE, GT, LE, LT, NE, OPCODE
from decider.steps.trees.schema import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    CompositeNode,
    ComputedFeature,
    Feature,
    LeafNode,
    LogicOp,
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
from decider.steps.trees.walker import CMP_B, CMP_E, CMP_F, CMP_I, CONTAINS, EXACT, LEAF, MATCH, PREFIX, SUFFIX
from decider.steps.values import ParamRef

KINDS = ("float", "int", "bool", "str")
_MODES = {StringMatchType.exact: EXACT, StringMatchType.starts_with: PREFIX, StringMatchType.ends_with: SUFFIX,
          StringMatchType.contains: CONTAINS}
TYPES = {"float": float, "int": int, "bool": bool, "str": str}
ALIASES = {
    **{k: k for k in KINDS},
    **dict.fromkeys(("float64", "float32"), "float"),
    **dict.fromkeys(("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"), "int"),
    "boolean": "bool",
    **dict.fromkeys(("string", "utf8", "categorical", "enum"), "str"),
}


def kind_of(spelling: t.Any, what: str) -> str:
    """`float`, `int`, `bool` or `str` for a Python type or a polars dtype name (`"Int64"`, `"String"`, ...)."""
    name = spelling.__name__ if isinstance(spelling, type) else str(spelling)
    kind = ALIASES.get(name.strip().lower())
    if kind is None:
        raise ValueError(f"{what}: {spelling!r} is not a type a tree understands; "
                         "use float, int, bool or str (or Float64, Int64, Boolean, String)")
    return kind


def _feature_demands(feature: Feature, demand: str, where: str) -> t.Iterator[tuple[str, str, str]]:
    if isinstance(feature.root, str):
        yield feature.root, demand, where
        return
    if demand == "str":
        raise ValueError(f"{where}: string_match on the computed feature {str(feature)!r}; "
                         "a computed feature is a number, so match a string column instead")
    for name in sorted(feature.root.expr.dependencies()):
        yield name, "expr", where


def _condition_demands(c: t.Any, where: str) -> t.Iterator[tuple[str, str, str]]:
    if isinstance(c, CompositeNode):
        for sub in c.conditions:
            yield from _condition_demands(sub, where)
    else:
        demand = ("bool" if isinstance(c, (UnaryIsTrue, UnaryIsFalse))
                  else "str" if isinstance(c, UnaryStringMatch) else "number")
        yield from _feature_demands(c.feature, demand, where)


def _demands(tree: Tree) -> t.Iterator[tuple[str, str, str]]:
    for nid, node in tree.nodes.items():
        data, where = node.data, f"tree node {nid!r}"
        if isinstance(data, UnaryNode):
            yield from _condition_demands(data.condition, where)
        elif isinstance(data, CompositeNode):
            yield from _condition_demands(data, where)
        elif not isinstance(data, LeafNode):
            demand = "str" if isinstance(data, CasesStringMatch) else "number"
            yield from _feature_demands(data.feature, demand, where)


_REFUSED = {
    ("number", "str"): "a numeric comparison on the string feature {f!r}; use string_match",
    ("number", "bool"): "a threshold on the boolean feature {f!r}; use is_true / is_false",
    ("bool", "str"): "is_true / is_false on the string feature {f!r}; use string_match",
    ("str", "float"): "string_match on the {k} feature {f!r}; declare it str in feature_types",
    ("str", "int"): "string_match on the {k} feature {f!r}; declare it str in feature_types",
    ("str", "bool"): "string_match on the {k} feature {f!r}; declare it str in feature_types",
    ("expr", "int"): "the computed feature reads the {k} feature {f!r}; computed features are float arithmetic",
    ("expr", "bool"): "the computed feature reads the {k} feature {f!r}; computed features are float arithmetic",
    ("expr", "str"): "the computed feature reads the {k} feature {f!r}; computed features are float arithmetic",
}


def feature_kinds(tree: Tree, declared: t.Mapping[str, str]) -> dict[str, str]:
    """Each feature the tree reads and its kind: declared, else inferred from how the tree uses it.

    `string_match` makes a feature `str`; only `is_true`/`is_false` makes it
    `bool`; anything else is `float`. Uses that make no sense for the kind
    (`<` on a string, a threshold on a bool) are errors naming the node.
    """
    demands: dict[str, list[tuple[str, str]]] = {}
    for name, demand, where in _demands(tree):
        demands.setdefault(name, []).append((demand, where))
    if unknown := sorted(set(declared) - set(demands)):
        raise ValueError(f"feature_types names {unknown}, which the tree does not read; it reads {sorted(demands)}")
    kinds = {}
    for name, uses in sorted(demands.items()):
        seen = {d for d, _ in uses}
        kind = declared.get(name) or ("str" if "str" in seen else "bool" if seen == {"bool"} else "float")
        for demand, where in uses:
            if (demand, kind) in _REFUSED:
                raise ValueError(f"{where}: " + _REFUSED[demand, kind].format(f=name, k=kind))
        kinds[name] = kind
    return kinds


@dataclass
class Program:
    """What `to_ir` needs from a tree: the row node's inputs, params, outputs and consts.

    `python` is true when the tree matches a string with a regex, without
    case, or trimmed, which compares no bytes; its row node then runs the
    Python walker in every mode and `consts` is empty.
    """

    inputs: tuple[Input, ...]
    params: tuple[ParamDecl, ...]
    outputs: tuple[Output, ...]
    consts: tuple[tuple[str, t.Any], ...]
    python: bool
    kinds: dict[str, str]
    # Per output: (rule slot, column, string choices or None); slot -1 is first-match over every rule.
    columns: tuple[tuple[int, str, t.Optional[tuple[str, ...]]], ...]
    # The arrays `consts` holds the addresses of: whoever holds the row node must keep these alive.
    arrays: list[np.ndarray]


@dataclass
class _Encoder:
    tree: Tree
    kinds: dict[str, str]
    value: t.Callable[..., t.Any]
    rows: list[list[int]] = field(default_factory=list)
    thr: dict[str, list] = field(default_factory=lambda: {"float": [], "int": []})
    params: dict[str, ParamDecl] = field(default_factory=dict)
    exprs: list[tuple[str, list, int]] = field(default_factory=list)
    econst: list[float] = field(default_factory=list)
    patterns: list[bytes] = field(default_factory=list)
    groups: list[int] = field(default_factory=lambda: [0])
    python: bool = False
    default: t.Optional[int] = None

    def slot(self, name: str) -> int:
        kind = self.kinds[name]
        return sorted(f for f, k in self.kinds.items() if k == kind).index(name)

    def add(self, *row: int) -> int:
        self.rows.append([*row, 0, 0, 0, 0, 0][:6])
        return len(self.rows) - 1

    def param(self, ref: ParamRef, kind: str, where: str) -> ParamDecl:
        decl = self.value(ref, TYPES[kind])
        seen = self.params.setdefault(decl.name, decl)
        if seen.annotation is not decl.annotation:
            raise ValueError(f"{where}: param {decl.name!r} is used as {seen.annotation.__name__} and as "
                             f"{kind}; one param has one type, so split it into two")
        return seen

    def threshold(self, value: t.Any, kind: str, where: str) -> int:
        if isinstance(value, ParamRef):
            self.param(value, kind, where)
            same = [n for n, d in self.params.items() if d.annotation is TYPES[kind]]
            return -1 - same.index(value.param)
        if kind == "int":
            if isinstance(value, bool) or float(value) != int(value):
                raise ValueError(f"{where}: threshold {value!r} is not a whole number, but the feature is an int")
            value = int(value)
        consts = self.thr[kind]
        consts.append(float(value) if kind == "float" else value)
        return len(consts) - 1

    def expression(self, computed: ComputedFeature) -> int:
        for k, (source, _, _) in enumerate(self.exprs):
            if source == computed.expression:
                return k
        rows, depth = to_postfix(computed.expr, self.slot, self.econst)
        self.exprs.append((computed.expression, rows, depth))
        return len(self.exprs) - 1

    def compare(self, feature: Feature, op: int, value: t.Any, then: int, other: int, where: str) -> int:
        if isinstance(feature.root, ComputedFeature):
            return self.add(CMP_E, self.expression(feature.root), op, self.threshold(value, "float", where),
                            then, other)
        kind = self.kinds[feature.root]
        return self.add(CMP_I if kind == "int" else CMP_F, self.slot(feature.root), op,
                        self.threshold(value, kind, where), then, other)

    def truth(self, feature: Feature, op: int, then: int, other: int, where: str) -> int:
        if isinstance(feature.root, str) and self.kinds[feature.root] == "bool":
            return self.add(CMP_B, self.slot(feature.root), op, self.threshold(0, "int", where), then, other)
        kind = "float" if isinstance(feature.root, ComputedFeature) else self.kinds[feature.root]
        return self.compare(feature, op, 0 if kind == "int" else 0.0, then, other, where)

    def isin(self, feature: Feature, values: t.Any, then: int, other: int, where: str) -> int:
        if isinstance(values, ParamRef):
            return self.compare(feature, EQ, values, then, other, where)
        for v in reversed(values):
            other = self.compare(feature, EQ, v, then, other, where)
        return other

    def between(self, feature: Feature, bounds: t.Any, ops: tuple[int, int], then: int, other: int,
                where: str) -> int:
        if bounds.max is not None:
            then = self.compare(feature, ops[1], bounds.max, then, other, where)
        if bounds.min is not None:
            then = self.compare(feature, ops[0], bounds.min, then, other, where)
        return then

    def match(self, feature: Feature, patterns: t.Sequence[t.Any], node: t.Any, then: int, other: int,
              where: str) -> int:
        refs = [self.threshold(p, "str", where) for p in patterns if isinstance(p, ParamRef)]
        if node.match_type not in _MODES or not node.case_sensitive or node.trim_whitespace:
            self.python = True
            return then
        literals = [p.encode() for p in patterns if not isinstance(p, ParamRef)]
        if literals:
            self.patterns += literals
            self.groups.append(len(self.patterns))
            refs.insert(0, len(self.groups) - 2)
        for g in reversed(refs):
            other = self.add(MATCH, self.slot(feature.root), _MODES[node.match_type], g, then, other)
        return other

    def condition(self, c: t.Any, then: int, other: int, where: str) -> int:
        if isinstance(c, CompositeNode):
            if c.op is LogicOp.NOT:
                return self.condition(c.conditions[0], other, then, where)
            for sub in reversed(c.conditions):
                if c.op is LogicOp.AND:
                    then = self.condition(sub, then, other, where)
                else:
                    other = self.condition(sub, then, other, where)
            return then if c.op is LogicOp.AND else other
        if isinstance(c, (UnaryIsTrue, UnaryIsFalse)):
            return self.truth(c.feature, NE if isinstance(c, UnaryIsTrue) else EQ, then, other, where)
        if isinstance(c, UnaryBetween):
            return self.between(c.feature, c, (GE, LE), then, other, where)
        if isinstance(c, UnaryIsIn):
            return self.isin(c.feature, c.values, then, other, where)
        if isinstance(c, UnaryStringMatch):
            return self.match(c.feature, c.patterns, c, then, other, where)
        return self.compare(c.feature, OPCODE[c.op], c.threshold, then, other, where)

    def node(self, nid: str, entries: dict[str, int]) -> int:
        node = self.tree.nodes[nid]
        data, where = node.data, f"tree node {nid!r}"
        if isinstance(data, LeafNode):
            if not -1 <= data.result_idx < len(self.tree.output.data):
                raise ValueError(f"{where}: result_idx {data.result_idx} is not a row of the output "
                                 f"({len(self.tree.output.data)} rows, or -1 for the default)")
            return self.add(LEAF, data.result_idx)
        children = [entries[c] if c is not None else self.leaf() for c in node.children]
        if isinstance(data, UnaryNode):
            return self.condition(data.condition, children[0], children[1], where)
        if isinstance(data, CompositeNode):
            return self.condition(data, children[0], children[1], where)
        entry = children[-1]
        for i in reversed(range(len(data.conditions))):
            when = data.conditions[i]
            if isinstance(data, CasesRanges):
                ops = (GE, LT) if data.end_logic is RangeEndLogic.lower_inclusive else (GT, LE)
                entry = self.between(data.feature, when, ops, children[i], entry, where)
            elif isinstance(data, CasesIsIn):
                entry = self.isin(data.feature, when.values, children[i], entry, where)
            else:
                entry = self.match(data.feature, when.patterns, data, children[i], entry, where)
        return entry

    def leaf(self) -> int:
        if self.default is None:
            self.default = self.add(LEAF, -1)
        return self.default

    def roots(self) -> list[int]:
        entries: dict[str, int] = {}
        for nid in _post_order(self.tree):
            entries[nid] = self.node(nid, entries)
        return [entries[r.root] for r in self.tree.rules]


def _post_order(tree: Tree) -> list[str]:
    # Iterative, so a chain of any depth encodes.
    order, seen = [], set()
    stack = [(r.root, False) for r in reversed(tree.rules)]
    while stack:
        nid, done = stack.pop()
        if done:
            order.append(nid)
        elif nid not in seen:
            seen.add(nid)
            stack.append((nid, True))
            stack += [(c, False) for c in tree.nodes[nid].children if c is not None and c not in seen]
    return order


def _output_values(tree: Tree) -> t.Iterator[tuple[str, str, list]]:
    dtypes = tree.output.dtypes
    rows = [*tree.output.data, tree.output.default or {}]
    for column, dtype in (dtypes.items() if isinstance(dtypes, dict) else dtypes):
        kind = kind_of(getattr(dtype, "type", dtype), f"output column {column!r}")
        yield column, kind, [row.get(column) for row in rows]


def _outputs(tree: Tree) -> tuple[list[Output], list[tuple], list[tuple]]:
    # Per output: its declaration, (rule slot, kind, values, validity or None), and what the reference returns.
    outputs, values, columns = [], [], []
    slots = [(-1, "")] if tree.mode == "first_match" else [
        (k, f"{r.name or f'rule_{k}'}.") for k, r in enumerate(tree.rules)]
    for slot, prefix in slots:
        for column, kind, row_values in _output_values(tree):
            name = prefix + column
            if kind == "str":
                texts = [None if v is None else str(v) for v in row_values]
                choices = tuple(dict.fromkeys(v for v in texts if v is not None))
                # ponytail: a column that is always null still needs one choice to be a Literal.
                outputs.append(Output(name, t.Literal[choices or ("",)]))
                values.append((slot, "int", [-1 if v is None else choices.index(v) for v in texts], None))
                columns.append((slot, column, choices))
                continue
            valid = [int(v is not None) for v in row_values]
            outputs.append(Output(name, TYPES[kind] if all(valid) else TYPES[kind] | None))
            values.append((slot, kind, [0 if v is None else TYPES[kind](v) for v in row_values],
                           None if all(valid) else valid))
            columns.append((slot, column, None))
    if not outputs:
        raise ValueError("the tree's output declares no columns (output.dtypes is empty)")
    return outputs, values, columns


def _pack(enc: _Encoder, roots: list[int], values: list[tuple]) -> tuple[tuple, list]:
    # Everything the walker reads, packed into one int64, one float64 and one
    # byte array: it gets their addresses, so no array crosses its per-row call.
    ints: list[int] = []
    floats: list[float] = []

    def put(seq: list, into: list) -> int:
        into.extend(seq)
        return len(into) - len(seq)

    layout = (
        put([x for row in enc.rows for x in row], ints),
        put(list(enc.thr["int"]), ints),
        put(list(roots), ints),
        len(roots),
        put([x for _, rows, _ in enc.exprs for row in rows for x in row], ints),
        put(np.cumsum([0] + [len(rows) for _, rows, _ in enc.exprs]).tolist(), ints),
        put(np.cumsum([0] + [len(p) for p in enc.patterns]).tolist(), ints),
        put(list(enc.groups), ints),
        put(list(enc.thr["float"]), floats),
        put(list(enc.econst), floats),
        max((d for _, _, d in enc.exprs), default=1),
    )
    specs = []
    for slot, kind, vals, valid in values:
        where = floats if kind == "float" else ints
        spec = (slot, put([float(v) if kind == "float" else int(v) for v in vals], where), len(vals),
                {"float": 0.0, "bool": False}.get(kind, 0))
        specs.append(spec if valid is None else (*spec, put(valid, ints)))
    arrays = [np.array(ints or [0], np.int64), np.array(floats or [0.0], np.float64),
              np.frombuffer(b"".join(enc.patterns) or bytes(1), np.uint8).copy()]
    consts = (("ints", arrays[0].ctypes.data), ("floats", arrays[1].ctypes.data), ("chars", arrays[2].ctypes.data),
              ("layout", layout), ("outputs", tuple(specs)))
    return consts, arrays


def encode(tree: Tree, feature_types: t.Mapping[str, str], value: t.Callable[..., t.Any]) -> Program:
    """The row node of `tree`, with `value` turning a `ParamRef` into a `ParamDecl` (`IRContext.value`).

    Example::

        program = encode(config.tree.to_tree(), {"income_cents": "int"}, ctx.value)
    """
    kinds = feature_kinds(tree, {k: kind_of(v, f"feature_types[{k!r}]") for k, v in feature_types.items()})
    enc = _Encoder(tree, kinds, value)
    roots = enc.roots()
    order = {k: i for i, k in enumerate(KINDS)}
    inputs = tuple(
        Input(name, bytes | None, NullPolicy.OPTIONAL) if kind == "str" else Input(name, TYPES[kind])
        for name, kind in sorted(kinds.items(), key=lambda nk: (order[nk[1]], nk[0]))
    )
    params = tuple(sorted(enc.params.values(), key=lambda d: order[d.annotation.__name__]))
    outputs, values, columns = _outputs(tree)
    consts, arrays = ((), []) if enc.python else _pack(enc, roots, values)
    return Program(inputs, params, tuple(outputs), consts, enc.python, kinds, tuple(columns), arrays)
