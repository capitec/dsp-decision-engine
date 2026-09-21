"""The tree document — decider 1's v3 vocabulary, kept as-is.

This is the answer to doc 06 §O15 ("the interior document schema") for the
*tree* interior: a tree is a data-shaped interior (doc 08 §3), so it is a
pydantic document, not code, and everything below is data.

**The schema is deliberately decider 1's.** Every field name here
(`node_type`/`type`, `feature`, `thresholds`, `end_logic`, `result_idx`,
`sourceIndex`, `op`, `patterns`, `match_type`, `strict`) is the name decider
1 used, in `decider/modules/rules/tree/v3/` and
`decider/modules/rules/common/`. A v3 tree document that decider 1 accepts
parses here unchanged; `tests/test_trees_migration.py` pins that against
configs lifted verbatim out of decider 1's own test suite.

Four divergences, each deliberate and each narrow:

1. **Union plumbing, not wire format.** decider 1 discriminates `NodeData`
   on `type` and wraps the three `cases` variants in a `CasesNode`
   `RootModel` whose `type` is a *property*, not a field. That works only
   because pydantic never has to look at it. Here the same JSON is
   dispatched by a callable `Discriminator` on `(type, op)`, so the
   discriminator reads a real field in every branch. **The documents are
   byte-identical either way** — this changes how the union is resolved,
   not what it accepts.

2. **`Feature` is a column name, or a computed feature.** decider 1 lets a
   feature be an expression string, evaluated *at runtime* with `simpleeval`
   (`common/feature.py`). An earlier draft of this module removed the
   computed arm outright, citing doc 08 §1's "config may not contain code" —
   doc 08 §1.1/§3.2 and doc 06 §O15 (both revised) now draw the line more
   precisely: that rule bans an unvalidated code *pointer*
   (`{module_name, function_name}`, resolved live), not a closed, statically
   validated expression. `decider2.expr` parses and validates decider 1's
   exact wire format at document-load time and compiles it to numba source
   once, at build time — see `_ComputedFeature` below, and `expr.py`'s own
   module docstring for the full grammar and the distinction from a pointer.
   A derived value still belongs in a step, placed in a module *before* the
   tree (`flow(Derived, my_tree)`), whenever it needs its own null policy,
   its own test, or reuse beyond one rule; a computed feature is for the
   inline-arithmetic case that used to require writing that step anyway.

3. **`InputRef` is a param, not a struct column.** decider 1 resolves an
   `InputRef` as `parameters.struct.field(key)` — a *column*, so it can
   technically vary per row, even though `common/parameters.py` documents
   it as a tuning knob ("externalised from the rule structure so they can
   be changed without rebuilding the rule"). Here it is the knob: an
   `InputRef(key="x")` threshold becomes the kernel param named `x`, shared
   by every node that references it. A genuinely per-row threshold is
   still expressible — as an ordinary input column, compared against with a
   `Feature`-to-`Feature` node — but that is not what an `InputRef` means.

4. **`thresholds` (plural, v2's `RangeNode`) is not a separate node type.**
   decider 1's own `to_v3_node()` already rewrites a v2 `RangeNode` into a
   `CasesRanges` with one `RangeCondition` per threshold; `from_v2_range`
   below is that same rewrite, so v2's `thresholds` list is still a
   supported *input*, just not a distinct node.
"""
from __future__ import annotations

import enum
import typing as t

from pydantic import BaseModel, Discriminator, Field, PrivateAttr, RootModel, Tag, model_validator

from decider2 import expr
from decider2.trees.interpreter import EQ, GE, GT, LE, LT, NE

if t.TYPE_CHECKING:
    # Codegen-only: every node/condition class below calls back into this
    # for the cross-cutting concerns codegen owns (naming and de-duping
    # kernel arguments, hoisting string tests, walking to a node's
    # children). Import guarded so schema.py stays free of a runtime
    # dependency on codegen.py — codegen.py already depends on this
    # module, and a class owning its own emission is not the same claim as
    # this module depending on codegen (see EncodeContext's own docstring).
    from decider2.trees.codegen import EncodeContext

# decider2.trees.schema._ThresholdedUnaryOp's six operators -> the
# interpreter's six comparison opcodes (decider2.trees.interpreter). One
# fixed table, not a per-class dispatch: every subclass below sets `op` to
# one of these six literal strings and this is the only place the string is
# turned into an opcode.
_OPCODE: dict[str, int] = {
    "<=": LE, "<": LT, "==": EQ, ">": GT, ">=": GE, "!=": NE,
}

__all__ = [
    "RangeEndLogic",
    "TStringMatchType",
    "TLogicOp",
    "InputRef",
    "ComputedFeatureRemoved",
    "Feature",
    "Threshold",
    "UnaryLessThanEqual",
    "UnaryLessThan",
    "UnaryEqual",
    "UnaryGreaterThan",
    "UnaryGreaterThanEqual",
    "UnaryNotEqual",
    "UnaryBetween",
    "UnaryIsIn",
    "UnaryStringMatch",
    "UnaryIsTrue",
    "UnaryIsFalse",
    "TUnaryOp",
    "RangeCondition",
    "StringMatchCondition",
    "IsInCondition",
    "CompositeCondition",
    "TCondition",
    "LeafNode",
    "UnaryNode",
    "CasesRanges",
    "CasesStringMatch",
    "CasesIsIn",
    "CompositeNode",
    "NodeData",
    "Position",
    "PositionedNode",
    "MultiEdgeData",
    "MultiSourceEdge",
    "TreeOutput",
    "TreeMetadata",
    "SubTree",
    "Tree",
    "from_v2_range",
]


# ---------------------------------------------------------------------------
# Enums — decider 1's `common/nodetypes.py`, value for value
# ---------------------------------------------------------------------------


class RangeEndLogic(str, enum.Enum):
    """Which end of a range is closed.

    decider 1's `common/nodetypes.RangeEndLogic` and the decision table's
    `BoundMode` are the *same two values* under two names, for the same
    stated reason — the decision table's docstring spells it out:

        lower_inclusive  ->  [lower, upper)   i.e.  lower <= x < upper
        upper_inclusive  ->  (lower, upper]   i.e.  lower <  x <= upper

        ``both_inclusive`` / ``both_exclusive`` are intentionally omitted:
        they create overlaps or gaps at shared boundaries, making contiguous
        range tables ambiguous.

    `decider2.tables.schema.BoundMode` is an alias of this enum, so the two
    halves of the migration cannot drift apart.
    """

    lower_inclusive = "lower_inclusive"
    upper_inclusive = "upper_inclusive"


class TStringMatchType(str, enum.Enum):
    """decider 1's string matching strategies.

    Only `exact` survives into a compiled kernel: doc 05 §1.5 is
    unconditional that "a string never enters a kernel as a string" — it
    enters as an int32 dictionary code, and a code comparison cannot express
    a prefix, a suffix, a substring or a regex. `decider2.trees.codegen`
    raises a named error for the other four rather than silently matching
    something else; see this module's `MIGRATION NOTES` in the package
    docstring.
    """

    exact = "exact"
    starts_with = "starts_with"
    contains = "contains"
    ends_with = "ends_with"
    regex = "regex"


class TLogicOp(str, enum.Enum):
    """Logical operators for composite conditions."""

    AND = "and"
    OR = "or"
    NOT = "not"


# ---------------------------------------------------------------------------
# References and features
# ---------------------------------------------------------------------------


class InputRef(BaseModel):
    """A named tunable — decider 1's `common/shared.InputRef`.

    In decider 1 this resolves to `parameters.struct.field(key)`. Here it
    resolves to the kernel param called `key` (doc 03 §4.4's `param()`),
    which is what decider 1's own `common/parameters.py` documents it as.
    Two nodes referencing the same key share one knob.
    """

    key: str = Field(description="Parameter key from the graph execution context")

    def __str__(self) -> str:
        return f"#{self.key}"


class ComputedFeatureRemoved(Exception):
    """No longer raised. Kept only so an old `except ComputedFeatureRemoved`
    or `pytest.raises(ComputedFeatureRemoved)` still imports.

    Doc 08 §3.2 used to remove decider 1's `_ComputedFeature` outright,
    citing doc 08 §1's "config may not contain code". That conflated a
    `{module_name, function_name}` **pointer** (genuinely banned — no
    declared interface, no schema, doc 01 §5.4) with a **restricted
    expression, validated and statically analysed** — which
    `_ComputedFeature` (`decider/modules/rules/common/feature.py:59`)
    actually was: it already carried `ALLOWED_POLARS_FUNCTIONS` and
    `extract_features_and_parameters`. The mechanical objection was
    narrower — `simpleeval` produced a `polars.Expr` at *runtime*, which
    cannot enter a numba kernel — and that objection is answered by
    compiling to numba source at build time instead (`decider2.expr`), not
    by refusing the feature. See that module's docstring and doc 08 §1.2,
    §3.2 and doc 06 §O15, all updated to say so.

    A computed feature can still fail to parse — `decider2.expr.parse`
    raises `expr.ExprError` (a `ValueError`, so pydantic reports it as an
    ordinary `ValidationError`) for anything outside the closed grammar,
    e.g. decider 1's own `p.bonus` attribute-access convention for
    referencing a parameter, which this module does not carry over
    (attribute access is unconditionally rejected — see `expr.py`). That is
    a grammar violation, not "computed features are removed"; it no longer
    raises this class.
    """


class _ComputedFeature(BaseModel):
    """A validated, statically-analysable expression — decider 1's
    `_ComputedFeature` (`common/feature.py:59`), decider 1's exact wire
    format (`{"type": "computed", "expression": "..."}`), with the
    `simpleeval`-at-runtime step replaced by `decider2.expr`: parsed and
    validated against a closed grammar here, at document-load time, and
    compiled to numba source once, at build time (see that module's
    docstring, and doc 08 §1.2/§3.2).

    There is no `build_expression`/`simple_eval` here and nothing that runs
    per row — `emit()` below is the entire runtime cost, and it runs once.
    """

    type: t.Literal["computed"] = "computed"
    expression: str = Field(
        description="A decider2.expr expression, e.g. 'monthly_income - monthly_expenses'"
    )
    _expr: expr.Expr = PrivateAttr()

    @model_validator(mode="after")
    def _parse(self) -> "_ComputedFeature":
        try:
            self._expr = expr.parse(self.expression)
        except expr.ExprError as e:
            raise ValueError(
                f"computed feature {self.expression!r} is not a valid decider2 "
                f"expression: {e}"
            ) from e
        return self

    def __str__(self) -> str:
        return self.expression

    def required_features(self) -> set[str]:
        return set(self._expr.dependencies())

    def required_params(self) -> set[str]:
        # Every name the expression reads is a dependency wired like any
        # other feature (`required_features` above); every numeric literal
        # it contains becomes its own anonymous, per-use param (see
        # `_ExprEmitAdapter.constant`) rather than a *named*, shared one —
        # the same distinction `_ThresholdedUnaryOp.required_params` draws
        # between a literal `Threshold` (not tracked here) and an
        # `InputRef` (tracked, because its name is the shared knob).
        return set()

    def emit(self, ctx: "EncodeContext", node_id: str) -> str:
        return self._expr.emit(_ExprEmitAdapter(ctx, node_id))


class _ExprEmitAdapter:
    """Bridges `decider2.expr.ExprContext` to one tree's `EncodeContext`, for
    one computed feature's use at one node.

    A computed feature's own free names become ordinary column arguments —
    `ctx.column` is exactly `EncodeContext`'s existing feature bookkeeping, so
    two nodes both reading `income` (one directly, one inside `income - x`)
    share the one signature argument. Its own numeric literals become
    ordinary anonymous params through `EncodeContext.threshold` — the same
    machinery a literal `Threshold` already uses (`_ThresholdedUnaryOp.
    encode`) — so a constant buried inside an expression retunes exactly
    like any other threshold, never recompiling. `_next` numbers them
    uniquely within this one use so `"x * 2 + y * 2"` gets two distinct
    params, not one collided name.

    This is the one place a computed feature stays SOURCE TEXT rather than
    array data (doc 08 §1.2/§3.2): its arithmetic is a value computation, not
    a branch, and `decider2.expr` already compiles it to a numba expression
    once, at build time — `EncodeContext` only has to fold that expression's
    *result* into a local variable and give it a feature slot like any other
    (`EncodeContext.computed_feature_index`).
    """

    def __init__(self, ctx: "EncodeContext", node_id: str) -> None:
        self._ctx = ctx
        self._node_id = node_id
        self._next = 0

    def name(self, ident: str) -> str:
        return self._ctx.column(ident)

    def constant(self, value: "int | float") -> str:
        role = f"expr{self._next}"
        self._next += 1
        return self._ctx.threshold(float(value), node_id=self._node_id, role=role)


class Feature(RootModel[t.Union[_ComputedFeature, str]]):
    """A column name, or a computed feature — decider 1's
    `common/feature.Feature`, in full: `root: Union[_ComputedFeature, str]`,
    same two arms, same wire format either way.

    decider2's divergence from decider 1 is only in how the computed arm is
    realised (`decider2.expr`'s closed grammar and build-time compile, not
    `simpleeval` at runtime) — never in whether it is admitted. See
    `_ComputedFeature` and `expr.py`'s module docstring.
    """

    root: t.Union[_ComputedFeature, str] = Field(description="Feature name to test, or a computed feature")

    def __str__(self) -> str:
        if isinstance(self.root, str):
            return self.root
        return str(self.root)

    def required_features(self) -> set[str]:
        if isinstance(self.root, str):
            return {self.root}
        return self.root.required_features()

    def required_params(self) -> set[str]:
        if isinstance(self.root, str):
            return set()
        return self.root.required_params()

    def feature_index(self, ctx: "EncodeContext", node_id: str) -> int:
        """This feature's slot in the walker's `feats` tuple (doc 08 §3.4):
        a plain column's own registered index, or a computed feature's
        freshly-allocated local slot (its expression is evaluated once, into
        a wrapper-local variable, before the tuple is built — see
        `_ComputedFeature.emit` and `EncodeContext.computed_feature_index`).
        Either way the result substitutes directly wherever a condition
        class currently does
        `feat_idx = self.feature.feature_index(ctx, node_id)`."""
        if isinstance(self.root, str):
            return ctx.plain_feature_index(self.root)
        return ctx.computed_feature_index(self.root, node_id)


Threshold = t.Union[float, int, InputRef]
"""decider 1's `Union[float, InputRef]`, verbatim.

This is the union the whole migration turns on: a literal and a reference
are *both* kernel arguments, so neither is ever baked into emitted source
and retuning either never recompiles (doc 08 §2, EXPERIMENTS.md §O).
"""


# ---------------------------------------------------------------------------
# Unary operators — decider 1's `common/nodes/operators.py`
# ---------------------------------------------------------------------------


class _BaseUnaryOp(BaseModel):
    type: t.Literal["unary"] = "unary"
    feature: Feature = Field(description="Feature name to test")

    def required_features(self) -> set[str]:
        return self.feature.required_features()

    def required_params(self) -> set[str]:
        return self.feature.required_params()

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        """Append this one condition's node(s) to the walker's flat arrays
        and return its entry program-counter — a comparison that, once
        entered, always reaches `then_pc` or `otherwise_pc` (never anything
        else). decider 1's per-op `build_condition`, moved onto the class it
        describes (the owner's own `TypeDiscriminatedBaseModule` pattern:
        `decider/modules/credit/decision_table/config.py`'s
        `Expression.__call__`), re-targeted from source text to array rows —
        `codegen.EncodeContext` is where the cross-cutting bookkeeping
        (argument naming/de-duping, string-test hoisting, node-array
        appends) lives; this method owns only what varies per op.

        `then_pc`/`otherwise_pc` are passed DOWN rather than discovered —
        the enclosing node already built both children's subtrees before
        asking its own condition to encode itself (see `_BinaryNode.encode`),
        so there is nothing to back-patch. `cond_idx` is this condition's
        position among its siblings in an enclosing `CompositeNode`/
        `CompositeCondition`'s `conditions` list — `None` for a lone
        `UnaryNode`, where `node_id` alone already names a unique argument.
        Threaded through so two same-shaped siblings (e.g. `x > 5 and x <
        10`) get distinct PARAM NAMES — for `params_schema()` and
        `POST /params` — instead of silently sharing one. Note this no
        longer has any bearing on *correctness*: two conditions can never
        collide on the same array SLOT regardless of naming, because a
        slot is an integer position, never a generated identifier (see
        `decider2.trees.interpreter`'s module docstring — this is the
        historical defect class this migration exists to retire).
        """
        raise NotImplementedError


class _ThresholdedUnaryOp(_BaseUnaryOp):
    threshold: Threshold = Field(
        description="Comparison value (number or InputRef for a runtime variable)"
    )

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        if isinstance(self.threshold, InputRef):
            params.add(self.threshold.key)
        return params

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        """The six primitive comparisons share this one shape: they differ
        only in `self.op`, a plain comparison operator string mapped once
        (module-level `_OPCODE`) onto the interpreter's six opcodes — so one
        method on the shared base serves all six, exactly one `CMP` node
        each, no dispatch needed."""
        feat_idx = self.feature.feature_index(ctx, node_id)
        suffix = f"_{cond_idx}" if cond_idx is not None else ""
        thr_slot = ctx.threshold_slot(self.threshold, node_id=node_id, role=f"thr{suffix}")
        return ctx.add_cmp(feat_idx, _OPCODE[self.op], thr_slot, then_pc, otherwise_pc)


class UnaryLessThanEqual(_ThresholdedUnaryOp):
    op: t.Literal["<="] = "<="


class UnaryLessThan(_ThresholdedUnaryOp):
    op: t.Literal["<"] = "<"


class UnaryEqual(_ThresholdedUnaryOp):
    op: t.Literal["=="] = "=="


class UnaryGreaterThan(_ThresholdedUnaryOp):
    op: t.Literal[">"] = ">"


class UnaryGreaterThanEqual(_ThresholdedUnaryOp):
    op: t.Literal[">="] = ">="


class UnaryNotEqual(_ThresholdedUnaryOp):
    op: t.Literal["!="] = "!="


class UnaryBetween(_BaseUnaryOp):
    """decider 1's `UnaryBetween`: closed on both ends (`min <= x <= max`).

    Note this is *not* `RangeCondition` — decider 1 gives the two different
    semantics on purpose. `UnaryBetween` is inclusive both ends; a
    `RangeCondition` inside a `CasesRanges` follows the node's `end_logic`.
    Both behaviours are reproduced exactly.
    """

    op: t.Literal["between"] = "between"
    min: t.Optional[Threshold] = None
    max: t.Optional[Threshold] = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "UnaryBetween":
        if self.min is None and self.max is None:
            raise ValueError("At least one of min or max must be specified")
        return self

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        for bound in (self.min, self.max):
            if isinstance(bound, InputRef):
                params.add(bound.key)
        return params

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        """Two chained `CMP` nodes, min tested first — the same evaluation
        order the old `" and ".join([min_expr, max_expr])` gave, built
        inside-out: the max test (evaluated second) is appended first, so
        the min test (evaluated first, the entry point returned) already
        knows where to send a passing row."""
        feat_idx = self.feature.feature_index(ctx, node_id)
        suffix = f"_{cond_idx}" if cond_idx is not None else ""
        entry = then_pc
        if self.max is not None:
            slot = ctx.threshold_slot(self.max, node_id=node_id, role=f"max{suffix}")
            entry = ctx.add_cmp(feat_idx, LE, slot, entry, otherwise_pc)
        if self.min is not None:
            slot = ctx.threshold_slot(self.min, node_id=node_id, role=f"min{suffix}")
            entry = ctx.add_cmp(feat_idx, GE, slot, entry, otherwise_pc)
        return entry


class UnaryIsIn(_BaseUnaryOp):
    """Numeric set membership — decider 1's `UnaryIsIn`.

    Each value is a separate kernel argument (an OR of equalities), so
    retuning any of them is a value change. Changing *how many* values there
    are is a structure change and recompiles, which is correct: it changes
    the emitted source.
    """

    op: t.Literal["isin"] = "isin"
    values: t.Union[t.List[Threshold], InputRef] = Field(
        description="Acceptable values, or an InputRef naming a runtime variable"
    )

    @model_validator(mode="after")
    def _validate_values(self) -> "UnaryIsIn":
        if isinstance(self.values, list) and not self.values:
            raise ValueError("values list must contain at least one element")
        return self

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        if isinstance(self.values, InputRef):
            params.add(self.values.key)
        else:
            for v in self.values:
                if isinstance(v, InputRef):
                    params.add(v.key)
        return params

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        feat_idx = self.feature.feature_index(ctx, node_id)
        idx = cond_idx if cond_idx is not None else 0
        return _encode_isin(ctx, feat_idx, self.values, node_id, idx, then_pc, otherwise_pc)


class UnaryStringMatch(_BaseUnaryOp):
    """decider 1's `UnaryStringMatch`, restricted to `match_type="exact"`.

    `patterns` is an OR. Each pattern becomes its own `str`-typed `param()`,
    resolved to the feature column's int32 dictionary code at param
    resolution (`runtime.invoke._resolve_str_param_code`), exactly as
    EXPERIMENTS.md §O describes — so a pattern is a kernel argument like any
    threshold, and retuning it does not recompile.
    """

    op: t.Literal["string_match"] = "string_match"
    patterns: t.List[t.Union[str, InputRef]] = Field(
        description="Patterns to match (OR logic); static strings and/or InputRefs"
    )
    match_type: TStringMatchType = Field(default=TStringMatchType.exact)
    case_sensitive: bool = Field(default=True)
    trim_whitespace: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_patterns(self) -> "UnaryStringMatch":
        if not self.patterns:
            raise ValueError("patterns list must contain at least one pattern")
        return self

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        for pattern in self.patterns:
            if isinstance(pattern, InputRef):
                params.add(pattern.key)
        return params

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        # cond_idx is unused: a string test names its literals by the
        # hoisted matcher step, not by node_id/role, so it never collides
        # with a sibling the way a threshold param would.
        return ctx.encode_string_match(
            self.feature, self.patterns, self.match_type, self.case_sensitive,
            self.trim_whitespace, node_id, then_pc, otherwise_pc,
        )


class UnaryIsTrue(_BaseUnaryOp):
    op: t.Literal["is_true"] = "is_true"

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        feat_idx = self.feature.feature_index(ctx, node_id)
        return ctx.add_is_true(feat_idx, then_pc, otherwise_pc)


class UnaryIsFalse(_BaseUnaryOp):
    op: t.Literal["is_false"] = "is_false"

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        feat_idx = self.feature.feature_index(ctx, node_id)
        return ctx.add_is_false(feat_idx, then_pc, otherwise_pc)


TUnaryOp = t.Annotated[
    t.Union[
        UnaryLessThanEqual,
        UnaryLessThan,
        UnaryEqual,
        UnaryGreaterThan,
        UnaryGreaterThanEqual,
        UnaryNotEqual,
        UnaryBetween,
        UnaryIsIn,
        UnaryStringMatch,
        UnaryIsTrue,
        UnaryIsFalse,
    ],
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------------
# Conditions — decider 1's `common/nodes/conditions.py`
# ---------------------------------------------------------------------------


class RangeCondition(BaseModel):
    """One band of a `CasesRanges`. Closed end decided by the node's
    `end_logic` (decider 1's `RangeCondition.build_range_condition`)."""

    min: t.Optional[Threshold] = None
    max: t.Optional[Threshold] = None

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RangeCondition":
        if self.min is None and self.max is None:
            raise ValueError("At least one of min or max must be specified")
        return self

    def required_params(self) -> set[str]:
        return {b.key for b in (self.min, self.max) if isinstance(b, InputRef)}

    def encode(
        self, ctx: "EncodeContext", feat_idx: int, end_logic: RangeEndLogic, node_id: str,
        idx: int, *, then_pc: int, otherwise_pc: int,
    ) -> int:
        """decider 1's `RangeCondition.build_range_condition`, as two chained
        `CMP` nodes instead of source text.

        `feat_idx`, `end_logic` and `idx` come from the enclosing
        `CasesRanges` node — this condition owns the bound comparison, not
        the branching the node builds around it (decider 1's own split
        between `RangeCondition` and the node that holds a list of them)."""
        lo_op, hi_op = (GE, LT) if end_logic is RangeEndLogic.lower_inclusive else (GT, LE)
        entry = then_pc
        if self.max is not None:
            slot = ctx.threshold_slot(self.max, node_id=node_id, role=f"max_{idx}")
            entry = ctx.add_cmp(feat_idx, hi_op, slot, entry, otherwise_pc)
        if self.min is not None:
            slot = ctx.threshold_slot(self.min, node_id=node_id, role=f"min_{idx}")
            entry = ctx.add_cmp(feat_idx, lo_op, slot, entry, otherwise_pc)
        return entry


class StringMatchCondition(BaseModel):
    """One branch of a `CasesStringMatch`."""

    patterns: t.List[t.Union[str, InputRef]] = Field(
        description="Patterns to match (OR logic)"
    )

    @model_validator(mode="after")
    def _validate_patterns(self) -> "StringMatchCondition":
        if not self.patterns:
            raise ValueError("patterns list must contain at least one pattern")
        return self

    def required_params(self) -> set[str]:
        return {p.key for p in self.patterns if isinstance(p, InputRef)}

    def encode(
        self,
        ctx: "EncodeContext",
        node_id: str,
        *,
        feature: Feature,
        match_type: TStringMatchType,
        case_sensitive: bool,
        trim_whitespace: bool,
        then_pc: int,
        otherwise_pc: int,
    ) -> int:
        """`feature`/`match_type`/`case_sensitive`/`trim_whitespace` are the
        enclosing `CasesStringMatch` node's — a branch only ever carries its
        own `patterns` (the wire format's shape, kept as-is)."""
        return ctx.encode_string_match(
            feature, self.patterns, match_type, case_sensitive, trim_whitespace, node_id,
            then_pc, otherwise_pc,
        )


class IsInCondition(BaseModel):
    """One branch of a `CasesIsIn`."""

    values: t.Union[t.List[Threshold], InputRef]

    def required_params(self) -> set[str]:
        if isinstance(self.values, InputRef):
            return {self.values.key}
        return {v.key for v in self.values if isinstance(v, InputRef)}

    def encode(
        self, ctx: "EncodeContext", feat_idx: int, node_id: str, idx: int, *,
        then_pc: int, otherwise_pc: int,
    ) -> int:
        return _encode_isin(ctx, feat_idx, self.values, node_id, idx, then_pc, otherwise_pc)


def _encode_isin(
    ctx: "EncodeContext", feat_idx: int, values: t.Any, node_id: str, idx: t.Any,
    then_pc: int, otherwise_pc: int,
) -> int:
    """An OR-of-equalities chain — each value its own `CMP` node, tested in
    declared order, first match wins. Shared by `UnaryIsIn.encode` and
    `IsInCondition.encode` — the same `isinstance` check the OLD `isin_test`
    made, kept for the same reason it stayed there: `values` is a
    list-of-thresholds-or-a-reference, not a node/condition type this tree
    dispatches on.

    decider 1 also allows `values` to be a bare `InputRef`, treated as
    *equality against that one parameter* (`UnaryIsIn.build_condition`), not
    `is_in` — that exact behaviour is reproduced: one `CMP(EQ)` node, no
    chain.

    Built backward (highest index first) so the LAST-built node becomes the
    FIRST tested: `entry` starts as `otherwise_pc` ("nothing left, fail"),
    and each earlier value's node routes its own false-edge to whatever was
    built so far — exactly the `" or ".join(...)` short-circuit order the
    old source text gave, without ever writing a name.
    """
    if isinstance(values, InputRef):
        slot = ctx.threshold_slot(values, node_id=node_id, role=f"isin_{idx}")
        return ctx.add_cmp(feat_idx, EQ, slot, then_pc, otherwise_pc)
    entry = otherwise_pc
    for j in reversed(range(len(values))):
        slot = ctx.threshold_slot(values[j], node_id=node_id, role=f"isin_{idx}_{j}")
        entry = ctx.add_cmp(feat_idx, EQ, slot, then_pc, entry)
    return entry


def _condition_tag(value: t.Any) -> str:
    """decider 1's `_condition_discriminator`, same default."""
    if isinstance(value, dict):
        return value.get("type", "unary")
    return getattr(value, "type", "unary")


TCondition = t.Annotated[
    t.Union[
        t.Annotated[TUnaryOp, Tag("unary")],
        t.Annotated["CompositeCondition", Tag("composite")],
    ],
    Discriminator(_condition_tag),
]


class CompositeCondition(BaseModel):
    """Nested AND/OR/NOT — decider 1's `CompositeCondition`."""

    type: t.Literal["composite"] = "composite"
    id: t.Optional[str] = None
    op: TLogicOp
    conditions: t.List[TCondition] = Field(description="Conditions to combine")

    @model_validator(mode="after")
    def _validate_conditions(self) -> "CompositeCondition":
        if self.op == TLogicOp.NOT and len(self.conditions) != 1:
            raise ValueError("NOT operator must have exactly 1 condition")
        if not self.conditions:
            raise ValueError("composite condition needs at least one condition")
        return self

    def required_features(self) -> set[str]:
        out: set[str] = set()
        for cond in self.conditions:
            out |= cond.required_features()
        return out

    def required_params(self) -> set[str]:
        out: set[str] = set()
        for cond in self.conditions:
            out |= cond.required_params()
        return out

    def encode(
        self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int,
        cond_idx: t.Optional[str] = None,
    ) -> int:
        """A nested AND/OR/NOT, encoded as a chain of its own sub-conditions
        — no wrapping needed the way old source text needed its own parens,
        because a chain composes into `then_pc`/`otherwise_pc` exactly like
        any primitive comparison (unlike a top-level `CompositeNode`'s
        encoding, which already targets the enclosing NODE's two children
        rather than a further-composing pair — see `CompositeNode.
        _encode_test`; the two call `_encode_logic` with the same op-chain
        logic, but are not the same method, for that reason).

        `cond_idx` is this condition's position among its own siblings; a
        nested child's path extends it (`f'{prefix}_{j}'`) rather than
        restarting it, so a composite three levels deep still names every
        leaf threshold's PARAM uniquely (see `_BaseUnaryOp.encode`'s
        docstring for why this no longer has any bearing on correctness —
        only on how a threshold shows up in `params_schema()`).
        `TCondition`'s two members — this class and every `TUnaryOp` — share
        this exact signature, so a caller holding a `TCondition` just calls
        `.encode(...)`: no branch on which one it got needed."""
        prefix = cond_idx if cond_idx is not None else "0"
        return _encode_logic(
            ctx, node_id, self.op, self.conditions, then_pc=then_pc, otherwise_pc=otherwise_pc,
            prefix=prefix,
        )


CompositeCondition.model_rebuild()


def _encode_logic(
    ctx: "EncodeContext", node_id: str, op: TLogicOp, conditions: t.Sequence["TCondition"],
    *, then_pc: int, otherwise_pc: int, prefix: str,
) -> int:
    """AND/OR/NOT over `conditions`, as a chain of their own `.encode(...)`
    calls — shared by `CompositeCondition.encode` (a nested composite,
    composing further) and `CompositeNode._encode_test` (a top-level
    composite, the whole node). Both give this the same two-edge contract
    every primitive condition gives: entered once, it always reaches
    `then_pc` or `otherwise_pc`.

    AND is built backward from `then_pc` — each earlier condition's
    then-edge is "go test the rest", so the first-built (last-processed, at
    the END of the loop) becomes the entry, evaluated first, short-circuiting
    to `otherwise_pc` on its own false. OR is the mirror, built backward from
    `otherwise_pc`. NOT (exactly one condition) is neither: it just swaps
    which edge is which for its single child.
    """
    if op is TLogicOp.NOT:
        return conditions[0].encode(
            ctx, node_id, then_pc=otherwise_pc, otherwise_pc=then_pc, cond_idx=f"{prefix}_0"
        )
    if op is TLogicOp.AND:
        entry = then_pc
        for j in reversed(range(len(conditions))):
            entry = conditions[j].encode(
                ctx, node_id, then_pc=entry, otherwise_pc=otherwise_pc, cond_idx=f"{prefix}_{j}"
            )
        return entry
    # OR
    entry = otherwise_pc
    for j in reversed(range(len(conditions))):
        entry = conditions[j].encode(
            ctx, node_id, then_pc=then_pc, otherwise_pc=entry, cond_idx=f"{prefix}_{j}"
        )
    return entry


# ---------------------------------------------------------------------------
# Nodes — decider 1's `tree/v3/nodes_ui.py`
#
# Each class below owns its own `encode(ctx, node_id, depth) -> int`: it
# appends its own row(s) to the walker's flat arrays and returns its entry
# program-counter, recursing into its children through `ctx.child_entry`
# (which calls back into whichever node type it finds there — no isinstance
# needed, the discriminated union above already resolved it). `codegen.py`
# no longer imports any of these classes; it holds only `EncodeContext` and
# the module-level scaffolding around `tree.root's .encode(ctx, ...)`.
#
# `_BinaryNode` and `_CasesNode` are the two shapes decider 1 also had two
# of (a single test routing then/otherwise; a list of conditions routing
# one-of-N-plus-otherwise) — factored once so `UnaryNode`/`CompositeNode`
# and the three `Cases*` classes each implement only what varies for them.
# ---------------------------------------------------------------------------


class LeafNode(BaseModel):
    """Terminal node. `result_idx` indexes `TreeOutput.data`; -1 is the
    default row (decider 1's `LeafNodeCore`, same sentinel)."""

    type: t.Literal["leaf"] = "leaf"
    id: t.Optional[str] = None
    result_idx: int = Field(
        default=-1, description="Index into output table. -1 indicates default/no-match."
    )

    def required_features(self) -> set[str]:
        return set()

    def required_params(self) -> set[str]:
        return set()

    def encode(self, ctx: "EncodeContext", node_id: str, depth: int) -> int:
        return ctx.add_leaf(self.result_idx)


class _BinaryNode(BaseModel):
    """Shared shape for `UnaryNode` and `CompositeNode`: one boolean test,
    `sourceIndex=0` is `then`, `1` is `otherwise`."""

    def _encode_test(self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int) -> int:
        raise NotImplementedError

    def encode(self, ctx: "EncodeContext", node_id: str, depth: int) -> int:
        then_pc = ctx.child_entry(node_id, 0, depth + 1)
        otherwise_pc = ctx.child_entry(node_id, 1, depth)
        return self._encode_test(ctx, node_id, then_pc=then_pc, otherwise_pc=otherwise_pc)


class UnaryNode(_BinaryNode):
    """Single condition. Edge `sourceIndex=0` is `then`, `1` is `otherwise`."""

    type: t.Literal["unary"] = "unary"
    id: t.Optional[str] = None
    condition: TUnaryOp = Field(description="The condition to evaluate")

    @property
    def arity(self) -> int:
        return 2

    def required_features(self) -> set[str]:
        return self.condition.required_features()

    def required_params(self) -> set[str]:
        return self.condition.required_params()

    def _encode_test(self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int) -> int:
        return self.condition.encode(ctx, node_id, then_pc=then_pc, otherwise_pc=otherwise_pc)


class _CasesNode(BaseModel):
    """Shared shape for the three `Cases*` node types: `sourceIndex=0..N-1`
    select `conditions[i]`, `sourceIndex=N` is `otherwise`. They differ
    only in what one condition encodes against `self.feature` — that part is
    `_encode_condition`, implemented per class."""

    def _encode_condition(
        self, ctx: "EncodeContext", feat_idx: int, node_id: str, idx: int, *,
        then_pc: int, otherwise_pc: int,
    ) -> int:
        raise NotImplementedError

    def encode(self, ctx: "EncodeContext", node_id: str, depth: int) -> int:
        feat_idx = self.feature.feature_index(ctx, node_id)  # type: ignore[attr-defined]
        n = len(self.conditions)  # type: ignore[attr-defined]
        # Build backward exactly like `_encode_logic`'s OR case: the LAST
        # arm's failure falls through to the shared "otherwise" child, and
        # each earlier arm's failure falls through to whatever was built so
        # far — so arm 0 (`ctx.child_entry(node_id, 0, ...)`) is the entry,
        # tested first, first match wins, matching decider 1's own
        # `if/elif/.../else` semantics exactly.
        otherwise_pc = ctx.child_entry(node_id, n, depth)
        entry = otherwise_pc
        for i in reversed(range(n)):
            then_pc = ctx.child_entry(node_id, i, depth + 1)
            entry = self._encode_condition(
                ctx, feat_idx, node_id, i, then_pc=then_pc, otherwise_pc=entry
            )
        return entry


class CasesRanges(_CasesNode):
    """Multi-way range branching. `sourceIndex=0..N-1` select
    `conditions[i]`; `sourceIndex=N` is `otherwise`."""

    type: t.Literal["cases"] = "cases"
    op: t.Literal["ranges"] = "ranges"
    id: t.Optional[str] = None
    feature: Feature
    conditions: t.List[RangeCondition] = Field(
        description="Range conditions, in source-index order"
    )
    end_logic: RangeEndLogic = Field(default=RangeEndLogic.lower_inclusive)
    strict: bool = Field(default=True)

    @model_validator(mode="after")
    def _validate(self) -> "CasesRanges":
        validate_range_conditions(self.conditions, self.strict)
        return self

    @property
    def arity(self) -> int:
        return len(self.conditions) + 1

    def required_features(self) -> set[str]:
        return self.feature.required_features()

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        for cond in self.conditions:
            params |= cond.required_params()
        return params

    def _encode_condition(
        self, ctx: "EncodeContext", feat_idx: int, node_id: str, idx: int, *,
        then_pc: int, otherwise_pc: int,
    ) -> int:
        return self.conditions[idx].encode(
            ctx, feat_idx, self.end_logic, node_id, idx, then_pc=then_pc, otherwise_pc=otherwise_pc
        )


class CasesStringMatch(_CasesNode):
    """Multi-way string matching."""

    type: t.Literal["cases"] = "cases"
    op: t.Literal["string_match"] = "string_match"
    id: t.Optional[str] = None
    feature: Feature
    conditions: t.List[StringMatchCondition] = Field(
        description="Pattern conditions, in source-index order"
    )
    match_type: TStringMatchType = Field(default=TStringMatchType.exact)
    case_sensitive: bool = True
    trim_whitespace: bool = False

    @property
    def arity(self) -> int:
        return len(self.conditions) + 1

    def required_features(self) -> set[str]:
        return self.feature.required_features()

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        for cond in self.conditions:
            params |= cond.required_params()
        return params

    def _encode_condition(
        self, ctx: "EncodeContext", feat_idx: int, node_id: str, idx: int, *,
        then_pc: int, otherwise_pc: int,
    ) -> int:
        # feat_idx is unused: a string test re-derives its own feature slot
        # from `feature=` (idempotent — `plain_feature_index` dedupes by
        # name), because it must also register the hoisted matcher, which a
        # plain numeric `feat_idx` alone does not carry.
        del feat_idx
        return self.conditions[idx].encode(
            ctx, node_id,
            feature=self.feature, match_type=self.match_type,
            case_sensitive=self.case_sensitive, trim_whitespace=self.trim_whitespace,
            then_pc=then_pc, otherwise_pc=otherwise_pc,
        )


class CasesIsIn(_CasesNode):
    """Multi-way categorical branching over numeric value sets."""

    type: t.Literal["cases"] = "cases"
    op: t.Literal["isin"] = "isin"
    id: t.Optional[str] = None
    feature: Feature
    conditions: t.List[IsInCondition] = Field(
        description="Value sets, in source-index order"
    )

    @property
    def arity(self) -> int:
        return len(self.conditions) + 1

    def required_features(self) -> set[str]:
        return self.feature.required_features()

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        for cond in self.conditions:
            params |= cond.required_params()
        return params

    def _encode_condition(
        self, ctx: "EncodeContext", feat_idx: int, node_id: str, idx: int, *,
        then_pc: int, otherwise_pc: int,
    ) -> int:
        return self.conditions[idx].encode(
            ctx, feat_idx, node_id, idx, then_pc=then_pc, otherwise_pc=otherwise_pc
        )


class CompositeNode(_BinaryNode):
    """AND/OR/NOT over conditions. `sourceIndex=0` is `then`, `1` is
    `otherwise` (decider 1's `BaseCompositeNode`)."""

    type: t.Literal["composite"] = "composite"
    id: t.Optional[str] = None
    op: TLogicOp
    conditions: t.List[TCondition] = Field(description="Conditions to combine")

    @model_validator(mode="after")
    def _validate_conditions(self) -> "CompositeNode":
        if self.op == TLogicOp.NOT and len(self.conditions) != 1:
            raise ValueError("NOT operator must have exactly 1 condition")
        if not self.conditions:
            raise ValueError("composite node needs at least one condition")
        return self

    @property
    def arity(self) -> int:
        return 2

    def required_features(self) -> set[str]:
        out: set[str] = set()
        for cond in self.conditions:
            out |= cond.required_features()
        return out

    def required_params(self) -> set[str]:
        out: set[str] = set()
        for cond in self.conditions:
            out |= cond.required_params()
        return out

    def _encode_test(self, ctx: "EncodeContext", node_id: str, *, then_pc: int, otherwise_pc: int) -> int:
        """This targets the enclosing NODE's own two children directly
        (unlike `CompositeCondition.encode`, which must compose into a
        further-enclosing pair) — see that method's docstring for why the
        two shapes differ, even though both call the same `_encode_logic`."""
        return _encode_logic(
            ctx, node_id, self.op, self.conditions, then_pc=then_pc, otherwise_pc=otherwise_pc,
            prefix="0",
        )


def validate_range_conditions(
    conditions: t.Sequence[RangeCondition], strict: bool
) -> None:
    """decider 1's `common/nodes/cases.validate_range_conditions`, same rules.

    Only statically-known bounds are checked: a bound that is an `InputRef`
    is not knowable at load time, and decider 1 skips it for the same
    reason. Sorted order and (in strict mode) contiguity.
    """
    if not conditions or not strict:
        return

    all_min_none = all(
        rc.min is None or isinstance(rc.min, InputRef) for rc in conditions
    )
    if all_min_none:
        static_max = [
            (i, rc.max)
            for i, rc in enumerate(conditions)
            if rc.max is not None and not isinstance(rc.max, InputRef)
        ]
        for j in range(len(static_max) - 1):
            if static_max[j][1] >= static_max[j + 1][1]:
                raise ValueError(
                    f"Ranges must be in sorted order. Range {static_max[j][0]} has "
                    f"max={static_max[j][1]}, but range {static_max[j + 1][0]} has "
                    f"max={static_max[j + 1][1]}."
                )
        return

    static = [
        (i, rc.min, rc.max)
        for i, rc in enumerate(conditions)
        if not isinstance(rc.min, InputRef) and not isinstance(rc.max, InputRef)
    ]
    if len(static) < 2:
        return

    for j in range(len(static) - 1):
        cur_idx, cur_min, cur_max = static[j]
        nxt_idx, nxt_min, _ = static[j + 1]
        cur_min_val = cur_min if cur_min is not None else float("-inf")
        nxt_min_val = nxt_min if nxt_min is not None else float("-inf")
        if cur_min_val >= nxt_min_val and cur_min is not None and nxt_min is not None:
            raise ValueError(
                f"Ranges must be in sorted order. Range {cur_idx} has min={cur_min}, "
                f"but range {nxt_idx} has min={nxt_min}."
            )
        if cur_max is not None and nxt_min is not None and cur_max != nxt_min:
            raise ValueError(
                f"Ranges are not continuous in strict mode. Range {cur_idx} ends at "
                f"{cur_max} but range {nxt_idx} starts at {nxt_min}."
            )


def _node_tag(value: t.Any) -> str:
    """Dispatch a node document on `(type, op)`.

    decider 1 dispatches on `type` alone and wraps the `cases` variants in a
    `RootModel` whose `type` is a property. Same accepted documents; this
    just resolves them with a discriminator that reads real fields.
    """
    get = value.get if isinstance(value, dict) else lambda k, d=None: getattr(value, k, d)
    node_type = get("type", None)
    if node_type is None:
        node_type = get("node_type", None)  # decider 1 v1/v2 spelling
    if node_type == "cases":
        return f"cases:{get('op', 'ranges')}"
    return str(node_type)


NodeData = t.Annotated[
    t.Union[
        t.Annotated[LeafNode, Tag("leaf")],
        t.Annotated[UnaryNode, Tag("unary")],
        t.Annotated[CasesRanges, Tag("cases:ranges")],
        t.Annotated[CasesStringMatch, Tag("cases:string_match")],
        t.Annotated[CasesIsIn, Tag("cases:isin")],
        t.Annotated[CompositeNode, Tag("composite")],
    ],
    Discriminator(_node_tag),
]


# ---------------------------------------------------------------------------
# Graph structure — decider 1's `tree/v1/edges.py` and `tree/v3/tree.py`
# ---------------------------------------------------------------------------


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class PositionedNode(BaseModel):
    id: str
    position: Position = Field(default_factory=Position)
    data: NodeData


class MultiEdgeData(BaseModel):
    """decider 1's `MultiEdgeData` — note the camelCase `sourceIndex`, which
    is the UI's spelling and is kept so UI documents round-trip."""

    sourceIndex: t.List[int]

    @model_validator(mode="before")
    @classmethod
    def _ensure_list(cls, value: t.Any) -> t.Any:
        if isinstance(value, dict) and "sourceIndex" in value:
            si = value["sourceIndex"]
            if not isinstance(si, (list, tuple)):
                return {**value, "sourceIndex": [si]}
        return value


class MultiSourceEdge(BaseModel):
    """decider 1's `GenericEdge[MultiEdgeData]`."""

    id: t.Optional[str] = None
    source: str
    target: str
    data: MultiEdgeData


class TreeOutput(BaseModel):
    """decider 1's `common/shared.TreeOutput` — the leaf value table.

    `data[result_idx]` is the row a leaf selects; `default` is what
    `result_idx == -1` selects. `dtypes` names each output column and its
    type, using decider 1's polars type spellings ("Float64", "Int64",
    "Boolean", "String").
    """

    data: t.List[t.Dict[str, t.Any]] = Field(default_factory=list)
    default: t.Optional[t.Dict[str, t.Any]] = None
    dtypes: t.List[t.Tuple[str, str]] = Field(default_factory=list)
    type_defs: t.Dict[str, t.Any] = Field(default_factory=dict)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.dtypes)

    @model_validator(mode="after")
    def _validate_rows(self) -> "TreeOutput":
        declared = set(self.columns)
        if not declared:
            return self
        for i, row in enumerate(self.data):
            missing = declared - set(row)
            if missing:
                raise ValueError(
                    f"TreeOutput.data[{i}] is missing declared column(s) "
                    f"{sorted(missing)} (declared by dtypes)."
                )
        if self.default is not None:
            missing = declared - set(self.default)
            if missing:
                raise ValueError(
                    f"TreeOutput.default is missing declared column(s) {sorted(missing)}."
                )
        return self


class TreeMetadata(BaseModel):
    name: t.Optional[str] = None
    description: t.Optional[str] = None


class SubTree(BaseModel):
    id: t.Optional[str] = None
    name: t.Optional[str] = None


class Tree(BaseModel):
    """A v3 tree document.

    Field for field this is decider 1's `tree/v3/tree.Tree`, minus the four
    fields that were decider-1 execution plumbing rather than tree
    structure: `input_schema` (a `PolarsSchema` for casting — decider2's
    boundary layer owns casting, doc 05 §1), `parameters`/`parameters_col`
    (decider 1's `WithParameters` struct-column mechanism, replaced by
    `param()`), and the `BaseExecuteModule` base class itself.
    """

    type: t.Literal["v3-tree"] = "v3-tree"
    name: str = "output"
    metadata: t.Optional[TreeMetadata] = None
    nodes: t.List[PositionedNode]
    edges: t.List[MultiSourceEdge] = Field(default_factory=list)
    subtrees: t.List[SubTree] = Field(default_factory=list)
    output: TreeOutput = Field(default_factory=TreeOutput)
    format_version: t.Literal[3] = Field(alias="formatVersion", default=3)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _validate_structure(self) -> "Tree":
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"tree has duplicate node id(s): {dupes}")
        if not self.nodes:
            raise ValueError("tree has no nodes")
        known = set(ids)
        for e in self.edges:
            if e.source not in known:
                raise ValueError(f"edge {e.id or ''} has unknown source '{e.source}'")
            if e.target not in known:
                raise ValueError(f"edge {e.id or ''} has unknown target '{e.target}'")
        return self

    # -- structure helpers the codegen needs -------------------------------

    def children(self) -> dict[str, dict[int, str]]:
        """`{node_id: {source_index: target_node_id}}` — decider 1's
        adjacency build in `v3/tree.to_flat_rule_tree`, verbatim."""
        out: dict[str, dict[int, str]] = {}
        for edge in self.edges:
            for si in edge.data.sourceIndex:
                out.setdefault(edge.source, {})[si] = edge.target
        return out

    def root_id(self) -> str:
        """decider 1's root selection: the node with no incoming edge,
        preferring the first declared subtree when there are several."""
        targets = {e.target for e in self.edges}
        roots = [n.id for n in self.nodes if n.id not in targets]
        if not roots:
            raise ValueError("Tree has no root nodes (circular structure)")
        if len(roots) == 1:
            return roots[0]
        subtree_order = {st.id: i for i, st in enumerate(self.subtrees) if st.id}
        node_order = {n.id: i for i, n in enumerate(self.nodes)}
        return sorted(
            roots,
            key=lambda r: (
                subtree_order.get(r, float("inf")),
                node_order.get(r, float("inf")),
            ),
        )[0]

    def node_map(self) -> dict[str, PositionedNode]:
        return {n.id: n for n in self.nodes}

    def required_features(self) -> set[str]:
        out: set[str] = set()
        for n in self.nodes:
            out |= n.data.required_features()
        return out

    def required_params(self) -> set[str]:
        out: set[str] = set()
        for n in self.nodes:
            out |= n.data.required_params()
        return out


# ---------------------------------------------------------------------------
# v2 compatibility
# ---------------------------------------------------------------------------


def from_v2_range(
    feature: str,
    thresholds: t.Sequence[Threshold],
    end_logic: RangeEndLogic = RangeEndLogic.lower_inclusive,
    *,
    id: t.Optional[str] = None,
) -> CasesRanges:
    """v2's `RangeNode(feature, thresholds, end_logic)` as a `CasesRanges`.

    This is decider 1's own `v2/nodes.RangeNode.to_v3_node()`, reproduced
    exactly — "Range i: [thrs[i-1], thrs[i])" — so a v2 `thresholds` list
    (the `Union[float, InputRef]` list the migration turns on) is still a
    supported input shape even though v3 has no `thresholds` field.
    """
    conditions = [
        RangeCondition(min=thresholds[i - 1] if i > 0 else None, max=thresholds[i])
        for i in range(len(thresholds))
    ]
    return CasesRanges(
        id=id, feature=Feature(feature), conditions=conditions,
        end_logic=end_logic, strict=False,
    )
