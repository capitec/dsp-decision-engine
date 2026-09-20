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

2. **`Feature` is a column name. `_ComputedFeature` is gone.** decider 1
   lets a feature be an expression string evaluated with `simpleeval`
   (`common/feature.py`). Doc 08 §3.2 removes it, and doc 06 §O15 records
   the decision: "a rule's leaves are declared features or **registered**
   feature ids, never expression strings (`_ComputedFeature` goes —
   §3.2)". A derived value is a step in code, placed in a module *before*
   the tree — `flow(Derived, my_tree)` — which is what decider2's pipeline
   already does for free. A document carrying a computed feature gets a
   load-time error naming that replacement rather than being silently
   accepted or silently dropped.

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

from pydantic import BaseModel, Discriminator, Field, RootModel, Tag, model_validator

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
    """Raised when a document carries decider 1's `_ComputedFeature`.

    Doc 08 §3.2 removes the expression-string feature; doc 06 §O15 records
    it as settled. This is a loud, named failure with the replacement in the
    message, because the two silent alternatives are both wrong: accepting
    it would put an interpreter in config (doc 08 §1.1 — "config may
    reference code by registered id. It may not contain code"), and dropping
    it would change the tree's answers without saying so.
    """


class Feature(RootModel[str]):
    """A column name — decider 1's `common/feature.Feature`, narrowed.

    decider 1's root union also admitted `_ComputedFeature`. See divergence
    2 in the module docstring: a derived value is a step in a module before
    the tree, not an expression in the document.
    """

    root: str = Field(description="Feature name to test")

    @model_validator(mode="before")
    @classmethod
    def _reject_computed(cls, value: t.Any) -> t.Any:
        is_computed = (
            isinstance(value, dict) and value.get("type") == "computed"
        ) or getattr(value, "type", None) == "computed"
        if is_computed:
            expression = (
                value.get("expression")
                if isinstance(value, dict)
                else getattr(value, "expression", "?")
            )
            raise ComputedFeatureRemoved(
                f"computed feature {expression!r} cannot be migrated as-is. "
                "decider 1 evaluated this string with simpleeval; doc 08 §3.2 "
                "removes expression-string features and doc 06 §O15 records "
                "the decision ('a rule's leaves are declared features or "
                "registered feature ids, never expression strings'). Compute "
                "it as an ordinary step in a module before the tree — "
                "`def my_feature(a, b): return a - b`, then "
                "`flow(my_feature, my_tree)` — and reference the result by "
                "name here."
            )
        return value

    def __str__(self) -> str:
        return self.root

    def required_features(self) -> set[str]:
        return {self.root}

    def required_params(self) -> set[str]:
        return set()


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


class _ThresholdedUnaryOp(_BaseUnaryOp):
    threshold: Threshold = Field(
        description="Comparison value (number or InputRef for a runtime variable)"
    )

    def required_params(self) -> set[str]:
        params = self.feature.required_params()
        if isinstance(self.threshold, InputRef):
            params.add(self.threshold.key)
        return params


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


class UnaryIsTrue(_BaseUnaryOp):
    op: t.Literal["is_true"] = "is_true"


class UnaryIsFalse(_BaseUnaryOp):
    op: t.Literal["is_false"] = "is_false"


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


class IsInCondition(BaseModel):
    """One branch of a `CasesIsIn`."""

    values: t.Union[t.List[Threshold], InputRef]

    def required_params(self) -> set[str]:
        if isinstance(self.values, InputRef):
            return {self.values.key}
        return {v.key for v in self.values if isinstance(v, InputRef)}


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


CompositeCondition.model_rebuild()


# ---------------------------------------------------------------------------
# Nodes — decider 1's `tree/v3/nodes_ui.py`
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


class UnaryNode(BaseModel):
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


class CasesRanges(BaseModel):
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


class CasesStringMatch(BaseModel):
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


class CasesIsIn(BaseModel):
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


class CompositeNode(BaseModel):
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
