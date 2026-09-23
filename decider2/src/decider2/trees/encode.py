"""Tree document -> `Step`s built directly, `fn` a pre-built njit closure
(doc 05 §4, doc 08 §3.4).

**Renamed from `codegen.py`.** That name told a reader the opposite of the
truth even after the previous migration: nothing here generates Python
SOURCE TEXT any more — not the tree's shape (already true before this
rename: a tree's shape is the flat `kind`/`feat_kind`/`feat_idx`/`op`/
`thr_slot`/`then`/`else`/`leaf_value` arrays `EncodeContext` builds) and,
as of this pass, not the per-tree WRAPPER either. `types.Step` is a frozen
dataclass whose `inputs: tuple[Input, ...]` and `params: tuple[ParamDecl,
...]` are already data — nothing forces them to come from
`inspect.signature`. So the wrapper `emit_tree` used to render as a `.py`
file (`def demo_path(income: float, root_thr: float = param(5000.0)) ->
int: ...`) is built here directly as a `Step`, with `fn` a real Python
closure over the tree's own numpy arrays (never written as source
literals) and `inputs`/`params` built straight from the walk's own
bookkeeping.

**Arity.** A closure cannot have one parameter per feature/threshold — that
count varies per tree, and a closure's own parameter list is fixed the
moment it is written. Every `fn` built here therefore has the SAME two-
argument shape regardless of tree size: `fn(args, params)`, where `args`
is the per-row five-tuple of typed row arrays and `params` the `(floats,
ints, pat_bytes, pat_off, grp_off)` bundle — thresholds plus the pattern
table (`types.Step.typed_args`). `decider2.compile.driver`'s packed-call
helpers are what every execution mode (interpreted/stepped/fused/fallback)
calls this shape through; see that module.

**Features are split by type, and the type is program data.** Every
feature the tree reads has a KIND — `float`, `int`, `bool` or `str` —
declared by the caller (`encode_tree(..., feature_types=...)`) or inferred
from how the tree uses it (`is_true`/`is_false` only -> `bool`;
`string_match` -> `str`; anything else -> `float`, the previous behaviour
exactly). A node row carries `feat_kind` beside `feat_idx`, so it says
"slot 7 of the int64 array", and its threshold lives in the int64
threshold tuple when the feature is an int. The previous encoder coerced
every feature — and every threshold — into float64, which collapses any
integer above 2**53 and made `9007199254740993 == 9007199254740992` answer
True: a silent wrong answer on exactly the scaled-int64 money columns doc
03 §1.2 mandates. Knowing the kind at build time is also what lets this
module REJECT what the single array could never see: an ordering
comparison on a categorical (`sector < 5` compared dictionary codes), a
threshold on a boolean, a fractional threshold on an int feature — each is
a `ValueError` here, naming the node.

Kinds are settled BEFORE the real encode by a cheap probe walk (`_infer_
kinds`): the same `encode()` methods run once with every position ignored,
recording only what each feature is asked to do, so that in the real walk
every slot position is stable the instant it is handed out (a computed
feature's expression closure captures its float-array positions at compile
time, during the walk — see `expr_feature_index`).

**The switch that is left lives on the classes it used to switch over, not
in this module.** Unchanged from the previous migration: every node and
condition class in `trees/schema.py` implements its own `encode(ctx, ...)`
— its row(s) of the flat arrays — recursing into its children through
`EncodeContext`. This module holds `EncodeContext` (naming and de-duping
kernel arguments, hoisting a string test, walking to a node's children,
counting leaves/depth for `.explain()`) plus the module-level scaffolding
that turns the walk's own bookkeeping into `Step`s. It imports none of the
node or condition classes, and never asks a node "which one are you".

Three properties this module exists to hold, carried over from the previous
migration:

**1. No decision-relevant constant reaches emitted source — because there
is no emitted source.** Every threshold — literal or `InputRef` — becomes a
`ParamDecl` (a kernel argument, `Step.params`). A literal is a param holding
that default; an `InputRef` is a param named by its key. Retuning either is
a values change: free, no recompile (doc 08 §2).

**2. A tree's SHAPE is data, not text, so there is no line cap, no fan-out
wall, no CPython indentation limit and no depth cap.** `TreeTooLarge`
survives only for `TableTooComplex`'s unrelated DNF-explosion guard
(`tables/schema.py`) — nothing in THIS module raises it any more.

**3. A string is tested at the node, by its bytes, against patterns that
are params.** docs/BOUNDARY-REWORK.md §3.1 (which retires doc 05 §1.5's
"a string never enters a kernel as a string" for trees). A `str` feature
is a `bytes`-annotated path-step input — the wire spelling of a STRING
SPAN, `(address, byte length)` into polars' own memory through the
compiled Arrow shim — and every `string_match` becomes a `STR_MATCH` node
(`EncodeContext.encode_string_match`) that compares those bytes against a
PATTERN GROUP: a node's literal patterns are ONE `list[str]` param, an
`InputRef` pattern is a `str` param, and both ride in the per-call
pattern table `compile.gather._typed_params` builds. The table's numba
type does not depend on how many patterns there are or how long, so
editing, adding or removing a pattern is a value change, never a
recompile — the same guarantee property 1 gives a threshold. What the
kernel cannot do (`regex`, `case_sensitive=False`, `trim_whitespace`) is
still refused by name with the frame-tier route (`UnsupportedInKernel`).
"""
from __future__ import annotations

import keyword
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numba import literal_unroll, njit
from pydantic import Field

from decider2.trees.interpreter import (
    BOOL, CONTAINS, EXACT, F64, GE, GT, I64, LE, LT, PREFIX, STR, STR_MATCH, SUFFIX, walk_tree,
)
from decider2.trees.schema import InputRef, Tree, TStringMatchType
from decider2.types import Input, NullPolicy, ParamDecl, Step

__all__ = [
    "TreeTooLarge",
    "UnsupportedInKernel",
    "EncodedTree",
    "EncodeContext",
    "encode_tree",
    "normalise_feature_type",
    "LINE_CAP",
    "safe_ident",
]

# Kept only so `tables/schema.py`'s `TableTooComplex(TreeTooLarge)` and its
# message (an unrelated DNF-explosion guard) still import; nothing in this
# module raises either any more.
LINE_CAP = 500

_IDENT_BAD = re.compile(r"[^0-9A-Za-z_]")

# The four feature kinds a tree can read, as the caller names them.
# `FeatureKind` (types.py) is the runtime slot; this is the authoring-level
# vocabulary, one name per Python annotation the boundary already
# understands (`compile.driver.numpy_dtype`).
_KIND_ALIASES: dict[str, str] = {
    "float": "float", "float64": "float", "float32": "float",
    "int": "int", "int64": "int", "int32": "int", "int16": "int", "int8": "int",
    "uint64": "int", "uint32": "int", "uint16": "int", "uint8": "int",
    "bool": "bool", "boolean": "bool",
    "str": "str", "string": "str", "utf8": "str", "categorical": "str", "enum": "str",
}
# A feature's kind -> the path step's INPUT annotation. `str` is `bytes`:
# the wire spelling of a string span (`types.FeatureKind.STR`), which the
# boundary produces from a String column (`boundary.extract`) and
# `score()` from `str.encode()`; `str` itself stays the dictionary-code
# convention of hand-written steps (doc 05 §1.5) and never names a tree
# input.
_INPUT_ANNOTATION_BY_KIND: dict[str, Any] = {"float": float, "int": int, "bool": bool, "str": bytes}
_SLOT_BY_KIND: dict[str, int] = {"float": F64, "int": I64, "bool": BOOL, "str": STR}
# A `_ParamSlot.annotation` -> the `ParamDecl` annotation. `patterns` is a
# string-match node's literal list — one param, one pattern GROUP (the
# module docstring's property 3).
_PARAM_ANNOTATION: dict[str, Any] = {"float": float, "int": int, "str": str, "patterns": list[str]}
_ORDERING = frozenset((LT, LE, GT, GE))
_MATCH_OP: dict[TStringMatchType, int] = {
    TStringMatchType.exact: EXACT,
    TStringMatchType.starts_with: PREFIX,
    TStringMatchType.ends_with: SUFFIX,
    TStringMatchType.contains: CONTAINS,
}


def normalise_feature_type(feature: str, value: Any) -> str:
    """`float`/`int`/`bool`/`str` from a Python type, a polars dtype, or
    decider 1's own spelling (`"Int64"`, `"Boolean"`, `"String"`)."""
    if value in (float, int, bool, str):
        return value.__name__
    key = str(value).strip().lower()
    kind = _KIND_ALIASES.get(key)
    if kind is None:
        raise ValueError(
            f"feature_types[{feature!r}] = {value!r} is not a recognised feature "
            "type. Use float, int, bool or str (or a polars spelling: Float64, "
            "Int64, Boolean, String, Categorical)."
        )
    return kind


class TreeTooLarge(ValueError):
    """No longer raised by this module. Kept as a base class for
    `tables.schema.TableTooComplex`, whose own guard (bounding And/Or
    disjunct explosion) is a genuinely different, still-relevant concern —
    see that module."""


class UnsupportedInKernel(ValueError):
    """A decider 1 construct with no compiled-kernel form.

    Raised rather than silently approximated. Every instance names what was
    asked for, why a kernel cannot express it, and the route that works.
    """


def safe_ident(name: str) -> str:
    """A valid Python identifier, deterministically derived from `name` —
    used for `Step`/param NAMES (still real identifiers even though nothing
    here writes them into source any more: `params_schema()`, `GET
    /params/schema`, `.explain()` and every other reporting surface reads
    these names back)."""
    ident = _IDENT_BAD.sub("_", name)
    if not ident or ident[0].isdigit():
        ident = f"v_{ident}"
    if keyword.iskeyword(ident):
        ident = f"{ident}_"
    return ident


@dataclass
class _ParamSlot:
    """One threshold, on its way to becoming a `ParamDecl`."""

    name: str
    default: Any
    annotation: str  # "float" | "int" | "str"
    origin: str      # human-readable: which node and which bound


@dataclass
class EncodedTree:
    """Everything `decider2.trees.build` needs to turn the walk into a
    `Module`: the built `Step`s themselves, plus what only the document
    knows (for `.explain()` and `TreeModule.decode()`)."""

    path_step: Step
    output_steps: tuple[Step, ...]
    path_column: str
    features: tuple[str, ...]
    string_features: tuple[str, ...]
    params: tuple[_ParamSlot, ...]
    leaf_count: int
    max_depth: int
    arrays: dict[str, list[int]]  # for determinism checks — see the report
    feature_kinds: dict[str, str] = field(default_factory=dict)  # name -> float|int|bool|str


# A negative feature slot, during the encode walk, is a DEFERRED reference:
# `-(k + 1)` names the k-th computed feature, whose float64 row position
# (after every plain float feature) isn't known until the whole tree has
# been walked. Resolved to a real, non-negative array index only once, in
# `resolve_arrays` below — never by shifting anything already assigned a
# plain (>= 0) index, which is stable the moment it is handed out.


class EncodeContext:
    """State threaded through one tree's encoding.

    A node or condition class in `trees/schema.py` calls back into this for
    the concerns that are not any one class's own: naming and de-duping
    feature and threshold arguments, hoisting a string comparison into its
    own step, walking from a node to its children, appending rows to the
    walker's flat arrays, and counting leaves and depth for `.explain()`.
    One instance per walk, and not reusable across trees: determinism (doc
    05 §4.2) depends on the walk being in one fixed order, and this
    accumulates node rows and param slots as it goes. Nothing here iterates
    a set.

    This class never asks a node or condition "which one are you" — it
    calls `.encode(...)` and lets the class append its own rows.

    `kinds` is every feature's settled kind (`float`/`int`/`bool`/`str`).
    `None` puts the context in PROBE mode: the same walk runs, every
    position it hands out is meaningless and discarded, and the only thing
    kept is `demands` — what each feature was asked to do — which
    `_infer_kinds` turns into `kinds` for the real walk. Every rejection
    below (`<` on a categorical, a threshold on a boolean, ...) fires in
    the real walk only, where the kind is known.
    """

    def __init__(self, tree: Tree, name: str, kinds: "dict[str, str] | None") -> None:
        self.tree = tree
        self.name = safe_ident(name)
        self._kinds = kinds
        self.probe = kinds is None
        # feature name -> {"order", "eq", "bool", "str", "expr"} — probe only
        self.demands: dict[str, set[str]] = {}

        # -- the walker's flat node arrays, one entry per program-counter --
        self._kind: list[int] = []
        self._feat_ref: list[int] = []   # plain feature index, or deferred (<0) computed
        self._op: list[int] = []
        self._thr_slot: list[int] = []   # may hold deferred (<0) entries
        self._then: list[int] = []
        self._else: list[int] = []
        self._leaf_value: list[int] = []

        # -- feature space: PLAIN (named, boundary) features, in first-use
        # order — the path step's `Input` order, and the order each kind's
        # row-array positions are counted in. --
        self.features: list[str] = []
        self._feature_index: dict[str, int] = {}
        # computed feature -> compiled closure (f64_row, thr_f) -> float
        self._computed: list[Any] = []

        # -- threshold space: every registered param (float, int, str and
        # patterns) --
        self.params: list[_ParamSlot] = []
        self._param_by_ident: dict[str, _ParamSlot] = {}
        # position among FLOAT params / among INT params / among PATTERN
        # GROUPS (`str` and `patterns` params share one index space: each
        # is one group of the pattern table), each in first-use order —
        # what `walk_tree` indexes `thr_f`/`thr_i`/`grp_off` by. The driver
        # derives the same positions from `Step.params` order alone
        # (`compile.gather._typed_params`), so the two cannot disagree.
        self._thr_pos: dict[str, dict[str, int]] = {"float": {}, "int": {}, "pat": {}}

        self.max_depth = 0
        self.leaf_count = 0
        self._node_map = tree.node_map()
        self._children = tree.children()
        self._seen: set[str] = set()

    # -- kinds ---------------------------------------------------------------

    def feature_kind(self, name: str) -> str:
        """The settled kind of a plain feature (`float` in probe mode)."""
        if self._kinds is None:
            return "float"
        return self._kinds.get(name, "float")

    def _slot_kind(self, feat_ref: int) -> int:
        """Which typed ROW ARRAY a feature reference reads from
        (`FeatureKind` value): a computed feature is always float64; a
        `str` feature is a STRING SPAN slot (`FeatureKind.STR`)."""
        if feat_ref < 0:
            return F64
        return _SLOT_BY_KIND[self.feature_kind(self.features[feat_ref])]

    def _demand(self, feat_ref: int, tag: str) -> None:
        if self.probe and feat_ref >= 0:
            self.demands.setdefault(self.features[feat_ref], set()).add(tag)

    def _feature_desc(self, feat_ref: int) -> str:
        name = self.features[feat_ref]
        return f"feature '{name}' (declared/inferred {self.feature_kind(name)})"

    # -- node-array building, called from schema.py's node/condition classes

    def _add_node(
        self, kind: int, *, feat_ref: int = 0, op: int = 0, thr_slot: int = 0,
        then_: int = 0, else_: int = 0, leaf_value: int = 0,
    ) -> int:
        pc = len(self._kind)
        self._kind.append(kind)
        self._feat_ref.append(feat_ref)
        self._op.append(op)
        self._thr_slot.append(thr_slot)
        self._then.append(then_)
        self._else.append(else_)
        self._leaf_value.append(leaf_value)
        return pc

    def add_leaf(self, result_idx: int) -> int:
        from decider2.trees.interpreter import LEAF

        self.leaf_count += 1
        return self._add_node(LEAF, leaf_value=result_idx)

    def add_cmp(
        self, feat_idx: int, op: int, thr_slot: int, then_pc: int, otherwise_pc: int,
    ) -> int:
        """One `CMP` row. A numeric comparison on a `str` or `bool` feature
        is rejected HERE, at build time, naming the node: a string has no
        numeric order (`sector < 5` used to compare dictionary codes,
        silently), and a boolean is tested with `is_true`/`is_false`, not
        a threshold. A string is tested with `string_match`, which builds
        a `STR_MATCH` node (`encode_string_match`), never a `CMP`."""
        from decider2.trees.interpreter import CMP

        self._demand(feat_idx, "order" if op in _ORDERING else "eq")
        if not self.probe and feat_idx >= 0:
            kind = self.feature_kind(self.features[feat_idx])
            if kind == "str":
                raise ValueError(
                    f"tree '{self.tree.name}': a numeric comparison on "
                    f"{self._feature_desc(feat_idx)}. A string is compared by its "
                    "bytes (docs/BOUNDARY-REWORK.md §3.1), which have no order and "
                    "no numeric meaning — the comparison would be silently wrong. "
                    "Use op='string_match' on it, or declare it a numeric type in "
                    "feature_types= if it really is one."
                )
            if kind == "bool":
                raise ValueError(
                    f"tree '{self.tree.name}': a threshold comparison on "
                    f"{self._feature_desc(feat_idx)}. Test a boolean with "
                    "op='is_true'/'is_false', or declare the feature int/"
                    "float in feature_types= if it is really a 0/1 number."
                )
        return self._add_node(CMP, feat_ref=feat_idx, op=op, thr_slot=thr_slot, then_=then_pc, else_=otherwise_pc)

    def _add_truth_test(self, node_kind: int, feat_idx: int, then_pc: int, otherwise_pc: int) -> int:
        self._demand(feat_idx, "bool")
        if not self.probe and feat_idx >= 0 and self.feature_kind(self.features[feat_idx]) == "str":
            raise ValueError(
                f"tree '{self.tree.name}': is_true/is_false on "
                f"{self._feature_desc(feat_idx)}. A string has no truth value "
                "(and an empty string is not a null). Use op='string_match' on it."
            )
        return self._add_node(node_kind, feat_ref=feat_idx, then_=then_pc, else_=otherwise_pc)

    def add_is_true(self, feat_idx: int, then_pc: int, otherwise_pc: int) -> int:
        from decider2.trees.interpreter import IS_TRUE

        return self._add_truth_test(IS_TRUE, feat_idx, then_pc, otherwise_pc)

    def add_is_false(self, feat_idx: int, then_pc: int, otherwise_pc: int) -> int:
        from decider2.trees.interpreter import IS_FALSE

        return self._add_truth_test(IS_FALSE, feat_idx, then_pc, otherwise_pc)

    # -- features -----------------------------------------------------------

    def column(self, name: str) -> str:
        """Register a plain named column as read by this tree (idempotent)
        and return its identifier. Order of first use is the path step's
        `Input` order — doc 05 §4.2's determinism depends on callers never
        re-ordering this themselves.

        This is also what a computed feature's own free names resolve
        through (`expr_feature_index`) — a plain column `x` and the same
        `x` read inside `x - y` share the one slot, because both paths end
        up calling this exact method with the exact same string.
        """
        if name not in self._feature_index:
            self._feature_index[name] = len(self.features)
            self.features.append(name)
        return safe_ident(name)

    def plain_feature_index(self, name: str) -> int:
        """The plain feature's index in `features` (first-use order) —
        stable the instant it is first assigned. This is a feature
        REFERENCE, not yet a row-array slot: `resolve_arrays` turns it into
        `(feat_kind, feat_idx)` once every kind is known."""
        self.column(name)
        return self._feature_index[name]

    def expr_feature_index(self, name: str, node_id: str) -> int:
        """A computed feature's free name, as the slot in the FLOAT64 row
        array its compiled closure will read (`expr.Expr.compile` captures
        this at compile time, during the walk, which is why the kind must
        already be settled). A feature declared `int`/`bool`/`str` cannot be
        read by an expression: `decider2.expr` is float64 arithmetic (doc
        03 §1.2: "accumulate in float64"), and reading an int64 into it
        would be exactly the silent widening this module exists to stop —
        so it is refused with the node named, rather than done quietly."""
        ref = self.plain_feature_index(name)
        self._demand(ref, "expr")
        if self.probe:
            return ref
        kind = self.feature_kind(name)
        if kind != "float":
            raise ValueError(
                f"tree '{self.tree.name}' node '{node_id}': the computed feature "
                f"reads '{name}', which is declared {kind}. A computed feature is "
                "float64 arithmetic (decider2.expr) and cannot read a typed "
                "int/bool/str column without widening it — the silent precision "
                "loss typed features exist to prevent (doc 03 §1.2). Declare "
                f"'{name}' float, or compute the value in a step before the tree."
            )
        return self._f64_position(ref)

    def _f64_position(self, ref: int) -> int:
        """`ref`'s slot within the float64 row array: how many earlier-
        registered features also live there. Fixed the moment `ref` is
        registered — a later feature can never shift it."""
        return sum(1 for r in range(ref) if self._slot_kind(r) == F64)

    def computed_feature_index(self, computed: Any, node_id: str) -> int:
        """A computed feature's slot — DEFERRED (negative-encoded) because
        the float64 row array's plain-feature count, the offset it resolves
        against, is not final until the whole tree has been walked (see
        `_resolve_deferred`). `computed.compile(...)` is `decider2.expr`'s
        own build-once closure compiler; nothing here re-implements it.
        `_ComputedFeature.compile` (schema.py) builds its own
        `_ExprCompileAdapter(ctx, node_id)` internally, so this just hands
        itself across.
        """
        fn = computed.compile(self, node_id)
        k = len(self._computed)
        self._computed.append(fn)
        return -(k + 1)

    # -- thresholds -----------------------------------------------------

    def _add_param(self, name: str, default: Any, annotation: str, origin: str) -> _ParamSlot:
        """Register a kernel argument, de-duplicating by name.

        An `InputRef` deliberately collides with itself: two nodes naming
        `#income_floor` share one knob, which is the whole point of a named
        reference (decider 1's `common/parameters.py`). A generated literal
        name never collides, because it carries the node's own id. One
        name with two TYPES (`#floor` against an int feature here and a
        float one there, or a matcher pattern reused as a number) is an
        error: the knob would have to be two different kernel arguments.
        """
        ident = safe_ident(name)
        existing = self._param_by_ident.get(ident)
        if existing is not None:
            if existing.annotation != annotation:
                raise ValueError(
                    f"tree '{self.tree.name}': param '{ident}' is used as "
                    f"{existing.annotation} ({existing.origin}) and as {annotation} "
                    f"({origin}). One InputRef is one kernel argument of one type "
                    "— split it into two keys, or give both features the same "
                    "type in feature_types=."
                )
            return existing
        slot = _ParamSlot(name=ident, default=default, annotation=annotation, origin=origin)
        self._param_by_ident[ident] = slot
        self.params.append(slot)
        return slot

    def threshold_slot(self, value: Any, *, node_id: str, role: str, feat_idx: "int | None" = None) -> int:
        """The walker's threshold-tuple slot for one threshold — a `CMP`
        node's `thr_slot` field, and also what a computed feature's own
        inline constant resolves through (`_ExprCompileAdapter.
        constant_index`, always float). `feat_idx` is the feature the
        threshold is compared against: its kind decides WHICH tuple —
        `int` features take int64 thresholds (comparing an int64 against a
        float64 threshold would promote the int and re-introduce the 2**53
        collapse one operand over), so the param is declared `int`, must be
        integral, and an `InputRef` defaults to 0. Stable the instant it is
        first assigned: position among that kind's params only, in first-
        use order (`str`-typed params — a matcher's own literals — never
        occupy a slot here, so their registration never shifts an already-
        handed-out position)."""
        kind = "float"
        if feat_idx is not None and feat_idx >= 0 and not self.probe:
            fk = self.feature_kind(self.features[feat_idx])
            if fk == "int":
                kind = "int"
            elif fk in ("bool", "str"):
                # `add_cmp` gives the fuller message; raise the same one
                # here so the threshold's own registration never happens.
                self.add_cmp(feat_idx, LT, 0, 0, 0)  # always raises for these kinds
        if isinstance(value, InputRef):
            default: Any = 0 if kind == "int" else 0.0
            slot = self._add_param(value.key, default, kind, f"InputRef #{value.key} ({node_id}.{role})")
        else:
            if kind == "int":
                if isinstance(value, bool) or float(value) != int(value):
                    raise ValueError(
                        f"tree '{self.tree.name}' node '{node_id}': threshold {value!r} "
                        f"({role}) is not an integer, but {self._feature_desc(feat_idx)} "
                        "is compared as an int64. Use an integral threshold, or "
                        "declare the feature float in feature_types=."
                    )
                slot = self._add_param(f"{node_id}_{role}", int(value), "int", f"{node_id}.{role}")
            else:
                slot = self._add_param(f"{node_id}_{role}", float(value), "float", f"{node_id}.{role}")
        positions = self._thr_pos[kind]
        ident = slot.name
        if ident not in positions:
            positions[ident] = len(positions)
        return positions[ident]

    # -- string matching at the node -------------------------------------

    def _pattern_group(self, name: str, default: Any, annotation: str, origin: str) -> int:
        """Register one pattern group (a `str` param — one pattern — or a
        `patterns` param — a node's literal list) and return its index in
        the pattern table's group space (`grp_off`), stable the moment it
        is first assigned, first-use order — the same rule as
        `threshold_slot`, over the shared str/patterns index space that
        `compile.gather._typed_params` numbers from `Step.params` order."""
        slot = self._add_param(name, default, annotation, origin)
        positions = self._thr_pos["pat"]
        if slot.name not in positions:
            positions[slot.name] = len(positions)
        return positions[slot.name]

    def encode_string_match(
        self,
        feature: Any,
        patterns: Sequence[Any],
        match_type: TStringMatchType,
        case_sensitive: bool,
        trim_whitespace: bool,
        node_id: str,
        then_pc: int,
        otherwise_pc: int,
        role: str = "patterns",
    ) -> int:
        """A string test, encoded as `STR_MATCH` node(s) that compare the
        feature's bytes at the node (docs/BOUNDARY-REWORK.md §3.1).

        `patterns` is an OR. Its literal members become ONE `list[str]`
        param named `<node_id>_<role>` — one pattern group, so editing the
        list (its texts, its length) is a value change; each `InputRef`
        member becomes a `str` param named by its key — a group of one,
        shared by every node naming the same key. One node per group,
        chained like `_encode_isin` (first group tested first, any match
        wins). Shared by `UnaryStringMatch.encode` and
        `StringMatchCondition.encode`; `role` keeps a `CasesStringMatch`'s
        sibling conditions on distinct param names.

        What a kernel cannot do is refused by name, with the frame-tier
        route: `regex` (no engine in nopython numba; BOUNDARY-REWORK.md
        §5), `case_sensitive=False` (Unicode case folding is a table the
        kernel does not have) and `trim_whitespace` (Unicode whitespace,
        likewise).
        """
        root = getattr(feature, "root", feature)
        if not isinstance(root, str):
            raise UnsupportedInKernel(
                f"node '{node_id}' string-matches a computed feature "
                f"('{feature}'). A computed feature (decider2.expr) emits a "
                "numeric expression, never a string column, so there are no "
                "bytes to match against. Give the tree a named string column "
                "instead."
            )
        if match_type is TStringMatchType.regex:
            raise UnsupportedInKernel(
                f"node '{node_id}' uses match_type='regex' on feature '{feature}'. "
                "There is no regex engine in nopython numba (docs/BOUNDARY-REWORK.md "
                "§5; EXPERIMENTS.md §O), and a backtracking engine in the kernel "
                "would be a per-row DoS axis opened by a rule edit. Evaluate the "
                "regex in the frame tier (`pl.col(...).str.contains(pattern)`) into "
                "a boolean column before the pipeline and branch on it here with "
                "op='is_true'; exact, starts_with, ends_with and contains are "
                "matched in the kernel."
            )
        if not case_sensitive:
            raise UnsupportedInKernel(
                f"node '{node_id}' sets case_sensitive=False on feature '{feature}'. "
                "The kernel compares bytes; Unicode case folding needs a table it "
                "does not have (docs/BOUNDARY-REWORK.md §5). Normalise the column "
                "once in the frame tier (`pl.col(...).str.to_lowercase()`) and "
                "lower-case the patterns in this document."
            )
        if trim_whitespace:
            raise UnsupportedInKernel(
                f"node '{node_id}' sets trim_whitespace=True on feature '{feature}'. "
                "Same reason as case_sensitive=False (Unicode whitespace): strip the "
                "column in the frame tier (`pl.col(...).str.strip_chars()`) before "
                "the pipeline."
            )

        fname = str(feature)
        feat_idx = self.plain_feature_index(fname)
        self._demand(feat_idx, "str")
        if not self.probe and self.feature_kind(fname) != "str":
            raise ValueError(
                f"tree '{self.tree.name}' node '{node_id}': string_match on "
                f"{self._feature_desc(feat_idx)}. A string_match needs a str "
                "column (its bytes). Declare the feature str in feature_types=, "
                "or use a numeric comparison."
            )
        mode = _MATCH_OP[match_type]
        literals = [p for p in patterns if not isinstance(p, InputRef)]
        groups: list[int] = []
        if literals:
            groups.append(self._pattern_group(
                f"{node_id}_{role}", list(literals), "patterns", f"{node_id}.{role}",
            ))
        for p in patterns:
            if isinstance(p, InputRef):
                groups.append(self._pattern_group(
                    p.key, "", "str", f"InputRef #{p.key} ({node_id}.{role})",
                ))
        entry = otherwise_pc
        for g in reversed(groups):
            entry = self._add_node(
                STR_MATCH, feat_ref=feat_idx, op=mode, thr_slot=g, then_=then_pc, else_=entry,
            )
        return entry

    # -- the tree walk, called from schema.py's node classes ---------------

    def walk(self) -> int:
        root = self.tree.root_id()
        return self._enter(root, depth=1)

    def _enter(self, node_id: str, depth: int) -> int:
        """Look up one node, guard it, and let it append its own row(s).

        The only dispatch left is the `node.data.encode(...)` call: whichever
        `NodeData` member the discriminated union already resolved
        `node.data` to is the one whose `encode` runs. Nothing here asks
        which one it is.
        """
        if node_id in self._seen:
            raise ValueError(
                f"tree '{self.tree.name}' revisits node '{node_id}' — a tree must "
                "be acyclic and each node reachable once. (A shared subtree would "
                "duplicate its array rows, doubling the work every row that "
                "reaches it does.)"
            )
        self._seen.add(node_id)
        self.max_depth = max(self.max_depth, depth)
        node = self._node_map[node_id]
        return node.data.encode(self, node_id, depth)

    def child_entry(self, parent_id: str, index: int, depth: int) -> int:
        """The entry program-counter for one branch.

        An unconnected branch is decider 1's `LeafRule(result_idx=-1)` —
        `v3/tree.to_flat_rule_tree`'s `get_child`, same default.
        """
        target = self._children.get(parent_id, {}).get(index)
        if target is None:
            self.max_depth = max(self.max_depth, depth)
            return self.add_leaf(-1)
        if target in self._seen:
            raise ValueError(
                f"tree '{self.tree.name}' revisits node '{target}' — a tree must "
                "be acyclic and each node reachable once. (A shared subtree would "
                "duplicate its array rows, doubling the work every row that "
                "reaches it does.)"
            )
        self._seen.add(target)
        self.max_depth = max(self.max_depth, depth)
        node = self._node_map[target]
        return node.data.encode(self, target, depth)

    # -- finished arrays, resolved against final offsets --------------------

    def slot_kinds(self) -> list[int]:
        """Each plain feature's row-array kind, in `features` order."""
        return [self._slot_kind(r) for r in range(len(self.features))]

    def n_f64_plain(self) -> int:
        return sum(1 for k in self.slot_kinds() if k == F64)

    def resolve_arrays(self) -> dict[str, list[int]]:
        """The walker's arrays, every deferred reference resolved: a
        feature reference becomes `(feat_kind, feat_idx)` — which typed
        row array, and the slot within it (its rank among the features
        sharing that kind, in first-use order; a computed feature is a
        float64 slot after every plain one) — and a deferred int literal
        becomes its position after the named int thresholds."""
        kinds = self.slot_kinds()
        position: list[int] = []
        seen_per_kind: dict[int, int] = {}
        for k in kinds:
            position.append(seen_per_kind.get(k, 0))
            seen_per_kind[k] = seen_per_kind.get(k, 0) + 1
        n_f64 = seen_per_kind.get(F64, 0)

        from decider2.trees.interpreter import LEAF

        feat_kind: list[int] = []
        feat_idx: list[int] = []
        for node_kind, ref in zip(self._kind, self._feat_ref):
            if node_kind == LEAF and not kinds:
                # A leaf reads no feature; its `feat_ref` is `_add_node`'s
                # placeholder 0. A tree whose every `Cases*` node has zero
                # arms registers no feature at all, so there is nothing for
                # the placeholder to resolve against.
                feat_kind.append(F64)
                feat_idx.append(0)
            elif ref < 0:
                feat_kind.append(F64)
                feat_idx.append(n_f64 + (-ref - 1))
            else:
                feat_kind.append(kinds[ref])
                feat_idx.append(position[ref])
        return {
            "kind": list(self._kind),
            "feat_kind": feat_kind,
            "feat_idx": feat_idx,
            "op": list(self._op),
            "thr_slot": list(self._thr_slot),
            "then": list(self._then),
            "else": list(self._else),
            "leaf_value": list(self._leaf_value),
        }


# ---------------------------------------------------------------------------
# Closure builders — real, hand-written functions; the only "variable" left
# is which of a small, fixed family gets picked, decided in Python at build
# time from the walk's own counts. Never a string spliced together.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nonempty(items: tuple) -> tuple:
    """A numba-safe homogeneous-TUPLE value: never truly empty (`()` has no
    element type numba can infer for a dynamic-index read that is never
    actually reached — e.g. a single-leaf tree's `walk_tree` call still has
    to TYPE-CHECK its unreachable `thr_f[thr_slot[pc]]`), so an empty
    tuple gets one inert `0.0` sentinel no node ever indexes into. The
    float64 threshold tuple's guard; `_nonempty_i` is the int64 one. The
    row arrays need no such padding: a zero-length numpy array has a real
    dtype already."""
    if len(items) == 0:
        return (0.0,)
    return items


@njit(cache=True)
def _nonempty_i(items: tuple) -> tuple:
    """`_nonempty` for the int64 threshold tuple (sentinel `0`)."""
    if len(items) == 0:
        return (0,)
    return items


def _build_path_fn(arrays: dict, start_pc: int, n_f64_plain: int, computed: list):
    """One tree's `path_fn(args, params) -> int`. `args` is the driver's
    typed row gather for this tree's features — the five-tuple `(f64, i64,
    b8, i32, span)` of `types.Step.typed_args`, each array exactly as wide
    as that kind's feature count (no per-count closure family here any
    more: replaces `_path0`..`_path6`, which raised past 6 computed
    features). `params` is the `(floats, ints, pat_bytes, pat_off,
    grp_off)` bundle the driver groups from `Step.params` by annotation:
    the two threshold tuples plus the pattern table.

    A computed feature's own arithmetic (`decider2.expr.Expr.compile`) is
    ALREADY a real closure, composed once per expression regardless of
    that expression's own depth — this only had a count ceiling on how
    many SEPARATE computed features one tree could have, not on how
    complex any one of them was. `numba.literal_unroll` iterates the
    (heterogeneous — each computed feature is its own compiled Dispatcher)
    tuple of them at whatever length it actually is, writing each result
    after the plain features in a fresh float64 row; see this module's
    report for the measured cost (real but small: most trees have zero
    computed features, so this path is rarely even reached) and the
    `NumbaExperimentalFeatureWarning` it emits (a mature, long-shipped
    numba feature despite the label — weighed and accepted over hand-
    rolling an RPN interpreter for `decider2.expr`, a materially larger and
    riskier change for the same "no arity ceiling" outcome here).
    """
    kind = np.array(arrays["kind"], dtype=np.int32)
    feat_kind = np.array(arrays["feat_kind"], dtype=np.int32)
    feat_idx = np.array(arrays["feat_idx"], dtype=np.int32)
    op = np.array(arrays["op"], dtype=np.int32)
    thr_slot = np.array(arrays["thr_slot"], dtype=np.int32)
    then_ = np.array(arrays["then"], dtype=np.int32)
    else_ = np.array(arrays["else"], dtype=np.int32)
    leaf_value = np.array(arrays["leaf_value"], dtype=np.int64)

    n_computed = len(computed)
    if n_computed == 0:
        # `inline="always"` alongside `cache=True`. `walk_tree` is itself
        # `inline="always"`, so THIS closure is where its body is compiled
        # and cached (the interpreted/stepped modes call `path_fn` from
        # Python); in fused mode `compile.driver.build_packed_kernel`'s
        # per-row loop absorbs `path_fn` in turn, and neither is compiled
        # as a function of its own there. DO NOT put a non-inlined layer
        # between the per-row kernel and the walker: the six row arrays
        # plus the structure arrays crossing a real call per row cost ~1.8x
        # on `apply()` end to end (EXPERIMENTS.md §X; `walk_tree`'s
        # docstring for the mechanism, and for why the typed split itself
        # is not the win).
        @njit(cache=True, inline="always")
        def path_fn(args, params):
            thr_f = _nonempty(params[0])
            thr_i = _nonempty_i(params[1])
            return walk_tree(
                args, thr_f, thr_i, params[2], params[3], params[4],
                kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc,
            )
        return path_fn

    computed_t = tuple(computed)
    n_f = n_f64_plain

    # `inline="always"` here too: `walk_tree` is already spliced into this
    # closure, and inlining the closure itself into the driver's per-row
    # loop removes the last real call on the row path (measured 166-194 ->
    # 149-167 ns/row on a 3-computed-feature tree, a small but consistent
    # win, EXPERIMENTS.md §X). It does not touch the cache discipline
    # below: this closure is never cached, and inlining a never-cached
    # closure creates no cache entry.
    @njit(inline="always")  # not cache=True: closure captures per-tree computed-feature
    # Dispatchers (decider2.expr.Expr.compile's own closures, one per
    # DISTINCT tree) -- the same trade-off decider2.compile.driver's
    # row-gather closures document, for the same reason: repeated
    # per-process factory calls (once per tree here, not once per row)
    # each capturing a freshly-built, DIFFERENT Dispatcher were measured to
    # grow numba's on-disk cache index rather than hit an existing entry.
    def path_fn(args, params):
        thr_f = _nonempty(params[0])
        thr_i = _nonempty_i(params[1])
        f64, i64, b8, i32, span = args
        feats = np.empty(n_f + n_computed, dtype=np.float64)
        for k in range(n_f):
            feats[k] = f64[k]
        j = n_f
        for cf in literal_unroll(computed_t):
            feats[j] = cf(f64, thr_f)
            j += 1
        return walk_tree(
            (feats, i64, b8, i32, span), thr_f, thr_i, params[2], params[3], params[4],
            kind, feat_kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc,
        )
    return path_fn


# One body per numeric/boolean output-column dtype: `_out_step_fn(args,
# params) -> value`, reading the reached leaf's `result_idx` (`args[0]`,
# always an int) from a captured values array — the same "rows are data"
# shape a decision table's own output columns already use, extended here
# from a table's rows to a tree's leaves.
def _build_output_fn(values: np.ndarray, default: Any, py_type: str):
    if py_type == "float":
        @njit(cache=True)
        def out_fn(args, params):
            idx = args[0]
            if idx < 0:
                return default
            return values[idx]
    elif py_type == "int":
        @njit(cache=True)
        def out_fn(args, params):
            idx = args[0]
            if idx < 0:
                return default
            return values[idx]
    else:  # bool
        @njit(cache=True)
        def out_fn(args, params):
            idx = args[0]
            if idx < 0:
                return default
            return values[idx]
    return out_fn


# decider 1 spells output dtypes with polars' own names in `TreeOutput.dtypes`.
# Only the three a kernel can write are emitted as steps
# (`boundary.writeback.KernelOutputs` holds float64/int64/bool groups); a
# "String" column is decoded after the fact by `TreeModule.decode`. Each
# entry is (python type name, numpy dtype name, value coercion) — the array
# analogue of the old `_literal`/`_PY_TYPE_BY_DTYPE` pair.
def _to_float(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _to_int(value: Any) -> int:
    return -1 if value is None else int(value)


def _to_bool(value: Any) -> bool:
    return False if value is None else bool(value)


_COERCE_BY_DTYPE: dict[str, tuple[str, str, Any]] = {
    "Float64": ("float", "float64", _to_float),
    "Float32": ("float", "float64", _to_float),
    "Int64": ("int", "int64", _to_int),
    "Int32": ("int", "int64", _to_int),
    "Boolean": ("bool", "bool_", _to_bool),
}


def _walk(ctx: EncodeContext) -> int:
    """`ctx.walk()` under a recursion ceiling sized to the tree — ordinary
    Python recursion, unbounded by CPython's own indentation limit (nothing
    here emits source text for it to apply to), only by CPython's
    *recursion* ceiling, raised for the duration of this call."""
    needed = len(ctx.tree.nodes) * 4 + 200
    old_limit = sys.getrecursionlimit()
    try:
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        return ctx.walk()
    finally:
        sys.setrecursionlimit(old_limit)


def _infer_kinds(tree: Tree, name: str, declared: Mapping[str, str]) -> dict[str, str]:
    """Every feature's kind, from a probe walk plus `declared`.

    Declared wins outright (the real walk then checks the tree's use of it
    is meaningful for that kind). Otherwise: `string_match` anywhere -> str;
    only ever `is_true`/`is_false` -> bool; anything else (a threshold, an
    `isin`, a computed expression) -> float — which is exactly what every
    feature was before kinds existed, so an undeclared tree encodes as it
    always did. A name in `declared` the tree never reads is a typo, and
    is reported as one."""
    probe = EncodeContext(tree, name, kinds=None)
    _walk(probe)
    unknown = sorted(set(declared) - set(probe.features))
    if unknown:
        raise ValueError(
            f"tree '{tree.name}': feature_types names {unknown}, which the tree "
            f"does not read. Features read: {sorted(probe.features)}."
        )
    kinds: dict[str, str] = {}
    for f in probe.features:
        if f in declared:
            kinds[f] = declared[f]
            continue
        d = probe.demands.get(f, set())
        if "str" in d:
            kinds[f] = "str"
        elif d and d <= {"bool"}:
            kinds[f] = "bool"
        else:
            kinds[f] = "float"
    return kinds


def encode_tree(
    tree: Tree, *, name: str | None = None, feature_types: Mapping[str, Any] | None = None,
) -> EncodedTree:
    """Walk one tree document and build its `Step`s directly.

    Holds, as `Step`s:

    * `<name>_path` — the path step: a `typed_args` packed step whose
      inputs carry each feature's own kind (`float`/`int`/`bool`, or
      `bytes` — a string span — for a `str` feature), whose params are
      every threshold and every string pattern, and whose `fn` makes ONE
      call into `walk_tree`, returning the reached leaf's `result_idx` as
      an `int`. **This is path capture** (doc 03 §7's `<Name>_path`
      convention, int64, "a value the node produces, and you emit it the
      way you emit any other");
    * one step per numeric or boolean output column, mapping the reached
      `result_idx` to that column's value via a captured array lookup.

    `feature_types` declares kinds (`{"income_cents": int, "is_staff":
    bool}`, Python types or polars spellings — `normalise_feature_type`);
    anything not declared is inferred (`_infer_kinds`).

    String-valued output columns are not built as steps: a kernel writes
    `float64`/`int64`/`bool` arrays (`boundary.writeback.KernelOutputs`), so
    a string column has nowhere to land. `decider2.trees.build.TreeModule.
    decode()` maps `<name>_path` back to them in polars afterwards.
    """
    name = safe_ident(name or tree.name or "tree")
    declared = {
        str(feature): normalise_feature_type(str(feature), value)
        for feature, value in (feature_types or {}).items()
    }
    kinds = _infer_kinds(tree, name, declared)
    ctx = EncodeContext(tree, name, kinds=kinds)
    start_pc = _walk(ctx)

    arrays = ctx.resolve_arrays()
    string_features = tuple(f for f in ctx.features if kinds[f] == "str")

    # -- path step ------------------------------------------------------
    # A `str` feature is a `bytes` input (a string span, `_INPUT_ANNOTATION_
    # BY_KIND`); every param — thresholds AND patterns — is a kernel
    # argument of the path step, so a pattern retunes like a threshold.
    path_fn = _build_path_fn(arrays, start_pc, ctx.n_f64_plain(), ctx._computed)
    path_inputs = tuple(
        Input(name=feature, annotation=_INPUT_ANNOTATION_BY_KIND[kinds[feature]], null_policy=NullPolicy.REQUIRED)
        for feature in ctx.features
    )
    path_params = tuple(
        ParamDecl(
            name=slot.name, annotation=_PARAM_ANNOTATION[slot.annotation], default=slot.default,
            field_info=Field(slot.default),
        )
        for slot in ctx.params
    )
    path_name = f"{name}_path"
    path_step = Step(
        name=path_name, fn=path_fn, inputs=tuple(path_inputs), params=path_params,
        doc=f"Which leaf {tree.name!r} reached, as its result_idx.",
        packed=True, typed_args=True, output_annotation=int,
    )

    # -- output steps ------------------------------------------------------
    output_steps: list[Step] = []
    for column, dtype in tree.output.dtypes:
        coerce = _COERCE_BY_DTYPE.get(dtype)
        if coerce is None:
            continue
        py_type, np_dtype, to_value = coerce
        fn_name = safe_ident(column)
        rows = list(tree.output.data)
        default_row = tree.output.default or {}
        values = np.array([to_value(row.get(column)) for row in rows], dtype=np_dtype)
        default_value = to_value(default_row.get(column))

        out_fn = _build_output_fn(values, default_value, py_type)
        output_annotation = {"float": float, "int": int, "bool": bool}[py_type]
        output_steps.append(
            Step(
                name=fn_name, fn=out_fn,
                inputs=(Input(name=path_name, annotation=int, null_policy=NullPolicy.REQUIRED),),
                params=(),
                doc=f"`{column}` for the leaf the tree reached, read from an array.",
                packed=True, output_annotation=output_annotation,
            )
        )

    return EncodedTree(
        path_step=path_step,
        output_steps=tuple(output_steps),
        path_column=path_name,
        features=tuple(ctx.features),
        string_features=string_features,
        params=tuple(ctx.params),
        leaf_count=ctx.leaf_count,
        max_depth=ctx.max_depth,
        arrays=arrays,
        feature_kinds={f: kinds[f] for f in ctx.features},
    )
