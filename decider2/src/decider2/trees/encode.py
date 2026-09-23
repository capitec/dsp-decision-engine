"""Tree document -> `Step`s built directly, `fn` a pre-built njit closure
(doc 05 §4, doc 08 §3.4).

**Renamed from `codegen.py`.** That name told a reader the opposite of the
truth even after the previous migration: nothing here generates Python
SOURCE TEXT any more — not the tree's shape (already true before this
rename: a tree's shape is the flat `kind`/`feat_idx`/`op`/`thr_slot`/`then`/
`else`/`leaf_value` arrays `EncodeContext` builds) and, as of this pass, not
the per-tree WRAPPER either. `types.Step` is a frozen dataclass whose
`inputs: tuple[Input, ...]` and `params: tuple[ParamDecl, ...]` are already
data — nothing forces them to come from `inspect.signature`. So the wrapper
`emit_tree` used to render as a `.py` file (`def demo_path(income: float,
root_thr: float = param(5000.0)) -> int: ...`) is built here directly as a
`Step`, with `fn` a real Python closure over the tree's own numpy arrays
(never written as source literals) and `inputs`/`params` built straight from
the walk's own bookkeeping.

**Arity.** A closure cannot have one parameter per feature/threshold — that
count varies per tree, and a closure's own parameter list is fixed the
moment it is written. Every `fn` built here therefore has the SAME two-
argument shape regardless of tree size: `fn(args, params)`, where `args[i]`
is `Step.inputs[i]`'s value and `params[i]` is `Step.params[i]`'s value, in
order (`types.Step.packed`). `decider2.compile.driver`'s packed-call helpers
are what every execution mode (interpreted/stepped/fused/fallback) calls
this shape through; see that module.

**The switch that is left lives on the classes it used to switch over, not
in this module.** Unchanged from the previous migration: every node and
condition class in `trees/schema.py` implements its own `encode(ctx, ...)`
— its row(s) of the flat arrays — recursing into its children through
`EncodeContext`. This module holds `EncodeContext` (naming and de-duping
kernel arguments, hoisting a string test, walking to a node's children,
counting leaves/depth for `.explain()`) plus the module-level scaffolding
that turns the walk's own bookkeeping into `Step`s. It imports none of the
node or condition classes.

**What is still, deliberately, NOT a switch over node kinds, but is worth
naming precisely.** A computed feature's arithmetic (`decider2.expr`) used
to render as a numba source fragment folded into the wrapper's own text;
`decider2.expr.Expr.compile` now builds it as a real closure instead
(`_ExprCompileAdapter` below), so nothing under this module ever produces a
string that becomes code. A hoisted string matcher's own dispatch
(`_StringMatcher`) is likewise a closure now, not a tiny `if`/`elif` source
fragment — see `_build_matcher_fn`.

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

**3. A string never enters the walker as a string.** Doc 05 §1.5. Every
string comparison is hoisted out of the tree into its own one-input step
(`EncodeContext.encode_string_match`), which takes that column as `str` (an
int32 dictionary code at runtime) plus one `str` param per literal, and
returns an `int`: which pattern matched, or -1.
"""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numba import literal_unroll, njit
from pydantic import Field

from decider2.trees.interpreter import EQ, walk_tree
from decider2.trees.schema import InputRef, Tree, TStringMatchType
from decider2.types import Input, NullPolicy, ParamDecl, Step

__all__ = [
    "TreeTooLarge",
    "UnsupportedInKernel",
    "EncodedTree",
    "EncodeContext",
    "encode_tree",
    "LINE_CAP",
    "safe_ident",
]

# Kept only so `tables/schema.py`'s `TableTooComplex(TreeTooLarge)` and its
# message (an unrelated DNF-explosion guard) still import; nothing in this
# module raises either any more.
LINE_CAP = 500

_IDENT_BAD = re.compile(r"[^0-9A-Za-z_]")


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
    annotation: str  # "float" | "str"
    origin: str      # human-readable: which node and which bound


@dataclass
class EncodedTree:
    """Everything `decider2.trees.build` needs to turn the walk into a
    `Module`: the built `Step`s themselves, plus what only the document
    knows (for `.explain()` and `TreeModule.decode()`)."""

    path_step: Step
    matcher_steps: tuple[Step, ...]
    output_steps: tuple[Step, ...]
    path_column: str
    features: tuple[str, ...]
    string_features: tuple[str, ...]
    params: tuple[_ParamSlot, ...]
    leaf_count: int
    max_depth: int
    arrays: dict[str, list[int]]  # for determinism checks — see the report


# A negative feature/threshold slot, during the encode walk, is a DEFERRED
# reference: `-(k + 1)` names the k-th entry of a list whose own final
# length isn't known until the whole tree has been walked (a computed
# feature's local, or a string match's anonymous literal). Resolved to a
# real, non-negative array index only once, in `_resolve_deferred` below —
# never by shifting anything already assigned a plain (>= 0) index, which is
# stable the moment it is handed out.
def _resolve_deferred(raw: Sequence[int], base: int) -> list[int]:
    return [v if v >= 0 else base + (-v - 1) for v in raw]


class EncodeContext:
    """State threaded through one tree's encoding.

    A node or condition class in `trees/schema.py` calls back into this for
    the concerns that are not any one class's own: naming and de-duping
    feature and threshold arguments, hoisting a string comparison into its
    own step, walking from a node to its children, appending rows to the
    walker's flat arrays, and counting leaves and depth for `.explain()`.
    One instance per `encode_tree` call, and not reusable across trees:
    determinism (doc 05 §4.2) depends on the walk being in one fixed order,
    and this accumulates node rows and param slots as it goes. Nothing here
    iterates a set.

    This class never asks a node or condition "which one are you" — it
    calls `.encode(...)` and lets the class append its own rows.
    """

    def __init__(self, tree: Tree, name: str) -> None:
        self.tree = tree
        self.name = safe_ident(name)

        # -- the walker's flat node arrays, one entry per program-counter --
        self._kind: list[int] = []
        self._feat_idx: list[int] = []   # may hold deferred (<0) entries
        self._op: list[int] = []
        self._thr_slot: list[int] = []   # may hold deferred (<0) entries
        self._then: list[int] = []
        self._else: list[int] = []
        self._leaf_value: list[int] = []

        # -- feature space: PLAIN (named, boundary) features, in first-use
        # order — this list's length is the offset a deferred (computed)
        # feature slot resolves against. --
        self.features: list[str] = []
        self._feature_index: dict[str, int] = {}
        # computed feature -> compiled closure (feats, thresholds) -> float
        self._computed: list[Any] = []

        # -- threshold space: every registered param (float AND str; str
        # entries never enter the `thresholds` tuple, but stay in `params`
        # for `.explain()`'s reporting) --
        self.params: list[_ParamSlot] = []
        self._param_by_ident: dict[str, _ParamSlot] = {}
        # position among FLOAT params ONLY, in first-threshold-use order —
        # the offset a deferred (anonymous literal) threshold slot resolves
        # against, and the order the `thresholds` tuple is built in.
        self._threshold_position: dict[str, int] = {}
        self._literals: list[float] = []

        # feature name -> hoisted string matcher
        self.matchers: dict[str, _StringMatcher] = {}

        self.max_depth = 0
        self.leaf_count = 0
        self._node_map = tree.node_map()
        self._children = tree.children()
        self._seen: set[str] = set()

    # -- node-array building, called from schema.py's node/condition classes

    def _add_node(
        self, kind: int, *, feat_idx: int = 0, op: int = 0, thr_slot: int = 0,
        then_: int = 0, else_: int = 0, leaf_value: int = 0,
    ) -> int:
        pc = len(self._kind)
        self._kind.append(kind)
        self._feat_idx.append(feat_idx)
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

    def add_cmp(self, feat_idx: int, op: int, thr_slot: int, then_pc: int, otherwise_pc: int) -> int:
        from decider2.trees.interpreter import CMP

        return self._add_node(CMP, feat_idx=feat_idx, op=op, thr_slot=thr_slot, then_=then_pc, else_=otherwise_pc)

    def add_is_true(self, feat_idx: int, then_pc: int, otherwise_pc: int) -> int:
        from decider2.trees.interpreter import IS_TRUE

        return self._add_node(IS_TRUE, feat_idx=feat_idx, then_=then_pc, else_=otherwise_pc)

    def add_is_false(self, feat_idx: int, then_pc: int, otherwise_pc: int) -> int:
        from decider2.trees.interpreter import IS_FALSE

        return self._add_node(IS_FALSE, feat_idx=feat_idx, then_=then_pc, else_=otherwise_pc)

    # -- features -----------------------------------------------------------

    def column(self, name: str) -> str:
        """Register a plain named column as read by this tree (idempotent)
        and return its identifier. Order of first use is the walker's
        `feats` tuple order — doc 05 §4.2's determinism depends on callers
        never re-ordering this themselves.

        This is also what a computed feature's own free names resolve
        through (`_ExprCompileAdapter.name_index`) — a plain column `x` and
        the same `x` read inside `x - y` share the one slot, because both
        paths end up calling this exact method with the exact same string.
        """
        if name not in self._feature_index:
            self._feature_index[name] = len(self.features)
            self.features.append(name)
        return safe_ident(name)

    def plain_feature_index(self, name: str) -> int:
        """The slot a plain (non-computed) feature occupies in the walker's
        `feats` tuple — stable the instant it is first assigned."""
        self.column(name)
        return self._feature_index[name]

    def computed_feature_index(self, computed: Any, node_id: str) -> int:
        """A computed feature's slot — DEFERRED (negative-encoded) because
        `len(self.features)`, the offset it resolves against, is not final
        until the whole tree has been walked (see `_resolve_deferred`).
        `computed.compile(...)` is `decider2.expr`'s own build-once closure
        compiler; nothing here re-implements it. `_ComputedFeature.compile`
        (schema.py) builds its own `_ExprCompileAdapter(ctx, node_id)`
        internally, so this just hands itself across.
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
        name never collides, because it carries the node's own id.
        """
        ident = safe_ident(name)
        existing = self._param_by_ident.get(ident)
        if existing is not None:
            return existing
        slot = _ParamSlot(name=ident, default=default, annotation=annotation, origin=origin)
        self._param_by_ident[ident] = slot
        self.params.append(slot)
        return slot

    def threshold_slot(self, value: Any, *, node_id: str, role: str) -> int:
        """The walker's `thresholds` tuple slot for one threshold — a `CMP`
        node's `thr_slot` field, and (since the previous migration) also
        what a computed feature's own inline constant resolves through
        (`_ExprCompileAdapter.constant_index`) — no separate string-token
        path is needed any more. Stable the instant it is first assigned:
        position among FLOAT params only, in first-use order (`tables.py`'s
        str-typed params — a matcher's own literals — never occupy a slot
        here, so their registration never shifts an already-handed-out
        position)."""
        slot = (
            self._add_param(value.key, 0.0, "float", f"InputRef #{value.key} ({node_id}.{role})")
            if isinstance(value, InputRef)
            else self._add_param(f"{node_id}_{role}", float(value), "float", f"{node_id}.{role}")
        )
        ident = slot.name
        if ident not in self._threshold_position:
            self._threshold_position[ident] = len(self._threshold_position)
        return self._threshold_position[ident]

    def literal_slot(self, value: float) -> int:
        """An anonymous threshold-space entry — never a named argument,
        just a plain literal embedded in the closure's own thresholds-
        tuple-building step (a string match's "which pattern index" test
        against a hoisted matcher's int result). DEFERRED for the same
        reason a computed feature's slot is: the offset it resolves against
        (how many named thresholds this tree ended up with) isn't final
        until the walk is done."""
        k = len(self._literals)
        self._literals.append(float(value))
        return -(k + 1)

    # -- hoisted string matching ---------------------------------------

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
    ) -> int:
        """A string comparison, hoisted into its own step, then encoded as
        an ordinary `CMP(EQ)` chain against that step's result.

        Doesn't return source that tests the hoisted step's int result —
        appends nodes that do. See the module docstring for why the matcher
        itself stays a separate step. Shared by `UnaryStringMatch.encode`
        and `StringMatchCondition.encode`.
        """
        root = getattr(feature, "root", feature)
        if not isinstance(root, str):
            raise UnsupportedInKernel(
                f"node '{node_id}' string-matches a computed feature "
                f"('{feature}'). A computed feature (decider2.expr) emits a "
                "numeric expression, never a string column, so it has no "
                "int32 dictionary code to match against (doc 05 §1.5). Give "
                "the tree a named string column instead."
            )
        if match_type is not TStringMatchType.exact:
            raise UnsupportedInKernel(
                f"node '{node_id}' uses match_type={match_type.value!r} on feature "
                f"'{feature}'. Doc 05 §1.5: a string enters a kernel as an int32 "
                "dictionary code, and a code comparison cannot express a prefix, a "
                "suffix, a substring or a regex — numba has no `re` in nopython at "
                "all. EXPERIMENTS.md §O measured the working route and it is 7x "
                "faster than the one you are asking for: shape the string in the "
                "frame tier (`pl.col(...).str.contains(...)`) into a boolean or a "
                "category column before the pipeline, and branch on that column "
                "here with op='is_true' or op='isin'."
            )
        if not case_sensitive:
            raise UnsupportedInKernel(
                f"node '{node_id}' sets case_sensitive=False on feature '{feature}'. "
                "Case folding is a string operation and a kernel only sees the "
                "dictionary code (doc 05 §1.5). Normalise the column once in the "
                "frame tier (`pl.col(...).str.to_lowercase()`) and lower-case the "
                "patterns in this document."
            )
        if trim_whitespace:
            raise UnsupportedInKernel(
                f"node '{node_id}' sets trim_whitespace=True on feature '{feature}'. "
                "Same reason as case_sensitive=False: strip the column in the frame "
                "tier (`pl.col(...).str.strip_chars()`) before the pipeline."
            )

        fname = str(feature)
        feat_idx = self.plain_feature_index(fname)
        matcher = self.matchers.get(fname)
        if matcher is None:
            # The step's OUTPUT name is its function name (types.Step.name),
            # and that is the name the tree body reads — so they must be the
            # same string. Qualified by the tree instance name because two
            # trees testing the same column with different literals must not
            # collide on one value (doc 03 §3.2's waterfall would silently
            # give the second tree the first one's answer).
            matcher = _StringMatcher(
                feature=fname,
                fn_name=f"{self.name}__m_{safe_ident(fname)}",
                literals=[],
            )
            self.matchers[fname] = matcher
        slots = [matcher.literal_slot(p, self) for p in patterns]
        entry = otherwise_pc
        for i in reversed(slots):
            lit = self.literal_slot(float(i))
            entry = self.add_cmp(feat_idx, EQ, lit, then_pc, entry)
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

    def resolve_arrays(self) -> dict[str, list[int]]:
        n_plain = len(self.features)
        n_named_thresholds = len(self._threshold_position)
        return {
            "kind": list(self._kind),
            "feat_idx": _resolve_deferred(self._feat_idx, n_plain),
            "op": list(self._op),
            "thr_slot": _resolve_deferred(self._thr_slot, n_named_thresholds),
            "then": list(self._then),
            "else": list(self._else),
            "leaf_value": list(self._leaf_value),
        }


@dataclass
class _StringMatcher:
    """A hoisted string test: one step, one `str` input, N `str` params.
    Behaviourally unchanged by this pass — only its OWN `fn` is now a
    closure (`_build_matcher_fn`) instead of a tiny `if`/`elif` source
    fragment."""

    feature: str
    fn_name: str
    literals: list[tuple[str, Any]] = field(default_factory=list)

    def literal_slot(self, pattern: Any, ctx: EncodeContext) -> int:
        """The index this literal answers with, registering it if new."""
        if isinstance(pattern, InputRef):
            key, default = pattern.key, ""
        else:
            key, default = f"{safe_ident(self.feature)}_pat_{len(self.literals)}", pattern
        for i, (existing, _) in enumerate(self.literals):
            if existing == key:
                return i
        self.literals.append((key, default))
        ctx._add_param(key, default, "str", f"string literal for '{self.feature}'")
        return len(self.literals) - 1

    @property
    def feature_ident(self) -> str:
        return safe_ident(self.feature)


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
    to TYPE-CHECK its unreachable `thresholds[thr_slot[pc]]`), so an empty
    tuple gets one inert `0.0` sentinel no node ever indexes into. Used for
    `thresholds` (`params + literals`, still a plain tuple — see
    `_build_path_fn`) only: `args`/`feats` is a numpy array below, whose
    empty case (`decider2.compile.driver._gather0`) already has a real
    dtype and needs no padding — wrapping an ARRAY in this same helper
    would force numba to unify `(0.0,)` (a 1-tuple) against an `array(
    float64, 1d)` return type, which does not type-check; that is why this
    stayed tuple-only rather than growing to cover both."""
    if len(items) == 0:
        return (0.0,)
    return items


def _build_path_fn(
    arrays: dict, start_pc: int, n_args: int, computed: list, literals: Sequence[float],
):
    """One tree's `path_fn(args, params) -> int`. `args` is `decider2.
    compile.driver`'s row-gather result for this tree's `n_args` PLAIN/
    matcher-backed features — a float64 array, however wide (no per-count
    closure family here any more: replaces `_path0`..`_path6`, which
    raised past 6 computed features).

    A computed feature's own arithmetic (`decider2.expr.Expr.compile`) is
    ALREADY a real closure, composed once per expression regardless of
    that expression's own depth — this only had a count ceiling on how
    many SEPARATE computed features one tree could have, not on how
    complex any one of them was. `numba.literal_unroll` iterates the
    (heterogeneous — each computed feature is its own compiled Dispatcher)
    tuple of them at whatever length it actually is, writing each result
    into a preallocated `feats` array; see this module's report for the
    measured cost (real but small: most trees have zero computed features,
    so this path is rarely even reached) and the `NumbaExperimentalFeature
    Warning` it emits (a mature, long-shipped numba feature despite the
    label — weighed and accepted over hand-rolling an RPN interpreter for
    `decider2.expr`, a materially larger and riskier change for the same
    "no arity ceiling" outcome here).
    """
    kind = np.array(arrays["kind"], dtype=np.int32)
    feat_idx = np.array(arrays["feat_idx"], dtype=np.int32)
    op = np.array(arrays["op"], dtype=np.int32)
    thr_slot = np.array(arrays["thr_slot"], dtype=np.int32)
    then_ = np.array(arrays["then"], dtype=np.int32)
    else_ = np.array(arrays["else"], dtype=np.int32)
    leaf_value = np.array(arrays["leaf_value"], dtype=np.int64)
    literals_t = tuple(float(v) for v in literals)

    n_computed = len(computed)
    if n_computed == 0:
        # `inline="always"` alongside `cache=True`. `walk_tree` is itself
        # `inline="always"`, so THIS closure is where its body is compiled
        # and cached (the interpreted/stepped modes call `path_fn` from
        # Python); in fused mode `compile.driver.build_packed_kernel`'s
        # per-row loop absorbs `path_fn` in turn, and neither is compiled
        # as a function of its own there. DO NOT put a non-inlined layer
        # between the per-row kernel and the walker: eight arrays crossing
        # a real call per row cost ~1.8x on `apply()` end to end
        # (EXPERIMENTS.md §X; `walk_tree`'s docstring for the mechanism).
        @njit(cache=True, inline="always")
        def path_fn(args, params):
            thresholds = _nonempty(params + literals_t)
            return walk_tree(
                args, thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc,
            )
        return path_fn

    computed_t = tuple(computed)

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
        thresholds = _nonempty(params + literals_t)
        feats = np.empty(n_args + n_computed, dtype=np.float64)
        for k in range(n_args):
            feats[k] = args[k]
        j = n_args
        for cf in literal_unroll(computed_t):
            feats[j] = cf(args, thresholds)
            j += 1
        return walk_tree(feats, thresholds, kind, feat_idx, op, thr_slot, then_, else_, leaf_value, start_pc)
    return path_fn


def _build_matcher_fn(n_literals: int):
    """`(args, params) -> int`: which of `params`' `n_literals` string-
    literal codes `args[0]` (the column's own int32 dictionary code)
    equals, or -1. `args`/`params` are both homogeneous int32-valued
    tuples, so a plain runtime loop over `params` — the same "array
    addressed by plain index" shape `walk_tree` itself uses — needs no
    per-literal-count variant the way the arity-bounded families above do.
    """

    @njit(cache=True)
    def matcher_fn(args, params):
        code = args[0]
        for i in range(len(params)):
            if params[i] == code:
                return i
        return -1

    return matcher_fn


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


def encode_tree(tree: Tree, *, name: str | None = None) -> EncodedTree:
    """Walk one tree document and build its `Step`s directly.

    Holds, as `Step`s:

    * one string-matcher step per string feature the tree tests (hoisted,
      see the module docstring);
    * `<name>_path` — the path step: packs its own `args`/`params` into the
      `feats`/`thresholds` tuples `walk_tree` needs and makes ONE call into
      it, returning the reached leaf's `result_idx` as an `int`. **This is
      path capture** (doc 03 §7's `<Name>_path` convention, int64, "a value
      the node produces, and you emit it the way you emit any other");
    * one step per numeric or boolean output column, mapping the reached
      `result_idx` to that column's value via a captured array lookup.

    String-valued output columns are not built as steps: a kernel writes
    `float64`/`int64`/`bool` arrays (`boundary.writeback.KernelOutputs`), so
    a string column has nowhere to land. `decider2.trees.build.TreeModule.
    decode()` maps `<name>_path` back to them in polars afterwards.
    """
    name = safe_ident(name or tree.name or "tree")
    ctx = EncodeContext(tree, name)

    # Ordinary Python recursion, unbounded by CPython's own indentation
    # limit (nothing here emits source text for it to apply to) — only by
    # CPython's *recursion* ceiling, raised for the duration of this call.
    import sys

    _needed = len(tree.nodes) * 4 + 200
    _old_limit = sys.getrecursionlimit()
    try:
        if _needed > _old_limit:
            sys.setrecursionlimit(_needed)
        start_pc = ctx.walk()
    finally:
        sys.setrecursionlimit(_old_limit)

    arrays = ctx.resolve_arrays()
    matchers = [ctx.matchers[f] for f in ctx.features if f in ctx.matchers]
    string_features = tuple(m.feature for m in matchers)

    # -- matcher steps ------------------------------------------------------
    matcher_steps: list[Step] = []
    for matcher in matchers:
        fn = _build_matcher_fn(len(matcher.literals))
        inputs = (Input(name=matcher.feature, annotation=str, null_policy=NullPolicy.REQUIRED),)
        params = tuple(
            ParamDecl(name=safe_ident(k), annotation=str, default=default, field_info=Field(default))
            for k, default in matcher.literals
        )
        matcher_steps.append(
            Step(
                name=matcher.fn_name, fn=fn, inputs=inputs, params=params,
                doc=f"Which pattern `{matcher.feature}` matches, or -1.",
                packed=True, output_annotation=int,
            )
        )

    # -- path step ------------------------------------------------------
    path_fn = _build_path_fn(arrays, start_pc, len(ctx.features), ctx._computed, ctx._literals)
    path_inputs: list[Input] = []
    for feature in ctx.features:
        if feature in ctx.matchers:
            path_inputs.append(
                # `annotation=float`, not `int`: the matcher step's OWN
                # output really is an int, but `feats` (walk_tree's own
                # homogeneous array) needs every entry AS a float64 — the
                # same cast the previous, text-generating pass spelled
                # inline as `float(matcher_fn_name)` (`_feature_expr`).
                # `decider2.compile.driver._packed_input_arrays` reads this
                # annotation to cast when gathering a column.
                Input(name=ctx.matchers[feature].fn_name, annotation=float, null_policy=NullPolicy.REQUIRED)
            )
        else:
            path_inputs.append(Input(name=feature, annotation=float, null_policy=NullPolicy.REQUIRED))
    path_params = tuple(
        ParamDecl(name=slot.name, annotation=float, default=slot.default, field_info=Field(slot.default))
        for slot in ctx.params
        if slot.annotation == "float"
    )
    path_name = f"{name}_path"
    path_step = Step(
        name=path_name, fn=path_fn, inputs=tuple(path_inputs), params=path_params,
        doc=f"Which leaf {tree.name!r} reached, as its result_idx.",
        packed=True, output_annotation=int,
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
        matcher_steps=tuple(matcher_steps),
        output_steps=tuple(output_steps),
        path_column=path_name,
        features=tuple(ctx.features),
        string_features=string_features,
        params=tuple(ctx.params),
        leaf_count=ctx.leaf_count,
        max_depth=ctx.max_depth,
        arrays=arrays,
    )
