"""Tree document -> a tiny per-tree wrapper over the generic walker (doc 05
§4, doc 08 §3.4).

**A tree is no longer codegen.** Doc 08 §3.4's own test — "can one compiled
loop evaluate every instance of this kind, with the instance supplied as
arrays? If yes, generic kernel. If it needs a `switch` over node types,
codegen." — used to be answered "codegen" for a tree, because nothing walked
an arbitrary tree generically. `decider2.trees.interpreter.walk_tree` is
exactly that generic loop (EXPERIMENTS.md §Q; its correction to §W explains
why an inline array walk, not a per-node function-pointer call, is the right
shape for a tree specifically — a tree node is a comparison, not a
computation). So a tree now satisfies the "yes" branch too: its shape is
DATA (the flat `kind`/`feat_idx`/`op`/`thr_slot`/`then`/`else`/`leaf_value`
arrays `EncodeContext` below builds), not a nested-`if`/`elif` switch this
module used to emit as source text.

**The switch that is left lives on the classes it used to switch over, not
in this module.** Every node and condition class in `trees/schema.py`
implements its own `encode(ctx, ...)` — its row(s) of the flat arrays —
recursing into its children through `EncodeContext`, the same pattern
decider 1's own `TypeDiscriminatedBaseModule` uses for a decision table's
`Expression.__call__` (`decider/modules/credit/decision_table/config.py`).
This module holds `EncodeContext` (the cross-cutting bookkeeping a class
cannot own by itself: naming and de-duping kernel arguments, hoisting a
string test, walking to a node's children, counting leaves/depth for
`.explain()`) plus the module-level scaffolding that renders one small
wrapper function around one call to `tree.root's .encode(ctx, ...)`. It
imports none of the node or condition classes: nothing here dispatches on
which one it got.

**What is still, deliberately, source text — and why.** Two things, both
narrow and neither a `switch` over node kinds:

1. A computed feature's arithmetic (`decider2.expr`) — a VALUE computation,
   not a branch, already compiled to a numba expression once, at build
   time (doc 08 §1.2/§3.2). `EncodeContext.computed_feature_index` folds its
   result into one wrapper-local variable and gives it a feature slot like
   any other.
2. A hoisted string matcher's own tiny `if`/`elif` (`_StringMatcher`,
   unchanged from before this migration) — bounded by how many DISTINCT
   patterns ONE string test names, never by tree size, and not the source of
   the sibling-collision defect class this migration exists to retire (that
   bug lived in naming a *comparison*, and a matcher's patterns are compared
   by plain equality with no generated identifier in the mix). The tree's
   OWN use of a matcher's result — "which pattern (if any) matched" — is
   still array-encoded: an ordinary `CMP` chain against literal pattern
   indices, exactly like every other condition.

Three properties this module exists to hold, carried over from before this
migration and, in the first case, now considerably stronger:

**1. No decision-relevant constant reaches emitted source.** Doc 05 §4.2,
verbatim: "No decision-relevant constant is emitted into driver source."
Every threshold — literal or `InputRef` — becomes a `param()` in the
generated wrapper's signature, which makes it a kernel argument (doc 05
§4.2, EXPERIMENTS.md §L). A literal is an argument holding that default; an
`InputRef` is an argument named by its key. Retuning either is a values
change: free, no recompile (doc 08 §2).

**2. The emitted-line cap doesn't apply to a tree's SHAPE any more, because
shape is no longer text.** Doc 05 §7's ~500-line cap existed because compile
time was super-linear in emitted lines (EXPERIMENTS.md §G). The wrapper this
module renders is a handful of lines regardless of how many nodes the tree
has — the arrays that scale with node count are DATA (`np.array([...])`
literals numba never has to compile branching logic for), and this
migration's own report measures the resulting compile-time win directly.
There is no fan-out wall, no CPython 100-level indentation limit, and no
depth cap: `TreeTooLarge` survives only for `TableTooComplex`'s unrelated
DNF-explosion guard (`tables/schema.py`) — nothing in THIS module raises it
any more.

**3. A string never enters the walker as a string.** Doc 05 §1.5. Every
string comparison is *hoisted out of the tree* into its own one-input step
(`EncodeContext.encode_string_match`), which takes that column as `str` (an
int32 dictionary code at runtime) plus one `str` `param()` per literal, and
returns an `int`: which pattern matched, or -1. The tree's own arrays then
compare that int against a literal pattern index. Two reasons it has to be a
separate step rather than inline: `runtime.invoke._resolve_str_param_code`
resolves a `str` param against *the one* `str` input its step reads and
raises on a step with several (pinned by `test_string_params.py`), and a
step reading a `str` input while declaring no `str` param is rejected
outright — the guard against doc 05 §1.5's silent-`False` hazard.
"""
from __future__ import annotations

import keyword
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

from decider2.trees.interpreter import EQ
from decider2.trees.schema import InputRef, Tree, TStringMatchType

__all__ = [
    "TreeTooLarge",
    "UnsupportedInKernel",
    "EmittedTree",
    "EncodeContext",
    "emit_tree",
    "LINE_CAP",
    "safe_ident",
]

# Kept only so `tables/schema.py`'s `TableTooComplex(TreeTooLarge)` and its
# message (an unrelated DNF-explosion guard, never a "lines" measurement any
# more either — see that module) still import. Nothing in this module raises
# either any more: a tree's emitted wrapper is a handful of lines regardless
# of node count, so there is no cap for it to hit.
LINE_CAP = 500

_IDENT_BAD = re.compile(r"[^0-9A-Za-z_]")


class TreeTooLarge(ValueError):
    """No longer raised by this module — a tree's wrapper is a handful of
    lines regardless of node count, so there is no line cap, no fan-out
    wall and no CPython indentation limit left to hit. Kept as a base class
    for `tables.schema.TableTooComplex`, whose own guard (bounding And/Or
    disjunct explosion) is a genuinely different, still-relevant concern —
    see that module."""


class UnsupportedInKernel(ValueError):
    """A decider 1 construct with no compiled-kernel form.

    Raised rather than silently approximated. Every instance names what was
    asked for, why a kernel cannot express it, and the route that works.
    """


def safe_ident(name: str) -> str:
    """A valid Python identifier, deterministically derived from `name`.

    Same job as `decider2.compile.codegen.safe_ident`; reimplemented rather
    than imported because that module is one this migration was told to add
    to but never restructure, and a tree's names (node ids, feature names
    from a UI) reach here in shapes a step name never does.
    """
    ident = _IDENT_BAD.sub("_", name)
    if not ident or ident[0].isdigit():
        ident = f"v_{ident}"
    if keyword.iskeyword(ident):
        ident = f"{ident}_"
    return ident


@dataclass
class _ParamSlot:
    """One threshold, on its way to becoming a kernel argument."""

    name: str
    default: Any
    annotation: str  # "float" | "str"
    origin: str      # human-readable: which node and which bound


@dataclass
class EmittedTree:
    """Everything `decider2.trees.build` needs to turn the wrapper into a
    Module."""

    source: str
    module_docstring: str
    path_fn_name: str
    matcher_fn_names: tuple[str, ...]
    output_fn_names: tuple[str, ...]
    features: tuple[str, ...]
    string_features: tuple[str, ...]
    params: tuple[_ParamSlot, ...]
    emitted_lines: int
    leaf_count: int
    max_depth: int


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
    One instance per `emit_tree` call, and not reusable across trees:
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
        # computed feature -> (wrapper-local variable name, numba expr text)
        self._computed: list[tuple[str, str]] = []

        # -- threshold space: every registered param (float AND str; str
        # entries never enter the `thresholds` tuple, but stay in `params`
        # for `.explain()`'s reporting, matching the pre-migration shape) --
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
        and return its identifier. Order of first use is the wrapper
        function's `feats` tuple order — doc 05 §4.2's determinism depends
        on callers never re-ordering this themselves.

        This is also what a computed feature's own free names resolve
        through (`_ExprEmitAdapter.name`, `decider2.trees.schema`) — a plain
        column `x` and the same `x` read inside `x - y` share the one slot,
        because both paths end up calling this exact method with the exact
        same string.
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
        `computed.emit(...)` is `decider2.expr`'s own compile-to-numba-source
        step (doc 08 §1.2/§3.2); nothing here re-implements it. `_ComputedFeature.
        emit` (schema.py) builds its own `_ExprEmitAdapter(ctx, node_id)`
        internally, so this just hands itself across."""
        expr_text = computed.emit(self, node_id)
        k = len(self._computed)
        local_name = f"_cf{k}"
        self._computed.append((local_name, expr_text))
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

    def threshold(self, value: Any, *, node_id: str, role: str) -> str:
        """The wrapper argument NAME for a threshold — used only where that
        name is referenced directly as source text (a computed feature's own
        inline constant, `_ExprEmitAdapter.constant`). A `CMP` node never
        calls this — see `threshold_slot`, which returns the array SLOT
        instead.

        Both arms of decider 1's `Union[float, InputRef]` register a param
        here: neither a literal nor a reference ever reaches emitted source
        (doc 05 §4.2). `isinstance` here is the one the task's own
        instructions call out as fine to keep: `Threshold` is "a literal, or
        a reference to one" — a two-case value, not a family of classes with
        their own behaviour to dispatch across.
        """
        if isinstance(value, InputRef):
            slot = self._add_param(value.key, 0.0, "float", f"InputRef #{value.key} ({node_id}.{role})")
        else:
            slot = self._add_param(f"{node_id}_{role}", float(value), "float", f"{node_id}.{role}")
        return slot.name

    def threshold_slot(self, value: Any, *, node_id: str, role: str) -> int:
        """The walker's `thresholds` tuple slot for one threshold — a `CMP`
        node's `thr_slot` field. Stable the instant it is first assigned:
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
        """An anonymous threshold-space entry — never a `param()`, never a
        named argument, just a plain literal embedded in the `thresholds`
        tuple-building line (a string match's "which pattern index" test
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
    Unchanged by this migration — see the module docstring for why."""

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

    def emit(self) -> list[str]:
        args = ", ".join(
            [f"{self.feature_ident}: str"]
            + [f"{safe_ident(k)}: str = param({d!r})" for k, d in self.literals]
        )
        lines = [
            f"def {self.fn_name}({args}) -> int:",
            f'    """Which pattern `{self.feature}` matches, or -1.',
            "",
            "    Hoisted out of the tree body so this step reads exactly one",
            "    `str` input (doc 05 §1.5; runtime.invoke._resolve_str_param_code",
            "    resolves each `str` param against it). The comparison is",
            "    between two int32 dictionary codes at runtime.",
            '    """',
        ]
        for i, (k, _) in enumerate(self.literals):
            branch = "if" if i == 0 else "elif"
            lines.append(f"    {branch} {self.feature_ident} == {safe_ident(k)}:")
            lines.append(f"        return {i}")
        lines.append("    return -1")
        return lines

    @property
    def feature_ident(self) -> str:
        return safe_ident(self.feature)


def _render_value(slot: _ParamSlot) -> str:
    return repr(slot.default)


def _tuple_expr(items: list[str]) -> str:
    """A numba-safe homogeneous-tuple literal: never truly empty (`()` has
    no element type numba can infer for a dynamic-index read that is never
    actually reached — e.g. a single-leaf tree's `walk_tree` call still has
    to TYPE-CHECK its unreachable `feats[feat_idx[pc]]`), so an empty tuple
    gets one inert `0.0` sentinel no node ever indexes into."""
    if not items:
        return "(0.0,)"
    if len(items) == 1:
        return f"({items[0]},)"
    return "(" + ", ".join(items) + ")"


def _feature_expr(ctx: EncodeContext, name: str) -> str:
    """The wrapper-body token for one plain feature's value, at `feats`
    tuple-building time: the hoisted matcher's int result, cast to float64
    (a small integer's exact float64 representation, so `==` against a
    literal pattern index stays exact), or the plain float argument as-is.
    """
    if name in ctx.matchers:
        return f"float({ctx.matchers[name].fn_name})"
    return safe_ident(name)


def emit_tree(tree: Tree, *, name: str | None = None) -> EmittedTree:
    """Render one tree as an importable wrapper around the generic walker.

    The emitted module holds, in this order:

    * one `__match_<feature>` step per string feature the tree tests
      (hoisted, see the module docstring);
    * this tree's flat node arrays (`kind`/`feat_idx`/`op`/`thr_slot`/
      `then`/`else`/`leaf_value`) as module-level numpy-array constants —
      DATA, addressed by plain array index, never a generated identifier
      (see `decider2.trees.interpreter`'s module docstring for why that
      retires the sibling-collision defect class this migration exists to
      fix, and this stage's report for the numba-disk-cache probe
      confirming a plain array global like these caches cleanly, unlike a
      `ctypes`/`cfunc` pointer global);
    * `<name>_path` — the wrapper itself: builds the `feats`/`thresholds`
      tuples from its own named arguments (never a module global) and makes
      ONE call into `decider2.trees.interpreter.walk_tree`, returning the
      reached leaf's `result_idx` as an `int`. **This is path capture**
      (doc 03 §7's `<Name>_path` convention, int64, "a value the node
      produces, and you emit it the way you emit any other");
    * one step per numeric or boolean output column, mapping the reached
      `result_idx` to that column's value via a module-level array lookup —
      the same "rows are data" shape a decision table's own output columns
      already use (`tables.codegen`), extended here from a table's rows to
      a tree's leaves.

    String-valued output columns are not emitted as steps: a kernel writes
    `float64`/`int64`/`bool` arrays (`boundary.writeback.KernelOutputs`), so
    a string column has nowhere to land. `decider2.trees.build.TreeModule.
    decode()` maps `<name>_path` back to them in polars afterwards, which is
    where string work belongs anyway (doc 05 §1.5, EXPERIMENTS.md §O).
    """
    name = safe_ident(name or tree.name or "tree")
    ctx = EncodeContext(tree, name)

    # The encode walk is ordinary Python recursion, one call per tree LEVEL
    # (never per node overall — an otherwise-chain of length L still costs
    # L stack frames, same as a then-chain of length L, because nothing
    # about doc 05 §7's old CPython-source-indentation problem applies to a
    # data walk). Doc 08 §3.4's whole point is that a tree this size should
    # build at all, so the limit that matters here is CPython's *recursion*
    # ceiling, not its indentation one — raised for the duration of this one
    # call, restored immediately after, rather than left touched globally.
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

    lines: list[str] = []
    lines.append(f'"""Generated tree kernel for {tree.name!r}.')
    lines.append("")
    lines.append("Content-addressed and written to a real file by")
    lines.append("decider2.compile.cache (doc 05 §4.1: numba cannot cache a function")
    lines.append("with no source file). Do not hand-edit — regenerate from the tree")
    lines.append("document instead.")
    lines.append("")
    lines.append("This tree's SHAPE lives in the plain arrays below, not in nested")
    lines.append("`if`/`elif` source (doc 08 §3.4): a structural edit changes their")
    lines.append("CONTENTS, so it recompiles only in the sense that this file's own")
    lines.append("content hash changes (decider2.compile.cache) — there is no per-node")
    lines.append("branching logic here for compile time to scale with any more.")
    lines.append("")
    lines.append("Every threshold below is a function ARGUMENT with a default, never")
    lines.append("a literal in this file (doc 05 §4.2, doc 08 §2): retuning one is a")
    lines.append("value change and does not bring you back here.")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from decider2.params import param")
    lines.append("from decider2.trees.interpreter import walk_tree")
    lines.append("")

    for matcher in matchers:
        lines.append("")
        lines += matcher.emit()
        lines.append("")

    prefix = f"_{name}"
    lines.append("")
    lines.append(f"{prefix}__kind = np.array({arrays['kind']!r}, dtype=np.int32)")
    lines.append(f"{prefix}__feat_idx = np.array({arrays['feat_idx']!r}, dtype=np.int32)")
    lines.append(f"{prefix}__op = np.array({arrays['op']!r}, dtype=np.int32)")
    lines.append(f"{prefix}__thr_slot = np.array({arrays['thr_slot']!r}, dtype=np.int32)")
    lines.append(f"{prefix}__then = np.array({arrays['then']!r}, dtype=np.int32)")
    lines.append(f"{prefix}__else = np.array({arrays['else']!r}, dtype=np.int32)")
    lines.append(f"{prefix}__leaf_value = np.array({arrays['leaf_value']!r}, dtype=np.int64)")
    lines.append(f"{prefix}__start_pc = {start_pc}")

    # The traversal function's signature: features first (string features
    # enter as their hoisted matcher's int result), then every float
    # threshold (a str-typed param belongs to a hoisted matcher, not here).
    sig_parts: list[str] = []
    for feature in ctx.features:
        ident = safe_ident(feature)
        if feature in ctx.matchers:
            sig_parts.append(f"{ctx.matchers[feature].fn_name}: int")
        else:
            sig_parts.append(f"{ident}: float")
    for slot in ctx.params:
        if slot.annotation == "str":
            continue  # belongs to a hoisted matcher, not the traversal
        sig_parts.append(f"{slot.name}: float = param({_render_value(slot)})")

    body: list[str] = []
    for local_name, expr_text in ctx._computed:
        # `float(...)`, unconditionally: `decider2.expr` admits boolean-
        # valued expressions too (a comparison, "x - y > 10", used directly
        # via `is_true` — EXPERIMENTS.md's own worked example), and every
        # entry of `feats` must be the SAME numba type for `walk_tree`'s
        # homogeneous tuple (`decider2.trees.interpreter`) — exactly the
        # cast `UnaryIsTrue`/`UnaryIsFalse`'s own `!= 0.0`/`== 0.0` opcodes
        # already expect a plain feature column to have gone through. A
        # no-op for an already-numeric expression.
        body.append(f"    {local_name} = float({expr_text})")

    feats_items = [_feature_expr(ctx, f) for f in ctx.features] + [ln for ln, _ in ctx._computed]
    thresholds_items = list(ctx._threshold_position) + [repr(v) for v in ctx._literals]
    feats_expr = _tuple_expr(feats_items)
    thresholds_expr = _tuple_expr(thresholds_items)
    body.append(f"    feats = {feats_expr}")
    body.append(f"    thresholds = {thresholds_expr}")
    body.append(
        f"    return walk_tree(feats, thresholds, {prefix}__kind, {prefix}__feat_idx, "
        f"{prefix}__op, {prefix}__thr_slot, {prefix}__then, {prefix}__else, "
        f"{prefix}__leaf_value, {prefix}__start_pc)"
    )

    path_fn = f"{name}_path"
    lines.append("")
    lines.append("")
    lines.append(f"def {path_fn}({', '.join(sig_parts)}) -> int:")
    lines.append(f'    """Which leaf `{tree.name}` reached, as its result_idx.')
    lines.append("")
    lines.append("    -1 is the default row (decider 1's LeafNode sentinel, kept).")
    lines.append('    """')
    lines += body
    lines.append("")

    # One step per numeric/boolean output column — its values are an ARRAY
    # (doc 08 §3.4's "free interior", extended from a table's rows to a
    # tree's leaves by this migration: editing an output value used to be
    # the one place a document value was deliberately emitted as source
    # (`_literal`, "what a leaf returns is the tree's shape"); it no longer
    # has to be, now that the tree's own branching is data too).
    output_fns: list[str] = []
    for column, dtype in tree.output.dtypes:
        coerce = _COERCE_BY_DTYPE.get(dtype)
        if coerce is None:
            continue
        py_type, np_dtype, to_value = coerce
        fn_name = safe_ident(column)
        rows = list(tree.output.data)
        default_row = tree.output.default or {}
        values = [to_value(row.get(column)) for row in rows]
        default_value = to_value(default_row.get(column))

        values_name = f"{prefix}__out_{fn_name}"
        lines.append("")
        lines.append(f"{values_name} = np.array({values!r}, dtype=np.{np_dtype})")
        lines.append(f"{values_name}_default = {default_value!r}")
        lines.append("")
        lines.append(f"def {fn_name}({path_fn}: int) -> {py_type}:")
        lines.append(f'    """`{column}` for the leaf the tree reached, read from an array')
        lines.append("    (doc 08 §3.4): editing this column is free, no recompile.\"\"\"")
        lines.append(f"    if {path_fn} < 0:")
        lines.append(f"        return {values_name}_default")
        lines.append(f"    return {values_name}[{path_fn}]")
        lines.append("")
        output_fns.append(fn_name)

    source = "\n".join(lines) + "\n"
    emitted = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith(("#", '"""')))

    return EmittedTree(
        source=source,
        module_docstring=f"tree {tree.name!r}",
        path_fn_name=path_fn,
        matcher_fn_names=tuple(m.fn_name for m in matchers),
        output_fn_names=tuple(output_fns),
        features=tuple(ctx.features),
        string_features=string_features,
        params=tuple(ctx.params),
        emitted_lines=emitted,
        leaf_count=ctx.leaf_count,
        max_depth=ctx.max_depth,
    )


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
