"""Tree document -> numba-compilable kernel source (doc 05 §4).

A tree is **codegen**, not a generic kernel — doc 08 §3.4's test decides it:
"can one compiled loop evaluate every instance of this kind, with the
instance supplied as arrays? If yes, generic kernel. If it needs a `switch`
over node types, codegen." A tree needs the switch, so its shape is emitted
as nested `if`/`elif` and an interior *shape* change recompiles (doc 08 §2,
class "interiors — shape"). Its thresholds do not: they are arguments.

Three properties this module exists to hold:

**1. No decision-relevant constant reaches emitted source.** Doc 05 §4.2,
verbatim: "No decision-relevant constant is emitted into driver source."
Every threshold — literal or `InputRef` — becomes a `param()` in the
generated function's signature, which makes it a kernel argument (doc 05
§4.2, EXPERIMENTS.md §L). A literal is an argument holding that default; an
`InputRef` is an argument named by its key. Retuning either is a values
change: free, no recompile (doc 08 §2). `tests/test_trees.py::
test_retuning_a_threshold_never_recompiles` pins `len(driver.signatures)`
across retunes, the same way `test_compile_driver.py` and
`test_string_params.py` pin it for plain steps.

**2. The emitted-line cap is enforced, not hoped for.** Doc 05 §7 requires
"a hard cap on emitted lines (~500) per kernel, enforced as a build error
naming the group", because compile time is super-linear in emitted lines
(EXPERIMENTS.md §G: ∝ lines^1.4, local exponent 1.96 between 60 and 100
rules; doc 01 §4b: a depth-10 fully-branching nest costs 92 s). `emit_tree`
raises `TreeTooLarge` naming the tree, its emitted line count and the
`arms^depth` fan-out that caused it. **Fan-out is the wall, not depth** —
doc 01 §4b measured 32 one-sided levels at 1.9 s against depth-10
full-binary at 92 s — so the guard counts lines, never depth.

**3. A string never enters the kernel as a string.** Doc 05 §1.5. Every
string comparison is *hoisted out of the tree* into its own one-input step
(`_emit_string_matchers`), which takes that column as `str` (an int32
dictionary code at runtime) plus one `str` `param()` per literal, and
returns an `int`: which pattern matched, or -1. The tree body then reads
that int. Two reasons it has to be a separate step rather than inline:
`runtime.invoke._resolve_str_param_code` resolves a `str` param against
*the one* `str` input its step reads and raises on a step with several
(pinned by `test_string_params.py::
test_a_step_reading_two_str_inputs_is_a_clear_error_not_a_guess`), and a
step reading a `str` input while declaring no `str` param is rejected
outright — which is the guard against doc 05 §1.5's silent-`False` hazard
("`int32 == 'private'` compiles cleanly in nopython and evaluates to
`False` forever").
"""
from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from decider2.trees.schema import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    CompositeCondition,
    CompositeNode,
    Feature,
    InputRef,
    IsInCondition,
    LeafNode,
    RangeCondition,
    RangeEndLogic,
    StringMatchCondition,
    TLogicOp,
    Tree,
    TStringMatchType,
    UnaryBetween,
    UnaryIsIn,
    UnaryIsFalse,
    UnaryIsTrue,
    UnaryNode,
    UnaryStringMatch,
)

__all__ = [
    "TreeTooLarge",
    "UnsupportedInKernel",
    "EmittedTree",
    "emit_tree",
    "LINE_CAP",
    "safe_ident",
]

# Doc 05 §7: "A hard cap on emitted lines (~500) per kernel, enforced as a
# build error naming the group." EXPERIMENTS.md §G measured 517 emitted
# lines at 5.96 s and 642 at 10.86 s, and notes the guardrail is "slightly
# conservative; the real crossover is 600-710 lines". 500 is kept rather
# than moved to 600 because §G's own headline is the ten-second promise, and
# 500 is the number doc 05 §7 states.
LINE_CAP = 500

# CPython's own hard limit is 100 levels of indentation; 90 leaves room for
# the function body's own base indent and for the output steps emitted
# alongside. Found by measurement, not by reading: a one-sided chain of 128
# nodes hit `IndentationError: too many levels of indentation` at import of
# the generated file (see `_emit_node`'s flattening note).
_MAX_NESTING = 90

_IDENT_BAD = re.compile(r"[^0-9A-Za-z_]")


class TreeTooLarge(ValueError):
    """The tree's emitted source exceeds `LINE_CAP` (doc 05 §7)."""


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
    """Everything `decider2.trees.build` needs to turn source into a Module."""

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


class _Emitter:
    """Builds one tree's source. One instance per `emit_tree` call.

    Not reusable and not shared: it accumulates param slots and matcher
    definitions as it walks, and determinism (doc 05 §4.2) depends on that
    walk being in one fixed order — node order for matchers, traversal order
    for params. Nothing here iterates a set.
    """

    def __init__(self, tree: Tree, name: str) -> None:
        self.tree = tree
        self.name = safe_ident(name)
        self.params: list[_ParamSlot] = []
        self._param_names: set[str] = set()
        self.features: list[str] = []
        self._feature_set: set[str] = set()
        # feature -> (step name, [literal param names]) for hoisted string tests
        self.matchers: dict[str, _StringMatcher] = {}
        self.max_depth = 0
        self.leaf_count = 0

    # -- names ------------------------------------------------------------

    def _use_feature(self, feature: Feature) -> str:
        fname = str(feature)
        if fname not in self._feature_set:
            self._feature_set.add(fname)
            self.features.append(fname)
        return safe_ident(fname)

    def _add_param(self, name: str, default: Any, annotation: str, origin: str) -> str:
        """Register a kernel argument, de-duplicating by name.

        An `InputRef` deliberately collides with itself: two nodes naming
        `#income_floor` share one knob, which is the whole point of a named
        reference (decider 1's `common/parameters.py`). A generated literal
        name never collides, because it carries the node's own id.
        """
        ident = safe_ident(name)
        if ident in self._param_names:
            return ident
        self._param_names.add(ident)
        self.params.append(
            _ParamSlot(name=ident, default=default, annotation=annotation, origin=origin)
        )
        return ident

    def _threshold(self, value: Any, *, node_id: str, role: str) -> str:
        """The expression text for a threshold — always an argument name.

        This is the single place the migration's central claim is made
        concrete: **both arms of decider 1's `Union[float, InputRef]` return
        a parameter name here.** Neither ever returns a literal, so no
        threshold can reach emitted source.
        """
        if isinstance(value, InputRef):
            return self._add_param(
                value.key, 0.0, "float", f"InputRef #{value.key} ({node_id}.{role})"
            )
        return self._add_param(
            f"{node_id}_{role}", float(value), "float", f"{node_id}.{role}"
        )

    # -- conditions -------------------------------------------------------

    def _range_test(
        self,
        var: str,
        cond: RangeCondition,
        end_logic: RangeEndLogic,
        node_id: str,
        idx: int,
    ) -> str:
        """decider 1's `RangeCondition.build_range_condition`, as source.

        lower_inclusive -> `min <= x <  max`
        upper_inclusive -> `min <  x <= max`
        """
        lo_op, hi_op = (">=", "<") if end_logic is RangeEndLogic.lower_inclusive else (">", "<=")
        parts = []
        if cond.min is not None:
            parts.append(f"{var} {lo_op} {self._threshold(cond.min, node_id=node_id, role=f'min_{idx}')}")
        if cond.max is not None:
            parts.append(f"{var} {hi_op} {self._threshold(cond.max, node_id=node_id, role=f'max_{idx}')}")
        return " and ".join(parts) if len(parts) > 1 else parts[0]

    def _isin_test(self, var: str, values: Any, node_id: str, idx: int) -> str:
        """An OR of equalities — each value its own argument.

        decider 1 also allows `values` to be a bare `InputRef`, which it
        treats as *equality against that one parameter*
        (`UnaryIsIn.build_condition`: `feature_expr == self.values.resolve(...)`,
        not `is_in`). That exact behaviour is reproduced.
        """
        if isinstance(values, InputRef):
            return f"{var} == {self._threshold(values, node_id=node_id, role=f'isin_{idx}')}"
        tests = [
            f"{var} == {self._threshold(v, node_id=node_id, role=f'isin_{idx}_{j}')}"
            for j, v in enumerate(values)
        ]
        return "(" + " or ".join(tests) + ")" if len(tests) > 1 else tests[0]

    def _string_test(
        self,
        feature: Feature,
        patterns: Sequence[Any],
        match_type: TStringMatchType,
        case_sensitive: bool,
        trim_whitespace: bool,
        node_id: str,
    ) -> str:
        """A string comparison, hoisted into its own step.

        Returns source that tests the hoisted step's int result, never the
        string column itself — see the module docstring for why this cannot
        be inlined.
        """
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
        self._use_feature(feature)
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
        var = matcher.fn_name
        tests = [f"{var} == {i}" for i in slots]
        return "(" + " or ".join(tests) + ")" if len(tests) > 1 else tests[0]

    def _unary_test(self, op: Any, node_id: str) -> str:
        """One `TUnaryOp` as a boolean source expression."""
        if isinstance(op, UnaryStringMatch):
            return self._string_test(
                op.feature, op.patterns, op.match_type, op.case_sensitive,
                op.trim_whitespace, node_id,
            )
        var = self._use_feature(op.feature)
        if isinstance(op, UnaryIsTrue):
            return f"{var} != 0"
        if isinstance(op, UnaryIsFalse):
            return f"{var} == 0"
        if isinstance(op, UnaryBetween):
            # decider 1's UnaryBetween is inclusive at BOTH ends — deliberately
            # unlike RangeCondition, which follows the node's end_logic.
            parts = []
            if op.min is not None:
                parts.append(f"{var} >= {self._threshold(op.min, node_id=node_id, role='min')}")
            if op.max is not None:
                parts.append(f"{var} <= {self._threshold(op.max, node_id=node_id, role='max')}")
            return " and ".join(parts) if len(parts) > 1 else parts[0]
        if isinstance(op, UnaryIsIn):
            return self._isin_test(var, op.values, node_id, 0)
        # The six primitive comparisons share one shape.
        return f"{var} {op.op} {self._threshold(op.threshold, node_id=node_id, role='thr')}"

    def _condition_test(self, cond: Any, node_id: str) -> str:
        if isinstance(cond, CompositeCondition):
            inner = [self._condition_test(c, node_id) for c in cond.conditions]
            if cond.op is TLogicOp.NOT:
                return f"(not ({inner[0]}))"
            joiner = " and " if cond.op is TLogicOp.AND else " or "
            return "(" + joiner.join(inner) + ")"
        return self._unary_test(cond, node_id)

    # -- the tree walk ----------------------------------------------------

    def walk(self) -> list[str]:
        node_map = self.tree.node_map()
        children = self.tree.children()
        root = self.tree.root_id()
        self._seen: set[str] = set()
        return self._emit_node(root, node_map, children, depth=1)

    def _child_lines(
        self,
        parent_id: str,
        index: int,
        node_map: dict,
        children: dict,
        depth: int,
    ) -> list[str]:
        """The body under one branch.

        An unconnected branch is decider 1's `LeafRule(result_idx=-1)` —
        `v3/tree.to_flat_rule_tree`'s `get_child`, same default.
        """
        target = children.get(parent_id, {}).get(index)
        if target is None:
            self.leaf_count += 1
            self.max_depth = max(self.max_depth, depth)
            return ["return -1"]
        return self._emit_node(target, node_map, children, depth)

    def _emit_node(
        self, node_id: str, node_map: dict, children: dict, depth: int
    ) -> list[str]:
        if node_id in self._seen:
            raise ValueError(
                f"tree '{self.tree.name}' revisits node '{node_id}' — a tree must "
                "be acyclic and each node reachable once. (A shared subtree would "
                "duplicate its emitted lines, which doc 05 §7's line cap counts.)"
            )
        self._seen.add(node_id)
        self.max_depth = max(self.max_depth, depth)
        node = node_map[node_id]
        data = node.data

        if isinstance(data, LeafNode):
            self.leaf_count += 1
            return [f"return {data.result_idx}"]

        if isinstance(data, (UnaryNode, CompositeNode)):
            if isinstance(data, UnaryNode):
                test = self._unary_test(data.condition, node_id)
            else:
                inner = [self._condition_test(c, node_id) for c in data.conditions]
                if data.op is TLogicOp.NOT:
                    test = f"not ({inner[0]})"
                else:
                    joiner = " and " if data.op is TLogicOp.AND else " or "
                    test = joiner.join(f"({i})" for i in inner) if len(inner) > 1 else inner[0]
            then_lines = self._child_lines(node_id, 0, node_map, children, depth + 1)
            else_lines = self._child_lines(node_id, 1, node_map, children, depth)
            # No `else:`. Every path through a tree ends in a `return` (a leaf
            # returns its result_idx; an unconnected branch returns -1), so the
            # otherwise-arm is reachable only when the then-arm did not
            # return, and it can follow at the SAME indentation.
            #
            # This is not cosmetic. CPython refuses more than 100 levels of
            # indentation — `IndentationError: too many levels of
            # indentation`, raised at import of the generated file, before
            # numba ever sees it. With a nested `else:`, a one-sided chain of
            # 128 nodes (a perfectly ordinary policy waterfall, and the shape
            # doc 01 §4b measured as the CHEAP one at 1.9 s for 32 levels)
            # could not be emitted at all. Flattened, an otherwise-chain costs
            # no indentation and only the then-arms nest.
            out = [f"if {test}:"]
            out += [f"    {ln}" for ln in then_lines]
            out += else_lines
            return out

        if isinstance(data, (CasesRanges, CasesStringMatch, CasesIsIn)):
            var = self._use_feature(data.feature)
            tests: list[str] = []
            for i, cond in enumerate(data.conditions):
                if isinstance(cond, RangeCondition):
                    tests.append(self._range_test(var, cond, data.end_logic, node_id, i))
                elif isinstance(cond, IsInCondition):
                    tests.append(self._isin_test(var, cond.values, node_id, i))
                elif isinstance(cond, StringMatchCondition):
                    tests.append(
                        self._string_test(
                            data.feature, cond.patterns, data.match_type,
                            data.case_sensitive, data.trim_whitespace, node_id,
                        )
                    )
                else:  # pragma: no cover - the union is closed
                    raise AssertionError(f"unhandled cases condition {type(cond)!r}")

            out: list[str] = []
            for i, test in enumerate(tests):
                keyword_ = "if" if i == 0 else "elif"
                out.append(f"{keyword_} {test}:")
                out += [f"    {ln}" for ln in self._child_lines(node_id, i, node_map, children, depth + 1)]
            # Same flattening as the unary case: the otherwise-arm follows at
            # this level rather than under an `else:`, because every branch
            # above it returns.
            out += self._child_lines(node_id, len(tests), node_map, children, depth)
            return out

        raise AssertionError(f"unhandled node type {type(data)!r}")  # pragma: no cover


@dataclass
class _StringMatcher:
    """A hoisted string test: one step, one `str` input, N `str` params."""

    feature: str
    fn_name: str
    literals: list[tuple[str, Any]] = field(default_factory=list)

    def literal_slot(self, pattern: Any, emitter: _Emitter) -> int:
        """The index this literal answers with, registering it if new."""
        if isinstance(pattern, InputRef):
            key, default = pattern.key, ""
        else:
            key, default = f"{safe_ident(self.feature)}_pat_{len(self.literals)}", pattern
        for i, (existing, _) in enumerate(self.literals):
            if existing == key:
                return i
        self.literals.append((key, default))
        emitter._add_param(key, default, "str", f"string literal for '{self.feature}'")
        return len(self.literals) - 1

    def emit(self, params_by_name: dict[str, _ParamSlot]) -> list[str]:
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


def emit_tree(tree: Tree, *, name: str | None = None, line_cap: int = LINE_CAP) -> EmittedTree:
    """Render one tree as importable kernel source.

    The emitted module holds, in this order:

    * one `__match_<feature>` step per string feature the tree tests
      (hoisted, see the module docstring);
    * `<name>_path` — the traversal itself, returning the `result_idx` of
      the leaf it reached, as an `int`. **This is path capture** (doc 03
      §7's `<Name>_path` convention, int64, "a value the node produces, and
      you emit it the way you emit any other"). decider 1 already had the
      right identity for this and threw it away at the boundary: a
      `LeafRule` carries `result_idx`, but `FlatRuleModule` immediately
      resolved it to the output row and only the row survived. Keeping the
      index *is* the feature — doc 01 §5.4 records that decider 1's tree
      modules had no diagnostic hook at all;
    * one step per numeric or boolean output column, mapping the reached
      `result_idx` to that column's value.

    String-valued output columns are not emitted as steps: a kernel writes
    `float64`/`int64`/`bool` arrays (`boundary.writeback.KernelOutputs`), so
    a string column has nowhere to land. `decider2.trees.build.TreeModule.
    decode()` maps `<name>_path` back to them in polars afterwards, which is
    where string work belongs anyway (doc 05 §1.5, EXPERIMENTS.md §O).
    """
    name = safe_ident(name or tree.name or "tree")
    emitter = _Emitter(tree, name)
    body = emitter.walk()

    # CPython refuses more than 100 levels of indentation, and raises
    # `IndentationError` at *import* of the generated file — before numba is
    # reached, with a message naming a line number in a generated file rather
    # than anything about the tree. Caught here instead, while the tree is
    # still the thing being talked about. The flattening in `_emit_node`
    # means only nested THEN-arms count toward this, so hitting it takes a
    # genuinely 90-deep chain of nested conditions.
    nesting = max((len(ln) - len(ln.lstrip())) // 4 for ln in body) + 1
    if nesting > _MAX_NESTING:
        raise TreeTooLarge(
            f"tree {tree.name!r} nests {nesting} levels of conditions, over the "
            f"{_MAX_NESTING}-level limit. CPython cannot compile more than 100 "
            "levels of indentation at all, so this tree has no emitted form. "
            "Only nested `then` arms count — an `otherwise` chain is flat — so "
            "this is a chain of ~90 conditions each inside the previous one's "
            "true branch. Split it into two trees composed with `|`."
        )

    params_by_name = {p.name: p for p in emitter.params}
    matchers = [emitter.matchers[f] for f in emitter.features if f in emitter.matchers]
    string_features = tuple(m.feature for m in matchers)

    lines: list[str] = []
    lines.append(f'"""Generated tree kernel for {tree.name!r}.')
    lines.append("")
    lines.append("Content-addressed and written to a real file by")
    lines.append("decider2.compile.cache (doc 05 §4.1: numba cannot cache a function")
    lines.append("with no source file). Do not hand-edit — regenerate from the tree")
    lines.append("document instead.")
    lines.append("")
    lines.append("Every threshold below is a function ARGUMENT with a default, never")
    lines.append("a literal in this file (doc 05 §4.2, doc 08 §2): retuning one is a")
    lines.append("value change and does not bring you back here.")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from decider2.params import param")
    lines.append("")

    for matcher in matchers:
        lines.append("")
        lines += matcher.emit(params_by_name)
        lines.append("")

    # The traversal function's signature: features first (string features
    # enter as their hoisted matcher's int result), then every threshold.
    sig_parts: list[str] = []
    for feature in emitter.features:
        ident = safe_ident(feature)
        if feature in emitter.matchers:
            sig_parts.append(f"{emitter.matchers[feature].fn_name}: int")
        else:
            sig_parts.append(f"{ident}: float")
    for slot in emitter.params:
        if slot.annotation == "str":
            continue  # belongs to a hoisted matcher, not the traversal
        sig_parts.append(f"{slot.name}: float = param({_render_value(slot)})")

    path_fn = f"{name}_path"
    lines.append("")
    lines.append(f"def {path_fn}({', '.join(sig_parts)}) -> int:")
    lines.append(f'    """Which leaf `{tree.name}` reached, as its result_idx.')
    lines.append("")
    lines.append("    -1 is the default row (decider 1's LeafNode sentinel, kept).")
    lines.append('    """')
    lines += [f"    {ln}" for ln in body]
    lines.append("")

    # One step per numeric/boolean output column.
    output_fns: list[str] = []
    for column, dtype in tree.output.dtypes:
        py_type = _PY_TYPE_BY_DTYPE.get(dtype)
        if py_type is None:
            continue
        fn_name = safe_ident(column)
        rows = list(tree.output.data)
        default = tree.output.default or {}
        lines.append("")
        lines.append(f"def {fn_name}({path_fn}: int) -> {py_type}:")
        lines.append(f'    """`{column}` for the leaf the tree reached."""')
        for i, row in enumerate(rows):
            branch = "if" if i == 0 else "elif"
            lines.append(f"    {branch} {path_fn} == {i}:")
            lines.append(f"        return {_literal(row.get(column), py_type)}")
        lines.append(f"    return {_literal(default.get(column), py_type)}")
        lines.append("")
        output_fns.append(fn_name)

    source = "\n".join(lines) + "\n"
    emitted = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith(("#", '"""')))

    if emitted > line_cap:
        raise TreeTooLarge(
            f"tree {tree.name!r} emits {emitted} lines, over the {line_cap}-line cap "
            f"(doc 05 §7). It has {emitter.leaf_count} leaves at depth "
            f"{emitter.max_depth}. Compile time is super-linear in emitted lines "
            f"(EXPERIMENTS.md §G: proportional to lines^1.4, and 642 lines already "
            f"cost 10.9 s), and fan-out is the wall rather than depth — doc 01 §4b "
            f"measured 32 one-sided levels at 1.9 s against a depth-10 "
            f"fully-branching nest at 92 s. Split this tree into two trees composed "
            f"with `|`, so each compiles inside the budget: four units of 25 rules "
            f"is ~18 s serial against 48-57 s for one unit of 100 (§G)."
        )

    return EmittedTree(
        source=source,
        module_docstring=f"tree {tree.name!r}",
        path_fn_name=path_fn,
        matcher_fn_names=tuple(m.fn_name for m in matchers),
        output_fn_names=tuple(output_fns),
        features=tuple(emitter.features),
        string_features=string_features,
        params=tuple(emitter.params),
        emitted_lines=emitted,
        leaf_count=emitter.leaf_count,
        max_depth=emitter.max_depth,
    )


# decider 1 spells output dtypes with polars' own names in `TreeOutput.dtypes`.
# Only the three a kernel can write are emitted as steps
# (`boundary.writeback.KernelOutputs` holds float64/int64/bool groups); a
# "String" column is decoded after the fact by `TreeModule.decode`.
_PY_TYPE_BY_DTYPE: dict[str, str] = {
    "Float64": "float",
    "Float32": "float",
    "Int64": "int",
    "Int32": "int",
    "Boolean": "bool",
}


def _literal(value: Any, py_type: str) -> str:
    """A leaf's output value as source.

    This is the one place a document value *is* emitted into source, and it
    is deliberate: an output row is the tree's SHAPE, not a tuning knob.
    Doc 08 §2 puts "rules added, removed or restructured" in the
    interiors-shape class, which recompiles; changing what a leaf *returns*
    is that kind of change. A threshold is the other kind and never lands
    here — see `_Emitter._threshold`.
    """
    if value is None:
        return {"float": "0.0", "int": "-1", "bool": "False"}[py_type]
    if py_type == "bool":
        return repr(bool(value))
    if py_type == "int":
        return repr(int(value))
    return repr(float(value))
