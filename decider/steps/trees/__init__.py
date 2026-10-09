"""Decision trees: `TreeConfig` runs a v3 tree or a flat-rule document as one row node."""
import typing as t

from pydantic import Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from decider._engine.ir.nodes import CallNode, IRNode, SequenceNode
from decider.steps._configurable import ConfigurableStep
from decider.steps.trees._condition import expr_condition
from decider.steps.trees._encode import encode, feature_kinds, kind_of
from decider.steps.trees._reference import reference
from decider.steps.trees._routes import Route, Visit, with_route_columns
from decider.steps.trees._split import DEFAULT_PART_OUTPUTS, split
from decider.steps.trees.schema import Node, Tree, TreeDocument
from decider.steps.trees._walker import walk

__all__ = ["Route", "TreeConfig", "Visit", "expr_condition"]


class TreeConfig(ConfigurableStep):
    """A decision tree as a step: a v3 tree document or decider 0.3 flat rules, run once per row.

    The tree's output columns are the step's outputs (`<rule>.<column>` per
    rule in a prioritized document with `mode: "all"`). A threshold, bound,
    value or pattern written `{"param": "hi_thresh", "default": 0.7}` is a
    param of the step, so retuning it never rebuilds or recompiles anything.
    A feature's type is inferred from how the tree uses it (`string_match`:
    `str`; only `is_true`/`is_false`: `bool`; otherwise `float`); declare it
    in `feature_types` to compare an int64 column exactly.

    A `mode: "all"` tree writes one output per rule and column, and a tree of
    more than `part_outputs` outputs (64 by default) is run as several parts
    of whole rules, each its own node `<name>/part_<k>`, because compile time
    grows with the square of the outputs one call returns. A part costs about
    15-20 us per request, so a bigger `part_outputs` (or `None`, no parts) makes
    `score()` faster and the first build slower (a 521-rule tree takes about
    4 minutes to bind unsplit); a smaller one (32) makes `run()` over a big frame
    up to twice as fast. A tree with params is never split.

    `null_handling` says what a null numeric or boolean feature does.
    `"otherwise"` (the default, as decider 0.3 trees behave): a test on it
    is unknown, NOT keeps it unknown, AND/OR follow three-valued logic and
    the node takes its otherwise branch (a cases node tries its next case).
    `"error"`: a null is a `MissingInputError`. A string match has its own
    `null_handling` (`no_match`, `match` or `error`).

    `path_output` names an extra String column holding the id of the leaf
    that answered (null when the default row answered); in `mode: "all"`
    one per rule, `<rule>.<path_output>`. `trace_output` names a String
    column holding the ordered path, the ids of every node walked joined
    with `>` (`"root>mid>hi"`, or `"root>mid"` when `mid`'s branch leads
    nowhere), so the decision record says how the tree got there, also
    through a node shared by two parents. In `mode: "first_match"` over
    several rules it is the path in the rule that answered (else the last
    rule's); in `mode: "all"` one per rule.

    `path_columns` is a function `fn(route, tree) -> dict` that adds the
    dict's keys as output columns, for a route format of your own. It is
    called once per route when the step is built, never per row: the tree
    is unfolded so every route, each branch of a node told apart, ends at a
    leaf of its own whose row holds the route's columns (so `path_output`
    names a leaf even where the default row answers). A `Route` has
    `visits`, a `Visit(node_id, node, branch, child)` per node passed, and
    `result_idx`. Set it in Python, or with `path_columns: fn` on a flow
    file's `step:` node naming a function in `components`; a document
    never holds it. A first-match tree must have one rule.

    Params are per call, not per row: `params={"risk_tree": {"hi_thresh": 2.0}}`
    moves the threshold for every row of that run. For a threshold that
    differs per row, compare against a computed feature over a column
    instead, e.g. `{"op": ">", "feature": {"type": "computed", "expression":
    "ratio - customer_limit"}, "threshold": 0.0}` reads `customer_limit` from
    each row. A param name is a Python identifier (it becomes a field name):
    write rule `CF-0412`'s threshold as `{"param": "cf_0412_amount"}`.

    `nodes` maps each node id to its node: the locators a session breaks on,
    e.g. `session.break_at("risk_tree#high")`.

    Example::

        risk = TreeConfig.load({
            "type": "tree", "name": "risk_tree",
            "tree": {
                "nodes": [
                    {"id": "root", "data": {"type": "unary", "condition": {
                        "op": ">", "feature": "ratio", "threshold": {"param": "hi_thresh", "default": 0.7}}}},
                    {"id": "high", "data": {"type": "leaf", "result_idx": 0}},
                ],
                "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
                "output": {"data": [{"risk_band": 1}], "default": {"risk_band": 0},
                           "dtypes": [["risk_band", "Int64"]]},
            },
        })
        risk.run(df)                                          # writes risk_band
        risk.run(df, params={"risk_tree": {"hi_thresh": 2.0}})
        # With "path_output": "risk_leaf" in the document, run() also writes risk_leaf: "high" or null;
        # with "trace_output": "risk_path", risk_path: "root>high", or "root" when ratio <= 0.7.

    The same tree as flat rules: `"tree": {"type": "flat_rule", "rule": {"rule": {"type": "unary",
    "condition": {...}, "then": {"type": "leaf", "result_idx": 0}}}, "output": {...}}`.
    """

    type: t.Literal["tree"] = "tree"
    tree: TreeDocument
    feature_types: t.Dict[str, str] = {}
    part_outputs: t.Optional[int] = Field(DEFAULT_PART_OUTPUTS, ge=1)
    null_handling: t.Literal["otherwise", "error"] = "otherwise"
    path_output: t.Optional[str] = None
    trace_output: t.Optional[str] = None
    # Code, set in Python or by a flow file's `path_columns:`; a document never holds it.
    path_columns: SkipJsonSchema[t.Optional[t.Callable[[Route, Tree], dict]]] = Field(None, exclude=True, repr=False)

    @field_validator("feature_types")
    @classmethod
    def _kinds(cls, value: t.Dict[str, str]) -> t.Dict[str, str]:
        return {k: kind_of(v, f"feature_types[{k!r}]") for k, v in value.items()}

    @property
    def nodes(self) -> t.Dict[str, Node]:
        """Every node the tree runs, by id: what a `<name>#<id>` locator points at."""
        return self._tree().nodes

    def _tree(self) -> Tree:
        tree = self.tree.to_tree()
        return tree if self.path_columns is None else with_route_columns(tree, self.path_columns)

    def to_ir(self, ctx: t.Any) -> IRNode:
        tree = self._tree()
        extra = (self.path_output is not None) + (self.trace_output is not None)
        parts = [tree] if self.path_columns is not None else split(tree, extra, self.part_outputs)
        origin = ctx.origin(self)
        if len(parts) > 1:
            inner = ctx.child(self.name)
            # A part reads each feature as the whole tree does, not as its own rules alone would infer it.
            kinds = feature_kinds(tree, self.feature_types)
            nodes = [self._row(part, inner.origin(self, f"part_{k}"), ctx, {f: kinds[f] for f in feature_kinds(part, {})})
                     for k, part in enumerate(parts)]
            # A param is numbered within its own node and named after its path, so a tree with params stays whole.
            if not any(n.params for n in nodes):
                return SequenceNode(origin, tuple(nodes))
        return self._row(tree, origin, ctx, self.feature_types)

    def _row(self, tree: Tree, origin: t.Any, ctx: t.Any, feature_types: t.Mapping[str, str]) -> CallNode:
        p = encode(tree, feature_types, ctx.value, self.null_handling, self.path_output, self.trace_output)
        ref = reference(tree, p.inputs, p.kinds, p.columns)
        # The node's consts are addresses into these arrays; the node holds its reference, which holds them.
        ref.arrays = p.arrays
        fn = _python(reference(tree, p.inputs, p.kinds, p.columns, shared=True), p.columns) if p.python else walk
        return CallNode(origin, "row", fn, p.inputs, p.outputs, p.params, reference=ref, consts=p.consts)


def _ignore(locator: str) -> None:
    pass


def _python(ref: t.Callable, columns: t.Sequence[tuple]) -> t.Callable:
    # String features need the raw strings, which a kernel never sees, so this
    # runs in Python in every mode (numba can't type it, so compiled modes fall
    # back to it). Like the kernel, it returns a string output as its index.
    # A list column's values are unhashable, so they are found by identity: `ref` returns the table's own objects.
    index = [None if c is None else (False, {v: k for k, v in enumerate(c)}) if isinstance(c, tuple)
             else (True, {id(v): k for k, v in enumerate(c.values)}) for _, _, c in columns]

    def indexed(row: tuple, params: t.Any, consts: tuple) -> tuple:
        out = ref(row, params, consts, _ignore)
        return tuple(v if m is None else -1 if v is None else m[1][id(v) if m[0] else v] for v, m in zip(out, index))

    # Kept to one call of a Python object, so numba fails typing it (a NumbaError, which falls back).
    def fn(row: tuple, params: t.Any, consts: tuple) -> tuple:
        return indexed(row, params, consts)

    return fn
