"""Decision trees: `TreeConfig` runs a v3 tree or a flat-rule document as one row node."""
import typing as t

from pydantic import field_validator

from decider.engine.ir.nodes import CallNode
from decider.steps.configurable import ConfigurableStep
from decider.steps.trees.encode import encode, kind_of
from decider.steps.trees.reference import reference
from decider.steps.trees.schema import *  # noqa: F401,F403
from decider.steps.trees.schema import Node, TreeDocument
from decider.steps.trees.schema import __all__ as _schema
from decider.steps.trees.walker import walk

__all__ = ["TreeConfig", *_schema]


class TreeConfig(ConfigurableStep):
    """A decision tree as a step: a v3 tree document or decider_old flat rules, run once per row.

    The tree's output columns are the step's outputs (`<rule>.<column>` per
    rule in a prioritized document with `mode: "all"`). A threshold, bound,
    value or pattern written `{"param": "hi_thresh", "default": 0.7}` is a
    param of the step, so retuning it never rebuilds or recompiles anything.
    A feature's type is inferred from how the tree uses it (`string_match`:
    `str`; only `is_true`/`is_false`: `bool`; otherwise `float`); declare it
    in `feature_types` to compare an int64 column exactly.

    `null_handling` says what a null numeric or boolean feature does.
    `"otherwise"` (the default, as decider_old trees behave): a test on it
    is unknown, NOT keeps it unknown, AND/OR follow three-valued logic and
    the node takes its otherwise branch (a cases node tries its next case).
    `"error"`: a null is a `MissingInputError`. A string match has its own
    `null_handling` (`no_match`, `match` or `error`).

    `path_output` names an extra String column holding the id of the leaf
    that answered (null when the default row answered); in `mode: "all"`
    one per rule, `<rule>.<path_output>`.

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
        # With "path_output": "risk_leaf" in the document, run() also writes risk_leaf: "high" or null.

    The same tree as flat rules: `"tree": {"type": "flat_rule", "rule": {"rule": {"type": "unary",
    "condition": {...}, "then": {"type": "leaf", "result_idx": 0}}}, "output": {...}}`.
    """

    type: t.Literal["tree"] = "tree"
    tree: TreeDocument
    feature_types: t.Dict[str, str] = {}
    null_handling: t.Literal["otherwise", "error"] = "otherwise"
    path_output: t.Optional[str] = None

    @field_validator("feature_types")
    @classmethod
    def _kinds(cls, value: t.Dict[str, str]) -> t.Dict[str, str]:
        return {k: kind_of(v, f"feature_types[{k!r}]") for k, v in value.items()}

    @property
    def nodes(self) -> t.Dict[str, Node]:
        """Every node the tree runs, by id: what a `<name>#<id>` locator points at."""
        return self.tree.to_tree().nodes

    def to_ir(self, ctx: t.Any) -> CallNode:
        tree = self.tree.to_tree()
        p = encode(tree, self.feature_types, ctx.value, self.null_handling, self.path_output)
        ref = reference(tree, p.inputs, p.kinds, p.columns)
        # The node's consts are addresses into these arrays; the node holds its reference, which holds them.
        ref.arrays = p.arrays
        return CallNode(ctx.origin(self), "row", _python(ref, p.columns) if p.python else walk, p.inputs,
                        p.outputs, p.params, reference=ref, consts=p.consts)


def _ignore(locator: str) -> None:
    pass


def _python(ref: t.Callable, columns: t.Sequence[tuple]) -> t.Callable:
    # String features need the raw strings, which a kernel never sees, so this
    # runs in Python in every mode (numba can't type it, so compiled modes fall
    # back to it). Like the kernel, it returns a string output as its index.
    index = [None if c is None else {v: k for k, v in enumerate(c)} for _, _, c in columns]

    def indexed(row: tuple, params: t.Any, consts: tuple) -> tuple:
        out = ref(row, params, consts, _ignore)
        return tuple(v if m is None else -1 if v is None else m[v] for v, m in zip(out, index))

    # Kept to one call of a Python object, so numba fails typing it (a NumbaError, which falls back).
    def fn(row: tuple, params: t.Any, consts: tuple) -> tuple:
        return indexed(row, params, consts)

    return fn
