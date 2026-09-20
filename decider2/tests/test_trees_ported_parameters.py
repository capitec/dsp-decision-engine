"""Ported from decider 1: `tests/rules/test_parameters.py` (6 tests).

Migration-conformance suite — see `decider2/tests/PORTED.md`.

All six port, but four needed real adaptation because decider2's parameter
model is structurally different from decider 1's, not just renamed
(`trees/schema.py`'s divergence 3):

* decider 1's `InputRef` resolves to `parameters.struct.field(key)` — a
  DataFrame *column*, fed through `FlatRuleModule(parameters={"k":
  ParameterInfo(default_value=...)})` and overridable per row by a runtime
  `parameters` struct column.
* decider2's `InputRef` resolves to a kernel `param()` — a CALL-level knob.
  Its default is set once at composition (`tree_module(tree, params=...)`,
  doc 03 §4.3's pre-bind) or left at the type's zero value; it is
  overridable per `pipeline.apply(params=...)` CALL, uniformly across
  every row that call scores. There is no decider2 mechanism for a
  DataFrame column to carry a different `InputRef` value per row — that is
  the one genuine capability gap here, made concrete in
  `test_inputref_cannot_vary_per_row_only_per_call` below rather than
  silently dropped.

Computed features (`Feature(type="computed", expression=...)`) are a KNOWN
GAP (doc 08 §3.2) — both tests that used one are ported as assertions that
`ComputedFeatureRemoved` is raised, decider2's actual, deliberate refusal.
"""
from __future__ import annotations

import polars as pl
import pytest

from decider2 import flow
from decider2.trees import (
    ComputedFeatureRemoved,
    InputRef,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    Tree,
    TreeOutput,
    UnaryBetween,
    UnaryLessThan,
    UnaryNode,
    tree_module,
)


def _threshold_tree(name: str) -> Tree:
    """`score < #thresh -> "low"`, else `"high"` — the one fixture shape
    every InputRef test below needs."""
    return Tree(
        name=name,
        edges=[
            MultiSourceEdge(source="root", target="low", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="high", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="root", data=UnaryNode(condition=UnaryLessThan(feature="score", threshold=InputRef(key="thresh")))),
            PositionedNode(id="low", data=LeafNode(result_idx=0)),
            PositionedNode(id="high", data=LeafNode(result_idx=1)),
        ],
        output=TreeOutput(data=[{"r": "low"}, {"r": "high"}], dtypes=[("r", "String")]),
    )


# ---------------------------------------------------------------------------
# InputRef — parameter-based thresholds
# ---------------------------------------------------------------------------


def test_inputref_uses_default_when_no_runtime_column(tmp_path):
    """decider 1: when no `parameters` column is in the frame, the default
    value is used. Adapted: decider2 has no `ParameterInfo(default_value=)`
    — the default is supplied where a `param()`'s default is ever supplied,
    `tree_module(tree, params={"thresh": 50.0})` (doc 03 §4.3's pre-bind).
    Same default value, same expected answer.
    """
    built = tree_module(_threshold_tree("t1"), build_dir=tmp_path, params={"thresh": 50.0})
    frame = pl.DataFrame({"score": [30.0, 70.0]})
    result = built.decode(flow(built.module).apply(frame))["r"].to_list()
    assert result == ["low", "high"]


def test_inputref_uses_runtime_override(tmp_path):
    """decider 1: when a `parameters` struct column is present, it
    overrides the default — decider 1's fixture used the SAME override
    value (80) for every row, so this ports unchanged in answer even
    though the override mechanism is different: decider2's is
    `pipeline.apply(frame, params={"t2": {"thresh": 80.0}})`, one value
    for the whole call rather than a per-row column. Renamed from decider
    1's `test_inputref_uses_runtime_struct_column` to name the mechanism
    accurately.
    """
    built = tree_module(_threshold_tree("t2"), build_dir=tmp_path)
    frame = pl.DataFrame({"score": [30.0, 70.0]})
    result = built.decode(
        flow(built.module).apply(frame, params={"t2": {"thresh": 80.0}})
    )["r"].to_list()
    assert result == ["low", "low"]


def test_inputref_cannot_vary_per_row_only_per_call(tmp_path):
    """decider 1's `test_inputref_runtime_overrides_default_per_row`: a
    runtime `parameters` struct column held a DIFFERENT threshold per row
    (20 for row 0, 80 for row 1) and decider 1 resolved each row against
    its own value, giving `["high", "low"]`.

    This does NOT port. `trees/schema.py` divergence 3 is explicit that an
    `InputRef` is a shared knob, not a per-row column: "Two nodes
    referencing the same key share one knob." One `pipeline.apply(params=
    ...)` call sets ONE value for every row it scores; there is no decider2
    construction that reproduces decider 1's per-row variation for an
    `InputRef` threshold (a genuinely per-row comparison would need an
    ordinary Feature-to-Feature node, which the current unary vocabulary
    does not have either).

    Ported as the actual, deliberate decider2 behaviour instead: the SAME
    override reaches every row of one call, so the two rows that decider 1
    resolved to DIFFERENT answers resolve to the SAME answer here, for
    either candidate value.
    """
    built = tree_module(_threshold_tree("t3"), build_dir=tmp_path)
    frame = pl.DataFrame({"score": [30.0, 70.0]})

    # decider 1: row 0 wants thresh=20 (30 is NOT < 20 -> "high").
    low_thresh = built.decode(
        flow(built.module).apply(frame, params={"t3": {"thresh": 20.0}})
    )["r"].to_list()
    # decider 1: row 1 wants thresh=80 (70 IS < 80 -> "low").
    high_thresh = built.decode(
        flow(built.module).apply(frame, params={"t3": {"thresh": 80.0}})
    )["r"].to_list()

    # decider2 cannot give row 0 and row 1 different thresholds in one
    # call: every row gets whichever value the call supplied.
    assert low_thresh == ["high", "high"]   # decider 1 got ["high", "low"]
    assert high_thresh == ["low", "low"]    # decider 1 got ["high", "low"]
    assert low_thresh != high_thresh        # the knob does move the answer...
    assert len(set(low_thresh)) == 1        # ...but only for the WHOLE call


def test_inputref_between_with_two_parameters(tmp_path):
    """decider 1: both min and max of a Between condition can be
    InputRef-bound. decider 1's fixture never actually overrode either at
    runtime (no `parameters` column in its frame), so this ports as a pure
    rename of the default-supplying mechanism, same as test 1 above.
    """
    tree = Tree(
        name="between2",
        edges=[
            MultiSourceEdge(source="root", target="in_range", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="out", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="root", data=UnaryNode(condition=UnaryBetween(
                feature="score",
                min=InputRef(key="lo"),
                max=InputRef(key="hi"),
            ))),
            PositionedNode(id="in_range", data=LeafNode(result_idx=0)),
            PositionedNode(id="out", data=LeafNode(result_idx=-1)),
        ],
        output=TreeOutput(data=[{"r": "in_range"}], default={"r": "out"}, dtypes=[("r", "String")]),
    )
    built = tree_module(tree, build_dir=tmp_path, params={"lo": 20.0, "hi": 80.0})
    frame = pl.DataFrame({"score": [10.0, 50.0, 90.0]})
    result = built.decode(flow(built.module).apply(frame))["r"].to_list()
    assert result == ["out", "in_range", "out"]


# ---------------------------------------------------------------------------
# Computed features — KNOWN GAP (doc 08 §3.2)
# ---------------------------------------------------------------------------


def test_computed_feature_two_column_expression_is_refused():
    """decider 1: a computed feature combining two columns
    (`"amount * quantity"`, evaluated with `simpleeval`) is evaluated
    correctly. decider2 removes expression-string features entirely (doc
    08 §3.2) — ported as the assertion decider2 actually makes:
    `ComputedFeatureRemoved`, naming the replacement (a step before the
    tree)."""
    with pytest.raises(ComputedFeatureRemoved, match="amount \\* quantity"):
        UnaryLessThan(
            feature={"type": "computed", "expression": "amount * quantity"},
            threshold=100.0,
        )


def test_computed_feature_using_a_parameter_is_refused():
    """decider 1: a computed feature referencing `p.bonus` (a parameter
    inside the expression string) is evaluated correctly. Same GAP as
    above, same replacement named in the error."""
    with pytest.raises(ComputedFeatureRemoved, match="amount \\+ p\\.bonus"):
        UnaryLessThan(
            feature={"type": "computed", "expression": "amount + p.bonus"},
            threshold=100.0,
        )
