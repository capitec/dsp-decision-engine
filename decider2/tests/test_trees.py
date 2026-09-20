"""The tree engine's own properties.

The load-bearing one is `test_retuning_a_threshold_never_recompiles`: the
whole migration rests on decider 1's `Union[float, InputRef]` threshold
collapsing to "a kernel argument" in both arms, and this pins it the same
way `test_compile_driver.py` and `test_string_params.py` pin it for
hand-written steps — by asserting `len(driver.signatures)` does not grow.
"""
from __future__ import annotations

import polars as pl
import pytest

from decider2 import flow
from decider2.compile.driver import build_driver
from decider2.testing import assert_equivalent
from decider2.trees import (
    CasesIsIn,
    CasesRanges,
    ComputedFeatureRemoved,
    CompositeNode,
    InputRef,
    IsInCondition,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    RangeCondition,
    Tree,
    TreeOutput,
    TreeTooLarge,
    UnaryBetween,
    UnaryGreaterThanEqual,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryNode,
    UnaryStringMatch,
    emit_tree,
    tree_module,
    UnsupportedInKernel,
)


def _two_level(threshold_a=30.0, threshold_b=700.0) -> Tree:
    """age < A, then score >= B — two thresholds, four leaves."""
    return Tree(
        name="risk",
        edges=[
            MultiSourceEdge(source="age_node", target="score_node", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="age_node", target="leaf_old", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="score_node", target="leaf_good", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="score_node", target="leaf_bad", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="age_node", data=UnaryNode(
                condition=UnaryLessThan(feature="age", threshold=threshold_a))),
            PositionedNode(id="score_node", data=UnaryNode(
                condition=UnaryGreaterThanEqual(feature="score", threshold=threshold_b))),
            PositionedNode(id="leaf_good", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_bad", data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_old", data=LeafNode(result_idx=2)),
        ],
        output=TreeOutput(
            data=[{"pts": 10.0}, {"pts": 20.0}, {"pts": 30.0}],
            default={"pts": 0.0},
            dtypes=[("pts", "Float64")],
        ),
    )


# ---------------------------------------------------------------------------
# The claim the migration rests on
# ---------------------------------------------------------------------------


def test_no_threshold_is_ever_written_into_emitted_source(tmp_path):
    """Doc 05 §4.2, verbatim: "No decision-relevant constant is emitted into
    driver source." A literal threshold appears exactly once — as a
    `param()` DEFAULT in the signature — and never in the body.
    """
    emitted = emit_tree(_two_level(threshold_a=31.5, threshold_b=701.5))
    # Only the traversal function — a leaf's OUTPUT value is deliberately
    # emitted (see `codegen._literal`: what a leaf returns is the tree's
    # shape, doc 08 §2's interiors-shape class, not a tuning knob).
    after_signature = emitted.source.split("-> int:", 1)[1]
    body = after_signature.split("\ndef ", 1)[0]

    assert "param(31.5)" in emitted.source
    assert "param(701.5)" in emitted.source
    assert "31.5" not in body
    assert "701.5" not in body
    assert "age_node_thr" in body  # it is read through its argument name


def test_retuning_a_threshold_never_recompiles(tmp_path):
    """Doc 08 §2's central guarantee, for a tree.

    EXPERIMENTS.md §L measured thresholds-as-literals at 8 full recompiles
    for 8 retunes, against 0 compile events as arguments with `signatures`
    staying 1->1. This asserts the tree engine lands on the second of those.
    """
    built = tree_module(_two_level(), build_dir=tmp_path)
    pipeline = flow(built.module)
    frame = pl.DataFrame({"age": [25.0, 45.0], "score": [800.0, 600.0]})

    steps, group_ids, owners, _ = pipeline.flatten_for_runtime()
    driver = build_driver(
        list(steps), list(group_ids), owners=list(owners),
        build_dir=tmp_path, terminal_names=frozenset({"risk_path", "pts"}),
    )
    assert driver.segments[0].kind == "compiled"  # or the claim is vacuous

    answers = []
    for age_threshold in (30.0, 40.0, 50.0, 18.0):
        out = pipeline.apply(
            frame, params={"risk": {"age_node_thr": age_threshold}}, mode="fused"
        )
        answers.append(out["pts"].to_list())

    assert len(driver.signatures) == 1
    assert answers[0] != answers[3]  # the retune actually changed answers


def test_a_literal_and_an_inputref_threshold_become_the_same_thing(tmp_path):
    """decider 1's `Union[float, InputRef]`, both arms.

    A literal becomes a param with that default; an `InputRef` becomes a
    param named by its key. Both are kernel arguments, and both retune
    through the same `params={...}` call.
    """
    literal_tree = _two_level()
    ref_tree = Tree(
        name="risk",
        edges=literal_tree.edges,
        nodes=[
            PositionedNode(id="age_node", data=UnaryNode(
                condition=UnaryLessThan(feature="age", threshold=InputRef(key="age_floor")))),
            *literal_tree.nodes[1:],
        ],
        output=literal_tree.output,
    )

    literal = tree_module(literal_tree, build_dir=tmp_path)
    ref = tree_module(ref_tree, name="risk_ref", build_dir=tmp_path)

    assert "age_node_thr" in literal.module.params_schema()
    assert "age_floor" in ref.module.params_schema()

    frame = pl.DataFrame({"age": [25.0, 45.0], "score": [800.0, 600.0]})
    from_literal = flow(literal.module).apply(
        frame, params={"risk": {"age_node_thr": 40.0}}
    )["pts"].to_list()
    from_ref = flow(ref.module).apply(
        frame, params={"risk_ref": {"age_floor": 40.0}}
    )["pts"].to_list()

    assert from_literal == from_ref


def test_one_inputref_named_twice_is_one_knob(tmp_path):
    """Two nodes referencing `#floor` share a single param — which is what
    a named reference is for (decider 1's `common/parameters.py`)."""
    tree = Tree(
        name="shared_ref",
        edges=[
            MultiSourceEdge(source="a", target="b", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="a", target="leaf_2", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="b", target="leaf_0", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="b", target="leaf_1", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="a", data=UnaryNode(
                condition=UnaryLessThan(feature="x", threshold=InputRef(key="floor")))),
            PositionedNode(id="b", data=UnaryNode(
                condition=UnaryLessThan(feature="y", threshold=InputRef(key="floor")))),
            PositionedNode(id="leaf_0", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_1", data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_2", data=LeafNode(result_idx=2)),
        ],
        output=TreeOutput(
            data=[{"pts": 0.0}, {"pts": 1.0}, {"pts": 2.0}],
            default={"pts": -1.0}, dtypes=[("pts", "Float64")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)

    assert list(built.module.params_schema()) == ["floor"]


# ---------------------------------------------------------------------------
# Path capture — doc 03 §7
# ---------------------------------------------------------------------------


def test_a_tree_reports_which_leaf_it_reached(tmp_path):
    """The single most-requested missing thing (COLD-READ.md).

    `<name>_path` is an ordinary int64 column carrying the reached leaf's
    `result_idx` — doc 03 §7's convention, no keyword and no special case.
    """
    built = tree_module(_two_level(), build_dir=tmp_path)
    frame = pl.DataFrame({"age": [25.0, 25.0, 45.0], "score": [800.0, 600.0, 0.0]})

    out = flow(built.module).apply(frame)

    assert built.path_column == "risk_path"
    assert out["risk_path"].to_list() == [0, 1, 2]
    assert out["risk_path"].dtype == pl.Int64


def test_the_path_column_survives_an_explicit_emit(tmp_path):
    """It is emittable like any other value, including alongside `.drop()`."""
    built = tree_module(_two_level(), build_dir=tmp_path)
    pipeline = flow(built.module).emit("risk_path")
    frame = pl.DataFrame({"age": [25.0], "score": [800.0]})

    assert "risk_path" in pipeline.apply(frame).columns


def test_an_unconnected_branch_reaches_the_default_leaf(tmp_path):
    """decider 1's `get_child` returns `LeafRule(result_idx=-1)` for a
    branch with no edge; the same -1 sentinel is produced here."""
    tree = Tree(
        name="partial",
        edges=[MultiSourceEdge(source="root", target="leaf_yes", data=MultiEdgeData(sourceIndex=[0]))],
        nodes=[
            PositionedNode(id="root", data=UnaryNode(
                condition=UnaryLessThan(feature="x", threshold=10.0))),
            PositionedNode(id="leaf_yes", data=LeafNode(result_idx=0)),
        ],
        output=TreeOutput(
            data=[{"pts": 1.0}], default={"pts": 99.0}, dtypes=[("pts", "Float64")]
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    out = flow(built.module).apply(pl.DataFrame({"x": [5.0, 50.0]}))

    assert out["partial_path"].to_list() == [0, -1]
    assert out["pts"].to_list() == [1.0, 99.0]


# ---------------------------------------------------------------------------
# Node kinds
# ---------------------------------------------------------------------------


def test_composite_and_between_and_isin_nodes(tmp_path):
    """One tree exercising `CompositeNode(AND)`, `UnaryBetween` (inclusive
    both ends, decider 1's own semantics) and `CasesIsIn`."""
    tree = Tree(
        name="kinds",
        edges=[
            MultiSourceEdge(source="comp", target="cases", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="comp", target="leaf_no", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="cases", target="leaf_a", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="cases", target="leaf_b", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(source="cases", target="leaf_other", data=MultiEdgeData(sourceIndex=[2])),
        ],
        nodes=[
            PositionedNode(id="comp", data=CompositeNode(
                op="and",
                conditions=[
                    UnaryBetween(feature="age", min=18.0, max=65.0),
                    UnaryIsTrue(feature="verified"),
                ],
            )),
            PositionedNode(id="cases", data=CasesIsIn(
                feature="region",
                conditions=[IsInCondition(values=[1, 2]), IsInCondition(values=[3])],
            )),
            PositionedNode(id="leaf_a", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_b", data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_other", data=LeafNode(result_idx=2)),
            PositionedNode(id="leaf_no", data=LeafNode(result_idx=-1)),
        ],
        output=TreeOutput(
            data=[{"pts": 1.0}, {"pts": 2.0}, {"pts": 3.0}],
            default={"pts": 0.0}, dtypes=[("pts", "Float64")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({
        "age": [30.0, 30.0, 30.0, 70.0, 18.0, 65.0],
        "verified": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        "region": [1.0, 3.0, 9.0, 1.0, 2.0, 1.0],
    })
    out = flow(built.module).apply(frame)

    # rows 0-2: composite passes, region routes; row 3: age>65 fails;
    # row 4: age==18 passes (inclusive); row 5: not verified.
    assert out["pts"].to_list() == [1.0, 2.0, 3.0, 0.0, 1.0, 0.0]
    assert_equivalent(flow(built.module), frame)


def test_a_tree_composes_with_ordinary_modules(tmp_path):
    """`flow(Affordability, my_tree, Scoring)` — the integration the task
    asks for. The tree is a `Module`; nothing else knows it is a tree."""

    def disposable_income(net_income: float, expenses: float) -> float:
        """Income remaining after committed expenses."""
        return net_income - expenses

    def final_score(pts: float) -> float:
        """Scale the tree's points."""
        return pts * 2.0

    tree = Tree(
        name="affordability_band",
        edges=[
            MultiSourceEdge(source="root", target="leaf_low", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="leaf_high", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="root", data=CasesRanges(
                feature="disposable_income",
                conditions=[RangeCondition(max=1000.0), RangeCondition(min=1000.0)],
                strict=False,
            )),
            PositionedNode(id="leaf_low", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_high", data=LeafNode(result_idx=1)),
        ],
        output=TreeOutput(
            data=[{"pts": 5.0}, {"pts": 50.0}],
            default={"pts": 0.0}, dtypes=[("pts", "Float64")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    pipeline = flow(disposable_income, built.module, final_score)
    frame = pl.DataFrame({"net_income": [5000.0, 2000.0], "expenses": [1500.0, 1500.0]})

    out = pipeline.apply(frame)

    assert out["final_score"].to_list() == [100.0, 10.0]
    assert "affordability_band" in pipeline.params_schema()
    assert_equivalent(pipeline, frame)


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def _chain(length: int, *, arm: int) -> Tree:
    """A chain of `length` conditions, continuing down `arm`.

    `arm=1` is the otherwise-chain (a policy waterfall) — flat source.
    `arm=0` is the then-chain — one indentation level per node.
    """
    nodes, edges = [], []
    for i in range(length):
        nodes.append(PositionedNode(id=f"n{i}", data=UnaryNode(
            condition=UnaryLessThan(feature="x", threshold=float(i)))))
        nodes.append(PositionedNode(id=f"l{i}", data=LeafNode(result_idx=0)))
        edges.append(MultiSourceEdge(
            source=f"n{i}", target=f"l{i}", data=MultiEdgeData(sourceIndex=[1 - arm])))
        if i + 1 < length:
            edges.append(MultiSourceEdge(
                source=f"n{i}", target=f"n{i + 1}", data=MultiEdgeData(sourceIndex=[arm])))
    return Tree(name="chain", nodes=nodes, edges=edges)


def test_a_tree_over_the_line_cap_is_a_build_error():
    """Doc 05 §7: "a hard cap on emitted lines (~500) per kernel, enforced
    as a build error naming the group"."""
    with pytest.raises(TreeTooLarge, match="over the 500-line cap"):
        emit_tree(_chain(400, arm=1))


def test_an_otherwise_chain_costs_no_indentation():
    """Every tree path ends in a `return`, so an `otherwise` arm needs no
    `else:` — which is what keeps a long policy waterfall emittable at all.

    Measured, not assumed: with a nested `else:` this shape hit CPython's
    own "too many levels of indentation" limit at 128 nodes, before numba
    was ever reached.
    """
    emitted = emit_tree(_chain(120, arm=1))
    body = emitted.source.split("-> int:", 1)[1].split("\ndef ", 1)[0]
    deepest = max((len(ln) - len(ln.lstrip())) for ln in body.splitlines() if ln.strip())

    assert "else:" not in body
    assert deepest <= 8  # the function's own indent, plus one level


def test_a_then_chain_deeper_than_cpython_allows_is_a_build_error():
    """The one shape that still nests. Caught with the tree named, rather
    than as an `IndentationError` pointing into a generated file."""
    with pytest.raises(TreeTooLarge, match="levels of conditions"):
        emit_tree(_chain(120, arm=0), line_cap=10**9)


def test_a_computed_feature_is_refused_with_its_replacement_named():
    """Doc 08 §3.2 removes `_ComputedFeature`; doc 06 §O15 records it
    settled. The error has to name the route that works, or a migration
    just stalls here."""
    with pytest.raises(ComputedFeatureRemoved, match="before the tree"):
        UnaryLessThan(
            feature={"type": "computed", "expression": "income - expenses"},
            threshold=1.0,
        )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"match_type": "contains"}, "frame tier"),
        ({"match_type": "regex"}, "nopython"),
        ({"case_sensitive": False}, "to_lowercase"),
        ({"trim_whitespace": True}, "strip_chars"),
    ],
)
def test_string_matching_a_kernel_cannot_do_is_refused_not_approximated(kwargs, expected):
    """Doc 05 §1.5: a string enters as an int32 code, so only `exact`
    survives. Each refusal names the frame-tier route instead."""
    tree = Tree(
        name="t",
        edges=[MultiSourceEdge(source="root", target="leaf", data=MultiEdgeData(sourceIndex=[0]))],
        nodes=[
            PositionedNode(id="root", data=UnaryNode(
                condition=UnaryStringMatch(feature="s", patterns=["a"], **kwargs))),
            PositionedNode(id="leaf", data=LeafNode(result_idx=0)),
        ],
    )
    with pytest.raises(UnsupportedInKernel, match=expected):
        emit_tree(tree)


def test_emitting_the_same_tree_twice_is_byte_identical(tmp_path):
    """Doc 05 §4.2's determinism requirement — same document in, same
    source out, so the content-addressed cache hits instead of growing."""
    first = emit_tree(_two_level())
    second = emit_tree(_two_level())

    assert first.source == second.source


def test_strict_range_validation_matches_decider1(tmp_path):
    """decider 1's `validate_range_conditions`: sorted and contiguous."""
    with pytest.raises(ValueError, match="not continuous"):
        CasesRanges(
            feature="x",
            conditions=[RangeCondition(max=10.0), RangeCondition(min=20.0)],
            strict=True,
        )


# ---------------------------------------------------------------------------
# Serving — doc 08 §4: retune a tree's threshold over /params, no recompile
# ---------------------------------------------------------------------------


def test_a_trees_thresholds_are_retunable_over_the_serving_params_endpoint(tmp_path):
    """The task's own acceptance bar, for a tree.

    A tree's thresholds must show up in `params_schema()` — which is what
    `GET /params/schema` serves — and a `POST /params` retune of one must be
    classified `VALUES` (doc 08 §2) and activate without recompiling. This
    is the whole reason a threshold is an argument rather than a literal.
    """
    from decider2.runtime.serve import ChangeClass

    built = tree_module(_two_level(), build_dir=tmp_path)
    pipeline = flow(built.module)
    record = {"age": 25.0, "score": 800.0}

    handle = pipeline.serve()
    assert handle.score(record)["pts"] == 10.0

    # GET /params/schema
    schema = pipeline.params_schema()
    assert schema["risk"]["age_node_thr"] == 30.0
    assert schema["risk"]["score_node_thr"] == 700.0

    # POST /params
    plan = handle.stage({"risk": {"score_node_thr": 900.0}})
    assert plan.klass is ChangeClass.VALUES
    assert plan.recompiles is False
    handle.activate()

    # Same record, different answer, no compile in between.
    assert handle.score(record)["pts"] == 20.0
