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
    UnaryBetween,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryNode,
    UnaryStringMatch,
    encode_tree,
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
    driver source." Stronger than before now that there is no more source
    to check: every threshold is a `ParamDecl` on `path_step.params`, named
    and defaulted — never a literal anywhere `fn`'s own closure could have
    embedded it (`fn` is built directly, doc 08 §3.4; there is no text
    generation step left for a constant to leak into).
    """
    encoded = encode_tree(_two_level(threshold_a=31.5, threshold_b=701.5))
    defaults = {p.name: p.default for p in encoded.path_step.params}

    assert 31.5 in defaults.values()
    assert 701.5 in defaults.values()
    assert "age_node_thr" in defaults  # read through its own param name


def test_retuning_a_threshold_never_recompiles(tmp_path):
    """Doc 08 §2's central guarantee, for a tree.

    EXPERIMENTS.md §L measured thresholds-as-literals at 8 full recompiles
    for 8 retunes, against 0 compile events as arguments with `signatures`
    staying 1->1. This asserts the tree engine lands on the second of those.
    """
    from decider2.runtime.invoke import DEFAULT_BUILD_DIR

    built = tree_module(_two_level(), build_dir=tmp_path)
    pipeline = flow(built.module)
    frame = pl.DataFrame({"age": [25.0, 45.0], "score": [800.0, 600.0]})

    steps, group_ids, owners, _ = pipeline.flatten_for_runtime()
    # `build_dir` must match what `pipeline.apply()` uses internally
    # (`Pipeline.apply` takes no `build_dir=` of its own) so this driver and
    # `apply()`'s own hit the SAME `build_driver` cache entry and are the
    # SAME object.
    driver = build_driver(
        list(steps), list(group_ids), owners=list(owners),
        build_dir=DEFAULT_BUILD_DIR, terminal_names=frozenset({"risk_path", "pts"}),
    )
    assert driver.segments[0].kind == "compiled"  # or the claim is vacuous

    answers = []
    for age_threshold in (30.0, 40.0, 50.0, 18.0):
        out = pipeline.apply(
            frame, params={"risk": {"age_node_thr": age_threshold}}, mode="fused"
        )
        answers.append(out["pts"].to_list())

    # 2, not 1: `risk_path` and `pts` are each their own `PackedCompiledSegment`
    # now (`decider2.compile.driver.build_packed_kernel`'s own docstring —
    # a packed step never fuses with a neighbour, a real, reported scope
    # cut of this pass). The claim this test exists to pin — retuning never
    # GROWS either signature — still holds across all four retunes above.
    assert len(driver.signatures) == 2
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


def test_two_same_shaped_sibling_thresholds_do_not_collide(tmp_path):
    """The regression this whole migration exists for.

    A generated-source codegen once emitted two sibling conditions of the
    same shape under one collided parameter name, so `5 < x < 10` compiled
    to `(x > root_thr) and (x < root_thr)` — never satisfiable, every AND of
    two same-shaped conditions silently `False`. It was found only by
    porting decider 1's own tests.

    The array-encoded walker retires the whole defect CLASS rather than
    patching the symptom: a condition is never a named identifier, only an
    integer array slot (`decider2.trees.interpreter`'s module docstring), so
    two same-shaped siblings cannot collide on a slot by construction. This
    asserts both directions: the two thresholds show up as two distinct,
    independently retunable params, AND the AND is genuinely satisfiable —
    which the historical bug made impossible for exactly this shape (two
    `_ThresholdedUnaryOp`s on the same feature, combined with AND).
    """
    tree = Tree(
        name="band",
        edges=[
            MultiSourceEdge(source="root", target="leaf_in", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="root", target="leaf_out", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="root", data=CompositeNode(
                op="and",
                conditions=[
                    UnaryGreaterThan(feature="x", threshold=5.0),
                    UnaryLessThan(feature="x", threshold=10.0),
                ],
            )),
            PositionedNode(id="leaf_in", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_out", data=LeafNode(result_idx=1)),
        ],
        output=TreeOutput(
            data=[{"pts": 1.0}, {"pts": 0.0}],
            default={"pts": -1.0}, dtypes=[("pts", "Float64")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)

    schema = built.module.params_schema()
    thresholds = {k: v for k, v in schema.items()}
    assert len(thresholds) == 2  # two distinct params, not one collided name
    assert set(thresholds.values()) == {5.0, 10.0}

    frame = pl.DataFrame({"x": [7.0, 3.0, 12.0, 5.0, 10.0]})
    out = flow(built.module).apply(frame)

    # 7 is strictly between 5 and 10 -> in-band; 3 and 12 are outside;
    # 5 and 10 are the (exclusive) boundaries themselves -> out-of-band.
    # Every one of these being 0.0 (never 1.0) is exactly the historical
    # bug: "never satisfiable".
    assert out["pts"].to_list() == [1.0, 0.0, 0.0, 0.0, 0.0]
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


def test_a_very_large_tree_no_longer_hits_a_line_cap():
    """Doc 05 §7's ~500-line cap existed because a tree's SHAPE was emitted
    as nested `if`/`elif` source, and compile time was super-linear in
    emitted lines (EXPERIMENTS.md §G). Once shape is DATA
    (`decider2.trees.interpreter`), and now that the wrapper itself is a
    pre-built closure rather than any source at all (doc 08 §3.4), there is
    no line count left to grow — so the 400-node otherwise-chain that used
    to be a hard `TreeTooLarge` build error just builds, and the feature
    SET (as opposed to the per-node threshold count) stays flat regardless
    of chain length: `_chain` reuses one column, `x`, at every node.
    """
    small = encode_tree(_chain(5, arm=1))
    large = encode_tree(_chain(400, arm=1))

    assert small.features == large.features == ("x",)
    assert large.leaf_count == 400 + 1  # every node's other arm, plus the tail


def test_a_then_chain_deeper_than_cpython_allows_now_just_builds():
    """The one shape that used to still nest (a `then`-chain, one CPython
    indentation level per node) and hit CPython's own "too many levels of
    indentation" limit at 128 nodes, before numba was ever reached. There is
    no source-level nesting left for that limit to apply to at all — a
    120-deep then-chain is array data, walked by one iterative loop
    (`decider2.trees.interpreter.walk_tree`), same as an otherwise-chain of
    the same length, and built as a closure directly (doc 08 §3.4), never
    text.
    """
    encoded = encode_tree(_chain(120, arm=0))

    assert encoded.leaf_count == 121
    assert encoded.max_depth == 121


def test_a_computed_feature_now_compiles_instead_of_being_refused():
    """Doc 08 §1.2/§3.2 (revised) and doc 06 §O15 (revised): a computed
    feature is a closed, statically-validated expression grammar
    (`decider2.expr`), compiled to numba source at build time — not the
    `_ComputedFeature`/`simpleeval` pointer-shaped objection doc 01 §5.4
    actually recorded. `ComputedFeatureRemoved` is no longer raised for
    this; the owner's own example (`x - y`, here `income - expenses`) just
    works, with no preprocessing step required."""
    cond = UnaryLessThan(
        feature={"type": "computed", "expression": "income - expenses"},
        threshold=1.0,
    )
    assert cond.feature.required_features() == {"income", "expenses"}


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
        encode_tree(tree)


def test_encoding_the_same_tree_twice_is_identical(tmp_path):
    """Doc 05 §4.2's determinism requirement — same document in, same
    arrays out, every time: the walk never depends on set/dict iteration
    order or anything else non-deterministic, so the SAME tree builds the
    SAME closure content twice — which is what lets numba's own closure-hash
    caching (`decider2.trees.encode`'s module docstring) hit on the second
    build instead of compiling again."""
    first = encode_tree(_two_level())
    second = encode_tree(_two_level())

    assert first.arrays == second.arrays
    assert first.features == second.features
    assert [(p.name, p.default) for p in first.params] == [(p.name, p.default) for p in second.params]


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
