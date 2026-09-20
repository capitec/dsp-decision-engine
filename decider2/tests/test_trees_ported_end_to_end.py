"""Ported from decider 1: `tests/rules/test_tree_end_to_end.py` (29 tests).

Migration-conformance suite — see `decider2/tests/PORTED.md` for the full
per-test ledger. Uses the same `_output()`/`_tree()`/`_run()` shape as
`test_trees_ported_conditions.py` (edge-based `Tree` in place of decider
1's nested `then=`/`otherwise=` rules); see that file's module docstring
for why.

What ported, what didn't, in one pass:

* Nested-tree flows (4 tests) port unchanged — a tree's branches can
  themselves be any node kind, same as decider 1.
* `PrioritizedFlatRuleModule`/`all`-mode tests (3) and the `output_fn`
  path-tracking tests (6) rest on machinery decider2 does not have
  (KNOWN GAPS). The path tests are ADAPTED instead of dropped: `<name>_path`
  (doc 03 §7) is a real replacement for *some* of what `output_fn` gave
  decider 1, so each is ported to assert on it where the intent survives,
  and left out where it does not (the "path depth" tests — decider2's path
  is the terminal leaf's `result_idx`, not a multi-segment breadcrumb).
* Multi-column output (2 tests) port unchanged; the third rests on
  `PrioritizedFlatRuleModule` (GAP).
* Four tests exercise decider 1's *internal* `impl.py` functions
  (`build_parameters_expr`, `prioritize_results`) or decider1-only
  execution-mode plumbing (`use_optimized_execution`,
  `RunPolarsExpression`) that decider2 has no analogue of at all — not
  because of a documented divergence, just because decider2's compiled-
  kernel architecture doesn't have an equivalent internal seam. GAP,
  recorded in PORTED.md.
* The four "empty conditions" tests and `get_required_parameters` tests
  (7 total) port, but the four empty-conditions ones surface a genuine
  finding: decider 1 accepts a `Cases*` node with zero conditions and
  always takes `otherwise`; decider2's schema also accepts it, but
  `tree_module()` then REJECTS it at build time with an unrelated-looking
  "declared but never referenced" error, because the node's feature is
  still computed as `required_features()` even though the empty-conditions
  body never reads it. See each test's docstring and PORTED.md.
"""
from __future__ import annotations

import polars as pl
import pytest

from decider2 import flow
from decider2.trees import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    CompositeNode,
    InputRef,
    IsInCondition,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    PositionedNode,
    RangeCondition,
    StringMatchCondition,
    TLogicOp,
    Tree,
    TreeOutput,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryIsIn,
    UnaryLessThan,
    UnaryNode,
    UnaryStringMatch,
    tree_module,
)

# ---------------------------------------------------------------------------
# Helpers — identical shape to test_trees_ported_conditions.py
# ---------------------------------------------------------------------------


def _output(*labels: str, default: str = "default") -> TreeOutput:
    return TreeOutput(
        data=[{"r": lbl} for lbl in labels],
        default={"r": default},
        dtypes=[("r", "String")],
    )


def _tree(name: str, nodes: dict, edges: list[tuple[str, int, str]], output: TreeOutput) -> Tree:
    positioned = [PositionedNode(id=nid, data=data) for nid, data in nodes.items()]
    by_pair: dict[tuple[str, str], list[int]] = {}
    for src, idx, tgt in edges:
        by_pair.setdefault((src, tgt), []).append(idx)
    built_edges = [
        MultiSourceEdge(source=s, target=t, data=MultiEdgeData(sourceIndex=idxs))
        for (s, t), idxs in by_pair.items()
    ]
    return Tree(name=name, nodes=positioned, edges=built_edges, output=output)


def _run(tree: Tree, frame: pl.DataFrame, *, name: str, tmp_path) -> pl.DataFrame:
    built = tree_module(tree, name=name, build_dir=tmp_path)
    return built.decode(flow(built.module).apply(frame))


# ---------------------------------------------------------------------------
# Nested tree flows
# ---------------------------------------------------------------------------


def test_nested_unary_then_cases_ranges(tmp_path):
    """decider 1: age < 30 -> CasesRanges on score (low/mid/high); age >= 30
    -> "adult". Verifies a `UnaryNode`'s then-branch can be a `CasesRanges`.

    (Also ported, independently, in `test_trees_migration.py`'s
    `test_nested_unary_then_cases_ranges_exactly_as_decider1` — kept here
    too because this file's job is a complete 1:1 file-for-file port of
    `test_tree_end_to_end.py`, not a dedupe against what another agent's
    session already covered.)
    """
    tree = _tree(
        "nested_u_cr",
        {
            "age": UnaryNode(condition=UnaryLessThan(feature="age", threshold=30.0)),
            "score": CasesRanges(
                feature="score",
                conditions=[
                    RangeCondition(max=40.0),
                    RangeCondition(min=40.0, max=70.0),
                    RangeCondition(min=70.0),
                ],
                end_logic="lower_inclusive",
                strict=False,
            ),
            "young_low": LeafNode(result_idx=0),
            "young_mid": LeafNode(result_idx=1),
            "young_high": LeafNode(result_idx=2),
            "adult": LeafNode(result_idx=3),
        },
        [
            ("age", 0, "score"), ("age", 1, "adult"),
            ("score", 0, "young_low"), ("score", 1, "young_mid"), ("score", 2, "young_high"),
        ],
        _output("young_low", "young_mid", "young_high", "adult", default="unknown"),
    )
    frame = pl.DataFrame({
        "age": [20.0, 25.0, 25.0, 35.0],
        "score": [20.0, 55.0, 80.0, 99.0],
    })
    out = _run(tree, frame, name="nested_u_cr", tmp_path=tmp_path)
    assert out["r"].to_list() == ["young_low", "young_mid", "young_high", "adult"]


def test_nested_cases_ranges_then_unary(tmp_path):
    """decider 1: `CasesRanges` on income -> low bucket -> `UnaryStringMatch`
    on region ("rural"/"urban"); high bucket -> "wealthy". Verifies a
    `CasesRanges` branch can be a `UnaryNode`."""
    tree = _tree(
        "nested_cr_u",
        {
            "income": CasesRanges(
                feature="income",
                conditions=[RangeCondition(max=50_000.0), RangeCondition(min=50_000.0)],
                end_logic="lower_inclusive",
                strict=False,
            ),
            "region": UnaryNode(condition=UnaryStringMatch(feature="region", patterns=["rural", "farm"], match_type="exact")),
            "rural_low": LeafNode(result_idx=0),
            "urban_low": LeafNode(result_idx=1),
            "wealthy": LeafNode(result_idx=2),
        },
        [
            ("income", 0, "region"), ("income", 1, "wealthy"),
            ("region", 0, "rural_low"), ("region", 1, "urban_low"),
        ],
        _output("rural_low", "urban_low", "wealthy", default="unknown"),
    )
    frame = pl.DataFrame({
        "income": [30_000.0, 30_000.0, 80_000.0],
        "region": ["rural", "city", "city"],
    })
    out = _run(tree, frame, name="nested_cr_u", tmp_path=tmp_path)
    assert out["r"].to_list() == ["rural_low", "urban_low", "wealthy"]


def test_three_level_nested_tree(tmp_path):
    """decider 1: age < 18 -> "minor"; else score < 50 -> status check
    ("active_low"/"inactive_low"); else "high_score"."""
    tree = _tree(
        "three_level",
        {
            "age": UnaryNode(condition=UnaryLessThan(feature="age", threshold=18.0)),
            "score": UnaryNode(condition=UnaryLessThan(feature="score", threshold=50.0)),
            "status": UnaryNode(condition=UnaryStringMatch(feature="status", patterns=["active"], match_type="exact")),
            "minor": LeafNode(result_idx=0),
            "active_low": LeafNode(result_idx=2),
            "inactive_low": LeafNode(result_idx=3),
            "high_score": LeafNode(result_idx=4),
        },
        [
            ("age", 0, "minor"), ("age", 1, "score"),
            ("score", 0, "status"), ("score", 1, "high_score"),
            ("status", 0, "active_low"), ("status", 1, "inactive_low"),
        ],
        _output("minor", "unused", "active_low", "inactive_low", "high_score", default="unknown"),
    )
    frame = pl.DataFrame({
        "age": [15.0, 25.0, 25.0, 30.0],
        "score": [99.0, 30.0, 30.0, 75.0],
        "status": ["x", "active", "inactive", "active"],
    })
    out = _run(tree, frame, name="three_level", tmp_path=tmp_path)
    assert out["r"].to_list() == ["minor", "active_low", "inactive_low", "high_score"]


def test_composite_inside_nested_tree(tmp_path):
    """decider 1: `UnaryNode` (age gate) -> `CompositeNode` (AND: score > 60
    AND status == "vip") -> leaf. Verifies a `CompositeNode` works as an
    embedded branch, not just a root."""
    tree = _tree(
        "comp_nested",
        {
            "age": UnaryNode(condition=UnaryGreaterThanEqual(feature="age", threshold=18.0)),
            "vip": CompositeNode(
                op=TLogicOp.AND,
                conditions=[
                    UnaryGreaterThan(feature="score", threshold=60.0),
                    UnaryStringMatch(feature="status", patterns=["vip"], match_type="exact"),
                ],
            ),
            "minor": LeafNode(result_idx=0),
            "vip_adult": LeafNode(result_idx=1),
            "regular_adult": LeafNode(result_idx=2),
        },
        [
            ("age", 0, "vip"), ("age", 1, "minor"),
            ("vip", 0, "vip_adult"), ("vip", 1, "regular_adult"),
        ],
        _output("minor", "vip_adult", "regular_adult", default="unknown"),
    )
    frame = pl.DataFrame({
        "age": [15.0, 25.0, 25.0, 30.0],
        "score": [90.0, 80.0, 80.0, 40.0],
        "status": ["vip", "vip", "basic", "vip"],
    })
    out = _run(tree, frame, name="comp_nested", tmp_path=tmp_path)
    assert out["r"].to_list() == ["minor", "vip_adult", "regular_adult", "regular_adult"]


# ---------------------------------------------------------------------------
# Path tracking — doc 03 §7's `<name>_path`, in place of decider 1's
# `output_fn`/`branch_stack` (KNOWN GAP; see module docstring).
# ---------------------------------------------------------------------------


def test_path_unary_then_branch(tmp_path):
    """decider 1 asserted a path STRING ("score,0" / "score,1") built from
    the branch_stack. decider2's `<name>_path` is the terminal leaf's own
    `result_idx` (doc 03 §7) — a single int, not a breadcrumb of every
    node visited. What survives: the then/otherwise branches are still
    distinguishable by their path value, and the value is the leaf's own
    declared `result_idx` (0 for the match, -1 for the default/otherwise
    sentinel — `LeafNode`'s own convention, already pinned by
    `test_trees.py::test_an_unconnected_branch_reaches_the_default_leaf`).
    """
    tree = _tree(
        "path_unary",
        {"root": UnaryNode(condition=UnaryLessThan(feature="score", threshold=50.0)), "low": LeafNode(result_idx=0), "high": LeafNode(result_idx=-1)},
        [("root", 0, "low"), ("root", 1, "high")],
        _output("low", default="high"),
    )
    frame = pl.DataFrame({"score": [30.0, 70.0]})
    out = _run(tree, frame, name="path_unary", tmp_path=tmp_path)

    assert out["r"].to_list() == ["low", "high"]
    assert out["path_unary_path"].to_list() == [0, -1]


def test_path_otherwise_branch_index_is_branch_count(tmp_path):
    """decider 1's convention: otherwise's path index equals the branch
    count (2, for two named conditions). decider2's convention is
    different but equally fixed: otherwise IS the `-1` default sentinel,
    not `len(conditions)` — ported to assert THAT convention, which is
    what actually distinguishes "matched branch N" from "no match" in
    decider2.
    """
    tree = _tree(
        "path_otherwise",
        {
            "root": CasesIsIn(feature="code", conditions=[IsInCondition(values=[1]), IsInCondition(values=[2])]),
            "one": LeafNode(result_idx=0),
            "two": LeafNode(result_idx=1),
            "other": LeafNode(result_idx=-1),
        },
        [("root", 0, "one"), ("root", 1, "two"), ("root", 2, "other")],
        _output("one", "two", default="other"),
    )
    frame = pl.DataFrame({"code": [1, 2, 9]})
    out = _run(tree, frame, name="path_otherwise", tmp_path=tmp_path)

    assert out["r"].to_list() == ["one", "two", "other"]
    assert out["path_otherwise_path"].to_list() == [0, 1, -1]


def test_path_nested_tree_depth_reflects_decisions(tmp_path):
    """decider 1 asserted path STRING depth (`"|".count`) grew with nesting.
    That concept has no decider2 analogue: `<name>_path` names only the
    TERMINAL leaf reached, never the route to it (doc 03 §7) — so "depth"
    is not a thing a tree reports, by design, not by omission. What ports
    is the part that does not depend on a breadcrumb: each of the three
    distinct outcomes still gets its own value, and answers are correct.
    """
    tree = _tree(
        "path_depth",
        {
            "age": UnaryNode(condition=UnaryLessThan(feature="age", threshold=30.0)),
            "score": UnaryNode(condition=UnaryLessThan(feature="score", threshold=50.0)),
            "young_low": LeafNode(result_idx=0),
            "young_high": LeafNode(result_idx=1),
            "adult": LeafNode(result_idx=2),
        },
        [("age", 0, "score"), ("age", 1, "adult"), ("score", 0, "young_low"), ("score", 1, "young_high")],
        _output("young_low", "young_high", "adult", default="unknown"),
    )
    frame = pl.DataFrame({"age": [20.0, 20.0, 40.0], "score": [30.0, 70.0, 99.0]})
    out = _run(tree, frame, name="path_depth", tmp_path=tmp_path)

    assert out["r"].to_list() == ["young_low", "young_high", "adult"]
    assert out["path_depth_path"].to_list() == [0, 1, 2]


def test_path_cases_ranges_records_correct_bucket_index(tmp_path):
    """decider 1's path index for a `CasesRanges` match IS the 0-based
    bucket position — this one ports almost verbatim, since a leaf's
    `result_idx` in decider2 was set to that same bucket position too."""
    tree = _tree(
        "path_cases",
        {
            "root": CasesRanges(
                feature="score",
                conditions=[RangeCondition(max=30.0), RangeCondition(min=30.0, max=70.0), RangeCondition(min=70.0)],
                end_logic="lower_inclusive",
                strict=False,
            ),
            "low": LeafNode(result_idx=0),
            "mid": LeafNode(result_idx=1),
            "high": LeafNode(result_idx=2),
            "none": LeafNode(result_idx=-1),
        },
        [("root", 0, "low"), ("root", 1, "mid"), ("root", 2, "high"), ("root", 3, "none")],
        _output("low", "mid", "high", default="none"),
    )
    frame = pl.DataFrame({"score": [10.0, 50.0, 80.0]})
    out = _run(tree, frame, name="path_cases", tmp_path=tmp_path)

    assert out["r"].to_list() == ["low", "mid", "high"]
    assert out["path_cases_path"].to_list() == [0, 1, 2]


def test_path_identical_inputs_always_produce_identical_paths(tmp_path):
    """decider 1: repeated identical rows always yield the same result and
    the same path. Ports unchanged."""
    tree = _tree(
        "path_repeat",
        {"root": UnaryNode(condition=UnaryLessThan(feature="v", threshold=20.0)), "match": LeafNode(result_idx=0), "no": LeafNode(result_idx=-1)},
        [("root", 0, "match"), ("root", 1, "no")],
        _output("match", default="no"),
    )
    frame = pl.DataFrame({"v": [15.0, 15.0, 15.0]})
    out = _run(tree, frame, name="path_repeat", tmp_path=tmp_path)

    paths = out["path_repeat_path"].to_list()
    assert len(set(paths)) == 1, f"expected all paths identical, got {paths}"
    assert out["r"].to_list() == ["match", "match", "match"]


# test_path_null_input_takes_otherwise_path does not port — see
# `test_trees_ported_conditions.py::test_null_input_is_routed_away_not_evaluated`
# and PORTED.md for the routing divergence this surfaces.


# ---------------------------------------------------------------------------
# Multi-column TreeOutput
# ---------------------------------------------------------------------------


def _multi_output() -> TreeOutput:
    return TreeOutput(
        data=[
            {"label": "low", "score": 0.1, "flag": False},
            {"label": "mid", "score": 0.5, "flag": False},
            {"label": "high", "score": 0.9, "flag": True},
        ],
        default={"label": "none", "score": 0.0, "flag": False},
        dtypes=[("label", "String"), ("score", "Float64"), ("flag", "Boolean")],
    )


def test_multi_column_output_all_fields_correct(tmp_path):
    """decider 1: each output row's label, score, and flag columns are
    populated correctly."""
    tree = _tree(
        "multi",
        {
            "root": CasesRanges(
                feature="x",
                conditions=[
                    RangeCondition(max=30.0),
                    RangeCondition(min=30.0, max=70.0),
                    RangeCondition(min=70.0, max=100.0),
                ],
                end_logic="lower_inclusive",
                strict=False,
            ),
            "l0": LeafNode(result_idx=0),
            "l1": LeafNode(result_idx=1),
            "l2": LeafNode(result_idx=2),
            "l3": LeafNode(result_idx=-1),
        },
        [("root", 0, "l0"), ("root", 1, "l1"), ("root", 2, "l2"), ("root", 3, "l3")],
        _multi_output(),
    )
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({"x": [10.0, 50.0, 80.0, 200.0]})
    out = built.decode(flow(built.module).apply(frame))

    assert out["label"].to_list() == ["low", "mid", "high", "none"]
    assert out["score"].to_list() == pytest.approx([0.1, 0.5, 0.9, 0.0])
    assert out["flag"].to_list() == [False, False, True, False]


def test_multi_column_output_default_row_on_no_match(tmp_path):
    """decider 1: when no condition matches, all columns come from the
    default row."""
    tree = _tree(
        "multi_default",
        {"root": UnaryNode(condition=UnaryLessThan(feature="x", threshold=0.0)), "m": LeafNode(result_idx=0), "n": LeafNode(result_idx=-1)},
        [("root", 0, "m"), ("root", 1, "n")],
        _multi_output(),
    )
    built = tree_module(tree, build_dir=tmp_path)
    frame = pl.DataFrame({"x": [50.0, 100.0]})
    out = built.decode(flow(built.module).apply(frame))

    assert out["label"].to_list() == ["none", "none"]
    assert out["score"].to_list() == pytest.approx([0.0, 0.0])
    assert out["flag"].to_list() == [False, False]


# test_multi_column_output_in_prioritized_first_match does not port —
# PrioritizedFlatRuleModule is a KNOWN GAP.


# ---------------------------------------------------------------------------
# "Empty conditions" — decider 1 defines this as always-otherwise; decider2
# accepts the SCHEMA but rejects the BUILD, which is the finding.
# ---------------------------------------------------------------------------


def test_cases_ranges_empty_conditions_returns_otherwise(tmp_path):
    """decider 1: `CasesRanges` with zero conditions falls straight to
    `otherwise` for every row — a legitimate, if degenerate, config.

    decider2's schema accepts the same shape (`CasesRanges(conditions=[])`
    is valid — `arity` is `len(conditions) + 1 == 1`, i.e. exactly the one
    "otherwise" edge). But `tree_module()` then REJECTS it: the node's
    feature is still counted into `required_features()` (schema.py's
    `CasesRanges.required_features`), which becomes a declared kernel
    parameter, yet the emitted body — correctly, since there are no
    conditions to test — never reads it, and codegen's own "declared but
    never referenced" guard (doc 03 §1, §2) catches that mismatch and
    refuses to build. Not a decider2 bug in the sense of wrong answers —
    there is no answer, the tree never compiles — but a real, confirmed
    divergence from decider 1's silent "always otherwise": ported here as
    the actual, reproducible behaviour rather than papered over. See
    PORTED.md.
    """
    tree = _tree(
        "empty_ranges",
        {
            "root": CasesRanges(feature="x", conditions=[], strict=False),
            "otherwise": LeafNode(result_idx=0),
        },
        [("root", 0, "otherwise")],
        _output("always"),
    )
    with pytest.raises(ValueError, match="never referenced"):
        tree_module(tree, build_dir=tmp_path)


def test_cases_string_match_empty_conditions_returns_otherwise(tmp_path):
    """Same finding as `test_cases_ranges_empty_conditions_returns_otherwise`,
    for `CasesStringMatch`."""
    tree = _tree(
        "empty_strmatch",
        {
            "root": CasesStringMatch(feature="s", match_type="exact", conditions=[]),
            "otherwise": LeafNode(result_idx=0),
        },
        [("root", 0, "otherwise")],
        _output("always"),
    )
    with pytest.raises(ValueError, match="never referenced"):
        tree_module(tree, build_dir=tmp_path)


def test_cases_isin_empty_conditions_returns_otherwise(tmp_path):
    """Same finding, for `CasesIsIn`."""
    tree = _tree(
        "empty_isin",
        {
            "root": CasesIsIn(feature="x", conditions=[]),
            "otherwise": LeafNode(result_idx=0),
        },
        [("root", 0, "otherwise")],
        _output("always"),
    )
    with pytest.raises(ValueError, match="never referenced"):
        tree_module(tree, build_dir=tmp_path)


def test_composite_rule_empty_conditions_evaluates_false(tmp_path):
    """decider 1: `CompositeRule(op=AND, conditions=[])` is a legal
    construction that always evaluates False (vacuous AND, defined that way
    by decider 1's implementation rather than by boolean logic, which would
    call an empty AND vacuously true).

    decider2's `CompositeNode`/`CompositeCondition` reject this at the
    SCHEMA layer instead — "composite node needs at least one condition"
    (`trees/schema.py`'s own `_validate_conditions`) — which is actually a
    clearer place to catch it than `CasesRanges`'s later, more confusing
    build-time failure above. Ported as the actual, deliberate decider2
    behaviour: a `pydantic.ValidationError` at construction, not a
    silently-always-False rule.
    """
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="at least one condition"):
        CompositeNode(op=TLogicOp.AND, conditions=[])


# ---------------------------------------------------------------------------
# required_params() — decider 1's get_required_parameters(), same idea
# ---------------------------------------------------------------------------


def test_cases_ranges_get_required_parameters_with_inputref_bounds():
    """decider 1: `get_required_parameters()` collects `InputRef` keys from
    condition bounds. decider2's equivalent method is `required_params()`
    (`schema.py`, every node type)."""
    node = CasesRanges(
        feature="x",
        conditions=[RangeCondition(min=InputRef(key="lo"), max=InputRef(key="hi"))],
        strict=False,
    )
    params = node.required_params()
    assert "lo" in params
    assert "hi" in params


def test_cases_string_match_get_required_parameters_with_inputref():
    """decider 1: `get_required_parameters()` collects `InputRef` keys from
    patterns."""
    node = CasesStringMatch(
        feature="s",
        match_type="exact",
        conditions=[StringMatchCondition(patterns=[InputRef(key="pat")])],
    )
    assert "pat" in node.required_params()


def test_cases_isin_get_required_parameters_with_inputref():
    """decider 1: `get_required_parameters()` collects an `InputRef` values
    key."""
    node = CasesIsIn(feature="x", conditions=[IsInCondition(values=InputRef(key="allowed"))])
    assert "allowed" in node.required_params()
