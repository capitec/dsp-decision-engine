"""Ported from decider 1: `tests/rules/test_conditions.py` (19 tests).

Migration-conformance suite (see `decider2/tests/PORTED.md` for the full
ledger). Every assertion below is decider 1's own — same inputs, same
expected outputs — translated only where the *vocabulary* differs
mechanically: decider 1 nests rules (`then=`/`otherwise=` on the rule
object itself); decider2 is a node/edge graph (`Tree(nodes=..., edges=...)`,
`MultiSourceEdge(source, target, data=MultiEdgeData(sourceIndex=[...]))`).
The `_tree()` helper below removes the edge-list boilerplate that
difference otherwise adds to every fixture, but the graph it builds is
exactly the one decider 1's nested rule described.

Four tests could not port unchanged — see each docstring and PORTED.md:
  * `test_null_and_boolean_operators` — its `is_null`/`is_not_null` half
    rests on `UnaryIsNull`, a KNOWN GAP (decider2 declares null policy in
    the signature instead, doc 03 §1). Its `is_true`/`is_false` half ports.
  * `test_string_match_types` — only `match_type="exact"` survives a
    kernel (doc 05 §1.5); the other four are ported as `UnsupportedInKernel`
    assertions, which is what decider2 actually does with them.
  * `test_string_match_case_insensitive_and_trim` — same reason, both cases
    become `UnsupportedInKernel` assertions.
  * `test_special_numeric_values_through_range_rules` — its `inf`/`-inf`/
    `nan` cases port unchanged; its `None` case does not — see that test's
    docstring for the routing divergence this surfaces.

Three tests rest on `PrioritizedFlatRuleModule`/`PrioritizationMode`, a
KNOWN GAP with no decider2 equivalent (a `Tree` is one rule; decider2 has
nothing in `trees`/`tables` that chains several with first-match-wins or
independent `all`-mode evaluation). They are recorded in PORTED.md, not
ported:
  * test_prioritized_first_match_wins
  * test_prioritized_falls_back_to_default
  * test_prioritized_module_with_parameters
"""
from __future__ import annotations

import polars as pl
import pytest

from decider2 import flow
from decider2.trees import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    CompositeCondition,
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
    UnaryBetween,
    UnaryEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryIsFalse,
    UnaryIsIn,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryLessThanEqual,
    UnaryNode,
    UnaryNotEqual,
    UnaryStringMatch,
    UnsupportedInKernel,
    tree_module,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output(*labels: str, default: str = "default") -> TreeOutput:
    """decider 1's `_output` helper, same shape."""
    return TreeOutput(
        data=[{"r": lbl} for lbl in labels],
        default={"r": default},
        dtypes=[("r", "String")],
    )


def _tree(name: str, nodes: dict, edges: list[tuple[str, int, str]], output: TreeOutput) -> Tree:
    """A `{node_id: NodeData}` map plus `(source, sourceIndex, target)`
    triples, assembled into decider2's `Tree(nodes=, edges=)` graph — the
    edge-based shape every fixture below needs in place of decider 1's
    nested `then=`/`otherwise=`."""
    positioned = [PositionedNode(id=nid, data=data) for nid, data in nodes.items()]
    by_pair: dict[tuple[str, str], list[int]] = {}
    for src, idx, tgt in edges:
        by_pair.setdefault((src, tgt), []).append(idx)
    built_edges = [
        MultiSourceEdge(source=s, target=t, data=MultiEdgeData(sourceIndex=idxs))
        for (s, t), idxs in by_pair.items()
    ]
    return Tree(name=name, nodes=positioned, edges=built_edges, output=output)


def _run(tree: Tree, frame: pl.DataFrame, *, name: str, tmp_path, params=None) -> list:
    built = tree_module(tree, name=name, build_dir=tmp_path, params=params)
    return built.decode(flow(built.module).apply(frame))["r"].to_list()


def _unary(name: str, condition, labels: tuple[str, str], *, default: str | None = None) -> Tree:
    """then=idx0, otherwise=idx1 (or the default row when `default` is set)."""
    otherwise = LeafNode(result_idx=-1 if default is not None else 1)
    output = _output(*labels, default=default) if default is not None else _output(*labels)
    return _tree(
        name,
        {"root": UnaryNode(condition=condition), "then": LeafNode(result_idx=0), "otherwise": otherwise},
        [("root", 0, "then"), ("root", 1, "otherwise")],
        output,
    )


# ---------------------------------------------------------------------------
# Unary numeric operators
# ---------------------------------------------------------------------------


def test_all_numeric_comparison_operators(tmp_path):
    """decider 1: all six comparison operators route correctly for boundary
    and non-boundary values."""
    df = pl.DataFrame({"x": [5.0, 10.0, 15.0]})
    assert _run(_unary("lt", UnaryLessThan(feature="x", threshold=10.0), ("low", "high")), df, name="lt", tmp_path=tmp_path) == ["low", "high", "high"]
    assert _run(_unary("le", UnaryLessThanEqual(feature="x", threshold=10.0), ("low", "high")), df, name="le", tmp_path=tmp_path) == ["low", "low", "high"]
    assert _run(_unary("eq", UnaryEqual(feature="x", threshold=10.0), ("yes", "no")), df, name="eq", tmp_path=tmp_path) == ["no", "yes", "no"]
    assert _run(_unary("gt", UnaryGreaterThan(feature="x", threshold=10.0), ("high", "low")), df, name="gt", tmp_path=tmp_path) == ["low", "low", "high"]
    assert _run(_unary("ge", UnaryGreaterThanEqual(feature="x", threshold=10.0), ("high", "low")), df, name="ge", tmp_path=tmp_path) == ["low", "high", "high"]
    assert _run(_unary("ne", UnaryNotEqual(feature="x", threshold=10.0), ("ne", "eq")), df, name="ne", tmp_path=tmp_path) == ["ne", "eq", "ne"]


def test_between_variants(tmp_path):
    """decider 1: Between with min-only, max-only, and both bounds."""
    df = pl.DataFrame({"x": [5.0, 15.0, 25.0, 35.0]})

    both = _unary("both", UnaryBetween(feature="x", min=10.0, max=20.0), ("yes",), default="no")
    assert _run(both, df, name="both", tmp_path=tmp_path) == ["no", "yes", "no", "no"]

    min_only = _unary("min_only", UnaryBetween(feature="x", min=20.0), ("yes",), default="no")
    assert _run(min_only, df, name="min_only", tmp_path=tmp_path) == ["no", "no", "yes", "yes"]

    max_only = _unary("max_only", UnaryBetween(feature="x", max=10.0), ("yes",), default="no")
    assert _run(max_only, df, name="max_only", tmp_path=tmp_path) == ["yes", "no", "no", "no"]


def test_null_and_boolean_operators(tmp_path):
    """decider 1 tests `is_null`/`is_not_null` AND `is_true`/`is_false` in
    one test. Only the boolean half ports: `UnaryIsNull`/`UnaryIsNotNull`
    are a KNOWN GAP — decider2 has no null-check condition node at all,
    because null policy is declared in the signature instead (doc 03 §1's
    four tiers), not tested for per-node. See PORTED.md.
    """
    df_bool = pl.DataFrame({"flag": [True, False, True]})
    out2_true = _unary("is_true", UnaryIsTrue(feature="flag"), ("T", "F"))
    assert _run(out2_true, df_bool, name="is_true", tmp_path=tmp_path) == ["T", "F", "T"]

    is_false = _tree(
        "is_false",
        {
            "root": UnaryNode(condition=UnaryIsFalse(feature="flag")),
            "then": LeafNode(result_idx=1),
            "otherwise": LeafNode(result_idx=0),
        },
        [("root", 0, "then"), ("root", 1, "otherwise")],
        _output("T", "F", default="?"),
    )
    assert _run(is_false, df_bool, name="is_false", tmp_path=tmp_path) == ["T", "F", "T"]


# ---------------------------------------------------------------------------
# String conditions
# ---------------------------------------------------------------------------


def test_string_match_types(tmp_path):
    """decider 1: exact, contains, starts_with, ends_with, regex all route
    correctly.

    Only `exact` survives into a compiled kernel (doc 05 §1.5 — a string is
    an int32 dictionary code there, and a code comparison cannot express a
    prefix, a suffix, a substring or a regex). `exact` is ported with
    decider 1's own expected answer; the other four are ported as
    `UnsupportedInKernel` assertions — decider2's actual, deliberate
    behaviour for exactly these configs, naming the frame-tier route.
    """
    df = pl.DataFrame({"s": ["hello world", "world", "hello", "goodbye"]})

    exact = _unary("exact", UnaryStringMatch(feature="s", patterns=["world"], match_type="exact"), ("match",), default="no")
    assert _run(exact, df, name="exact", tmp_path=tmp_path) == ["no", "match", "no", "no"]

    for match_type in ("contains", "starts_with", "ends_with", "regex"):
        tree = _unary(
            f"mt_{match_type}",
            UnaryStringMatch(feature="s", patterns=["ello" if match_type == "contains" else "hello"], match_type=match_type),
            ("match",),
            default="no",
        )
        with pytest.raises(UnsupportedInKernel, match="frame tier"):
            tree_module(tree, name=f"mt_{match_type}", build_dir=tmp_path)


def test_string_match_case_insensitive_and_trim(tmp_path):
    """decider 1: case-insensitive and whitespace-trimmed matching both
    route correctly. Neither survives a kernel (doc 05 §1.5 — no case
    folding or stripping inside nopython); both are ported as
    `UnsupportedInKernel` assertions, decider2's actual behaviour.
    """
    ci = _unary(
        "ci",
        UnaryStringMatch(feature="s", patterns=["hello"], match_type="exact", case_sensitive=False),
        ("match",),
        default="no",
    )
    with pytest.raises(UnsupportedInKernel, match="to_lowercase"):
        tree_module(ci, name="ci", build_dir=tmp_path)

    trim = _unary(
        "trim",
        UnaryStringMatch(feature="s", patterns=["HELLO"], match_type="exact", case_sensitive=True, trim_whitespace=True),
        ("match",),
        default="no",
    )
    with pytest.raises(UnsupportedInKernel, match="strip_chars"):
        tree_module(trim, name="trim", build_dir=tmp_path)


# ---------------------------------------------------------------------------
# Cases (multi-way branching)
# ---------------------------------------------------------------------------


def test_cases_ranges_lower_and_upper_inclusive(tmp_path):
    """decider 1: range cases with lower_inclusive and upper_inclusive end
    logic."""
    df = pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]})
    out = _output("low", "mid", "high", default="none")

    def _cases(end_logic: str) -> Tree:
        return _tree(
            f"cases_{end_logic}",
            {
                "root": CasesRanges(
                    feature="score",
                    conditions=[
                        RangeCondition(max=30.0),
                        RangeCondition(min=30.0, max=70.0),
                        RangeCondition(min=70.0),
                    ],
                    end_logic=end_logic,
                    strict=False,
                ),
                "l0": LeafNode(result_idx=0),
                "l1": LeafNode(result_idx=1),
                "l2": LeafNode(result_idx=2),
                "l3": LeafNode(result_idx=-1),
            },
            [("root", 0, "l0"), ("root", 1, "l1"), ("root", 2, "l2"), ("root", 3, "l3")],
            out,
        )

    r_li = _run(_cases("lower_inclusive"), df, name="li", tmp_path=tmp_path)
    assert r_li[1] == "mid"   # score=30 -> mid (inclusive lower)
    assert r_li[3] == "high"  # score=70 -> high (inclusive lower)

    r_ui = _run(_cases("upper_inclusive"), df, name="ui", tmp_path=tmp_path)
    assert r_ui[1] == "low"   # score=30 -> low (inclusive upper)
    assert r_ui[3] == "mid"   # score=70 -> mid (inclusive upper)


def test_cases_isin(tmp_path):
    """decider 1: IsIn cases route numeric categoricals to the right
    branch."""
    out = _output("grp1", "grp2", default="other")
    tree = _tree(
        "isin",
        {
            "root": CasesIsIn(
                feature="code",
                conditions=[IsInCondition(values=[1, 2]), IsInCondition(values=[3, 4])],
            ),
            "l0": LeafNode(result_idx=0),
            "l1": LeafNode(result_idx=1),
            "l2": LeafNode(result_idx=-1),
        },
        [("root", 0, "l0"), ("root", 1, "l1"), ("root", 2, "l2")],
        out,
    )
    assert _run(tree, pl.DataFrame({"code": [1, 3, 5]}), name="isin", tmp_path=tmp_path) == ["grp1", "grp2", "other"]


def test_cases_string_match(tmp_path):
    """decider 1 groups strings by `match_type="starts_with"` prefix. No
    exact-match config produces the same groupings (the whole point of the
    test is prefix grouping), so it is ported as `UnsupportedInKernel` —
    decider2's actual, deliberate refusal of this config, per doc 05 §1.5.
    """
    tree = _tree(
        "strmatch",
        {
            "root": CasesStringMatch(
                feature="s",
                match_type="starts_with",
                conditions=[
                    StringMatchCondition(patterns=["a"]),
                    StringMatchCondition(patterns=["b", "c"]),
                ],
            ),
            "l0": LeafNode(result_idx=0),
            "l1": LeafNode(result_idx=1),
            "l2": LeafNode(result_idx=-1),
        },
        [("root", 0, "l0"), ("root", 1, "l1"), ("root", 2, "l2")],
        _output("A", "B", default="other"),
    )
    with pytest.raises(UnsupportedInKernel, match="frame tier"):
        tree_module(tree, name="strmatch", build_dir=tmp_path)


# ---------------------------------------------------------------------------
# Composite conditions
# ---------------------------------------------------------------------------


def test_composite_and_or_not(tmp_path):
    """decider 1: AND, OR, NOT composites all produce correct results.

    This is where porting found a real decider2 bug (fixed in
    `decider2/trees/codegen.py`): two sibling conditions inside one
    `CompositeNode`'s `conditions` list previously shared ONE threshold
    `param()`, because their role name (`f"{node_id}_thr"`) did not vary
    with position in the list — only with the node, which is the same for
    every sibling. `x > 5 and x < 10` silently compiled to `x > root_thr
    and x < root_thr` (both reading the SAME argument), which is never
    satisfiable, so every AND with two same-shaped conditions silently
    always returned "no". See PORTED.md for the full writeup; this test is
    the one that caught it.
    """
    df = pl.DataFrame({"x": [3.0, 7.0, 12.0]})
    out = _output("yes", default="no")

    and_tree = _tree(
        "and",
        {
            "root": CompositeNode(
                op=TLogicOp.AND,
                conditions=[
                    UnaryGreaterThan(feature="x", threshold=5.0),
                    UnaryLessThan(feature="x", threshold=10.0),
                ],
            ),
            "yes": LeafNode(result_idx=0),
            "no": LeafNode(result_idx=-1),
        },
        [("root", 0, "yes"), ("root", 1, "no")],
        out,
    )
    assert _run(and_tree, df, name="and", tmp_path=tmp_path) == ["no", "yes", "no"]

    or_tree = _tree(
        "or",
        {
            "root": CompositeNode(
                op=TLogicOp.OR,
                conditions=[
                    UnaryLessThan(feature="x", threshold=5.0),
                    UnaryGreaterThan(feature="x", threshold=10.0),
                ],
            ),
            "yes": LeafNode(result_idx=0),
            "no": LeafNode(result_idx=-1),
        },
        [("root", 0, "yes"), ("root", 1, "no")],
        out,
    )
    assert _run(or_tree, df, name="or", tmp_path=tmp_path) == ["yes", "no", "yes"]

    not_tree = _tree(
        "not",
        {
            "root": CompositeNode(op=TLogicOp.NOT, conditions=[UnaryGreaterThan(feature="x", threshold=5.0)]),
            "yes": LeafNode(result_idx=0),
            "no": LeafNode(result_idx=-1),
        },
        [("root", 0, "yes"), ("root", 1, "no")],
        out,
    )
    assert _run(not_tree, df, name="not", tmp_path=tmp_path) == ["yes", "no", "no"]


def test_nested_composite(tmp_path):
    """decider 1: (x > 0 AND x < 10) OR x > 20, using a nested
    `CompositeCondition` inside a `CompositeNode`."""
    df = pl.DataFrame({"x": [-5.0, 5.0, 15.0, 25.0]})
    out = _output("yes", default="no")

    inner = CompositeCondition(
        op=TLogicOp.AND,
        conditions=[
            UnaryGreaterThan(feature="x", threshold=0.0),
            UnaryLessThan(feature="x", threshold=10.0),
        ],
    )
    tree = _tree(
        "nested",
        {
            "root": CompositeNode(op=TLogicOp.OR, conditions=[inner, UnaryGreaterThan(feature="x", threshold=20.0)]),
            "yes": LeafNode(result_idx=0),
            "no": LeafNode(result_idx=-1),
        },
        [("root", 0, "yes"), ("root", 1, "no")],
        out,
    )
    assert _run(tree, df, name="nested", tmp_path=tmp_path) == ["no", "yes", "no", "yes"]


# ---------------------------------------------------------------------------
# Default / leaf behaviour
# ---------------------------------------------------------------------------


def test_default_returned_when_no_match(tmp_path):
    """decider 1: `result_idx=-1` always returns the default, not an output
    row."""
    df = pl.DataFrame({"x": [100.0, 200.0]})
    tree = _unary("default", UnaryLessThan(feature="x", threshold=0.0), ("match",), default="FALLBACK")
    assert _run(tree, df, name="default", tmp_path=tmp_path) == ["FALLBACK", "FALLBACK"]


def test_correct_output_row_indexed(tmp_path):
    """decider 1: `result_idx` correctly selects a row from the output
    table."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    tree = _tree(
        "rowidx",
        {
            "root": CasesIsIn(
                feature="x",
                conditions=[
                    IsInCondition(values=[1.0]),
                    IsInCondition(values=[2.0]),
                    IsInCondition(values=[3.0]),
                ],
            ),
            "l0": LeafNode(result_idx=0),
            "l1": LeafNode(result_idx=1),
            "l2": LeafNode(result_idx=2),
            "l3": LeafNode(result_idx=-1),
        },
        [("root", 0, "l0"), ("root", 1, "l1"), ("root", 2, "l2"), ("root", 3, "l3")],
        _output("row0", "row1", "row2", default="none"),
    )
    assert _run(tree, df, name="rowidx", tmp_path=tmp_path) == ["row0", "row1", "row2"]


# ---------------------------------------------------------------------------
# Special numeric values
# ---------------------------------------------------------------------------


def test_special_numeric_values_through_range_rules(tmp_path):
    """decider 1: inf, -inf, nan, None all handled correctly through
    numeric range conditions — only finite positives match `[0, 100)`.

    The `inf`/`-inf`/`nan` cases port unchanged: none of them is a polars
    *null*, so all three reach the compiled kernel as ordinary (if
    unusual) float64 values and fail the comparison exactly as decider 1's
    polars expressions did.

    The `None` case does NOT port. decider2 declares a tree feature's
    signature as plain `float` (no `| None`) unless told otherwise, which
    is doc 03 §1 tier 1, `NullPolicy.REQUIRED`: "routed, not raised, by
    default". A `None` in a REQUIRED column never reaches the kernel at
    all — `decider2.boundary.extract_frame` routes that WHOLE ROW to a
    `Decision` (default `Decision.REFER`) before the tree ever runs, which
    is a materially different thing from decider 1's "the comparison reads
    null and returns False". See `test_null_input_is_routed_away_not_evaluated`
    below and PORTED.md for the full writeup, including the
    `_scatter_back` placeholder caveat that makes this untestable through
    `.apply()`'s output frame alone.
    """
    df = pl.DataFrame({"v": pl.Series([1.0, float("inf"), float("-inf"), float("nan")], dtype=pl.Float64)})
    tree = _unary("special", UnaryBetween(feature="v", min=0.0, max=100.0), ("match",), default="no")
    results = _run(tree, df, name="special", tmp_path=tmp_path)
    assert results[0] == "match"  # 1.0 in [0, 100)
    assert results[1] == "no"     # inf not in range
    assert results[2] == "no"     # -inf not in range
    assert results[3] == "no"     # nan never compares true


def test_null_input_is_routed_away_not_evaluated(tmp_path):
    """The divergence `test_special_numeric_values_through_range_rules`
    above documents, made concrete and asserted directly against the real
    API (`decider2.boundary.extract_frame`) rather than against
    `.apply()`'s output frame — `runtime.invoke._scatter_back`'s own
    docstring admits the terminal column's value for a routed-away row is
    presently just a placeholder (`nan` for float, 0 for int64/bool), "not
    a claim that 0/False is the routed row's real answer", so asserting a
    specific `_path` value for that row would be pinning an admitted
    placeholder, not real behaviour.

    What IS real and documented (doc 03 §1, `boundary/nulls.py`): a `None`
    in a `NullPolicy.REQUIRED` column is never handed to the tree at all —
    the row is routed to `Decision.REFER` at extraction. decider 1 instead
    let the null reach the comparison and fall through to `otherwise`. Both
    behaviours agree the row does not match a real bucket; they disagree
    on the mechanism, and only decider2's is externally observable through
    a `Decision` rather than a leaf value.
    """
    from decider2.boundary import extract_frame
    from decider2.types import Decision

    tree = _unary("routed", UnaryBetween(feature="v", min=0.0, max=100.0), ("match",), default="no")
    built = tree_module(tree, name="routed", build_dir=tmp_path)
    pipeline = flow(built.module)

    frame = pl.DataFrame({"v": pl.Series([50.0, None], dtype=pl.Float64)})
    extracted = extract_frame(frame, pipeline.interface.inputs, policy=None)

    assert extracted.routing.mask.tolist() == [False, True]
    assert extracted.routing.column == (None, "v")
    assert extracted.routing.decision is Decision.REFER
    # Only the valid row ever reaches the kernel.
    assert extracted.kernel_frame.height == 1


# ---------------------------------------------------------------------------
# Gap: UnaryIsIn operator
# ---------------------------------------------------------------------------


def test_unary_is_in(tmp_path):
    """decider 1: `UnaryIsIn` routes rows whose value is in the list to
    `then`, others to `otherwise`."""
    df = pl.DataFrame({"code": [1, 2, 3, 4, 5]})
    tree = _unary("uin", UnaryIsIn(feature="code", values=[1, 3, 5]), ("allowed",), default="denied")
    assert _run(tree, df, name="uin", tmp_path=tmp_path) == ["allowed", "denied", "allowed", "denied", "allowed"]


# ---------------------------------------------------------------------------
# InputRef bounds inside Cases nodes
# ---------------------------------------------------------------------------


def test_cases_ranges_inputref_bounds(tmp_path):
    """decider 1: `CasesRanges` where min/max come from `InputRef`
    parameters, not static values.

    Adapted: decider 1's `FlatRuleModule(parameters={"lo": ParameterInfo(
    default_value=20.0), ...})` gave the InputRef its default at the
    *module* boundary. decider2 has no `ParameterInfo` — an `InputRef`
    becomes a kernel `param()` with no default of its own (`schema.py`
    divergence 3), so the default has to be supplied where a `param()`'s
    default is ever supplied: `tree_module(tree, params={...})`, doc 03
    §4.3's pre-bind. Same two values, same expected answer.
    """
    df = pl.DataFrame({
        "score": [10.0, 40.0, 80.0],
    })
    tree = _tree(
        "inputref_bounds",
        {
            "root": CasesRanges(
                feature="score",
                conditions=[
                    RangeCondition(max=InputRef(key="lo")),
                    RangeCondition(min=InputRef(key="lo"), max=InputRef(key="hi")),
                ],
                end_logic="lower_inclusive",
                strict=False,
            ),
            "l0": LeafNode(result_idx=0),
            "l1": LeafNode(result_idx=1),
            "l2": LeafNode(result_idx=-1),
        },
        [("root", 0, "l0"), ("root", 1, "l1"), ("root", 2, "l2")],
        _output("low", "mid", default="high"),
    )
    result = _run(tree, df, name="inputref_bounds", tmp_path=tmp_path, params={"lo": 20.0, "hi": 60.0})
    assert result == ["low", "mid", "high"]


def test_cases_string_match_inputref_pattern(tmp_path):
    """decider 1: `CasesStringMatch` where a pattern comes from an
    `InputRef` parameter. `match_type="exact"` (the default), so this
    ports without hitting the string-kernel restriction."""
    df = pl.DataFrame({"status": ["gold", "silver", "bronze"]})
    tree = _tree(
        "inputref_pattern",
        {
            "root": CasesStringMatch(
                feature="status",
                match_type="exact",
                conditions=[StringMatchCondition(patterns=[InputRef(key="vip_tier")])],
            ),
            "vip": LeafNode(result_idx=0),
            "standard": LeafNode(result_idx=-1),
        },
        [("root", 0, "vip"), ("root", 1, "standard")],
        _output("vip", default="standard"),
    )
    result = _run(tree, df, name="inputref_pattern", tmp_path=tmp_path, params={"vip_tier": "gold"})
    assert result == ["vip", "standard", "standard"]
