"""Ported from decider 1: `tests/rules/test_tree_migration.py` (6 tests).

Migration-conformance suite — see `decider2/tests/PORTED.md`.

Two of six rest on decider 1's v1 tree format (`decider.modules.rules.tree.
v1.tree.Tree`, `V1Tree`, the v1->v2->v3 upgrade chain) and decider 1's
`Tree.default_tree()` factory. Both are KNOWN GAPS: doc 08 §3's decision was
that v2 ports via `from_v2_range()` (a straight rewrite into a v3
`CasesRanges`, already pinned in `test_trees.py`), not that the full v1
upgrade chain gets reproduced — decider2 has no v1 parser, no `V1Tree`, no
`.upgrade()`, and no default-tree factory. `test_v1_tree_as_basemodule_
produces_correct_output` is the one exception: its own assertion only
exercises the plain numeric subtree, which is expressible directly as a v3
`Tree` with the same answer, so it ports that way (option (a) — same
config's *intent*, not the same document).
"""
from __future__ import annotations

import polars as pl
import pydantic
import pytest

from decider2 import flow
from decider2.trees import (
    CasesRanges,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    Position,
    PositionedNode,
    RangeCondition,
    Tree,
    TreeOutput,
    UnaryLessThan,
    UnaryNode,
    tree_module,
)

# test_v1_parses_and_upgrades_to_v3 does not port — decider2 has no v1
# parser or upgrade chain (KNOWN GAP). Recorded in PORTED.md.


def test_v1_tree_as_basemodule_produces_correct_output(tmp_path):
    """decider 1's `_V1_DICT` fixture has two independent subtrees (a
    numeric age split and a string-match status split); decider 1's own
    test asserts only the numeric one, since that is the v1 document's
    selected root (`subtrees[0]`). Ported directly as the equivalent v3
    `Tree` decider 1's own `to_v3_node()`/upgrade chain would produce for
    that subtree — same structure, same answer, decider 1's exact
    assertion.
    """
    tree = Tree(
        name="age_split",
        edges=[
            MultiSourceEdge(source="num_root", target="leaf_young", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(source="num_root", target="leaf_adult", data=MultiEdgeData(sourceIndex=[1])),
        ],
        nodes=[
            PositionedNode(id="num_root", data=UnaryNode(condition=UnaryLessThan(feature="age", threshold=30.0))),
            PositionedNode(id="leaf_young", data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_adult", data=LeafNode(result_idx=1)),
        ],
        output=TreeOutput(
            data=[{"group": "young"}, {"group": "adult"}],
            default={"group": "unknown"},
            dtypes=[("group", "String")],
        ),
    )
    built = tree_module(tree, build_dir=tmp_path)
    df = pl.DataFrame({"age": [20.0, 40.0, 25.0, 35.0]})
    out = built.decode(flow(built.module).apply(df))
    groups = out["group"].to_list()

    # decider 1's own assertions, verbatim: "Subtree 0 (age_split) is root
    # -- age < 30 -> young, else adult".
    assert groups[0] == "young"   # age=20
    assert groups[1] == "adult"   # age=40
    assert groups[2] == "young"   # age=25
    assert groups[3] == "adult"   # age=35


def test_v3_numerical_and_range_nodes_execute_correctly(tmp_path):
    """decider 1: V3 tree with a numeric `UnaryNode` and a `CasesRanges`
    produces correct outputs. decider 1's own comment on the expected
    answer: "lower_inclusive: [min, max) -- 30 is start of mid"."""
    output = TreeOutput(
        data=[{"band": "low"}, {"band": "mid"}, {"band": "high"}],
        default={"band": "unknown"},
        dtypes=[("band", "String")],
    )
    tree = Tree(
        name="band",
        edges=[
            MultiSourceEdge(id="e0", source="root", target="leaf_low", data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(id="e1", source="root", target="leaf_mid", data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(id="e2", source="root", target="leaf_high", data=MultiEdgeData(sourceIndex=[2])),
            MultiSourceEdge(id="e3", source="root", target="leaf_unk", data=MultiEdgeData(sourceIndex=[3])),
        ],
        nodes=[
            PositionedNode(id="root", position=Position(x=0, y=0), data=CasesRanges(
                feature="score",
                conditions=[
                    RangeCondition(max=30.0),
                    RangeCondition(min=30.0, max=70.0),
                    RangeCondition(min=70.0),
                ],
                end_logic="lower_inclusive",
                strict=False,
            )),
            PositionedNode(id="leaf_low", position=Position(x=-150, y=100), data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_mid", position=Position(x=0, y=100), data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_high", position=Position(x=150, y=100), data=LeafNode(result_idx=2)),
            PositionedNode(id="leaf_unk", position=Position(x=300, y=100), data=LeafNode(result_idx=-1)),
        ],
        output=output,
    )
    built = tree_module(tree, build_dir=tmp_path)
    df = pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]})
    out = built.decode(flow(built.module).apply(df))

    assert out["band"].to_list() == ["low", "mid", "mid", "high", "high"]


def test_tree_parse_rejects_missing_nodes():
    """decider 1: `V1Tree.model_validate` raises on a dict with no 'nodes'
    key. Adapted to decider2's own `Tree` (v1 does not exist here): the
    same absence of a required field raises the same
    `pydantic.ValidationError`, for the same reason — `nodes` has no
    default."""
    with pytest.raises(pydantic.ValidationError):
        Tree.model_validate({"name": "x", "edges": []})


def test_tree_parse_rejects_unknown_node_type():
    """decider 1: `V1Tree.model_validate` raises on an unrecognised
    `node_type`. Adapted to decider2's `Tree`: `NodeData`'s discriminator
    (`_node_tag` in `trees/schema.py`) does not match any known `(type,
    op)` tag for a bogus type, so pydantic's union validation fails the
    same way — `ValidationError`, not a silent fallback."""
    bad = {
        "name": "x",
        "nodes": [{"id": "n1", "data": {"type": "bogus_type"}}],
        "edges": [],
    }
    with pytest.raises(pydantic.ValidationError):
        Tree.model_validate(bad)


# test_can_create_default_tree does not port — decider2 has no
# `Tree.default_tree()` factory (decider 1's `Tree` there is a different,
# versioned wrapper type this package never built). KNOWN GAP, recorded in
# PORTED.md.
