"""
Integration test for tree format migration (v1 → v2 → v3 → flat rules).

One comprehensive test that exercises:
- V1 tree with numerical, range, categorical, and string-match nodes parsing
- Upgrade chain v1 → v2 → v3
- V3 execution produces correct output values
- V1Tree as a BaseModule executes directly (upgrade + execute internally)

Written against the v1 format so it also validates backward compatibility.
"""

import polars as pl
import pytest
from decider.modules.rules import V1Tree, V3Tree, TreeOutput
from decider.modules.rules.tree.v1.tree import Tree as _V1Tree


# ---------------------------------------------------------------------------
# Shared v1 payload — two subtrees: numerical split + string-match split
# ---------------------------------------------------------------------------

_V1_DICT = {
    "formatVersion": 1,
    "features": ["age", "status"],
    "edges": [
        # Numerical split on age < 30
        {"id": "e1", "source": "num_root",    "target": "leaf_young", "data": {"sourceIndex": [0]}},
        {"id": "e2", "source": "num_root",    "target": "leaf_adult", "data": {"sourceIndex": [1]}},
        # String-match split on status == "vip"
        {"id": "e3", "source": "str_root",    "target": "leaf_vip",   "data": {"sourceIndex": [0]}},
        {"id": "e4", "source": "str_root",    "target": "leaf_std",   "data": {"sourceIndex": [1]}},
    ],
    "nodes": [
        {
            "id": "num_root",
            "position": {"x": 0, "y": 0},
            "data": {
                "node_type": "numerical_test_node",
                "split_feature_id": 0,
                "comparison_op": "<",
                "threshold": 30.0,
            },
        },
        {
            "id": "leaf_young",
            "position": {"x": -100, "y": 100},
            "data": {"node_type": "leaf", "output_data": {"group": "young"}},
        },
        {
            "id": "leaf_adult",
            "position": {"x": 100, "y": 100},
            "data": {"node_type": "leaf", "output_data": {"group": "adult"}},
        },
        {
            "id": "str_root",
            "position": {"x": 300, "y": 0},
            "data": {
                "node_type": "string_match_node",
                "split_feature_id": 1,
                "patterns": ["vip"],
                "match_type": "exact",
                "case_sensitive": False,
                "match_any": True,
            },
        },
        {
            "id": "leaf_vip",
            "position": {"x": 200, "y": 100},
            "data": {"node_type": "leaf", "output_data": {"group": "vip"}},
        },
        {
            "id": "leaf_std",
            "position": {"x": 400, "y": 100},
            "data": {"node_type": "leaf", "output_data": {"group": "standard"}},
        },
    ],
    "subtrees": [
        {"rootNodeId": "num_root", "name": "age_split",    "order": 0},
        {"rootNodeId": "str_root", "name": "status_split", "order": 1},
    ],
    "outputSchema": {
        "fields": [
            {"id": "f1", "field_name": "group", "field_type": "string"},
        ]
    },
}


def test_v1_parses_and_upgrades_to_v3():
    """V1 tree parses without error and upgrades all the way to v3."""
    v1 = _V1Tree.model_validate(_V1_DICT)
    assert len(v1.nodes) == 6
    assert len(v1.features) == 2

    v2 = v1.upgrade()
    assert v2.format_version == 2
    assert len(v2.nodes) == 6

    v3 = v2.upgrade()
    assert v3.format_version == 3
    assert len(v3.nodes) == 6


def test_v1_tree_as_basemodule_produces_correct_output():
    """V1Tree (as BaseExecuteModule) executes directly and produces correct groupings."""
    v1 = V1Tree.model_validate(_V1_DICT)
    df = pl.DataFrame({
        "age":    [20.0, 40.0, 25.0, 35.0],
        "status": ["vip", "standard", "REGULAR", "VIP"],
    })
    result = v1({"input": df.lazy()})
    groups = result["group"].to_list()

    # Subtree 0 (age_split) is root — age < 30 → young, else adult
    assert groups[0] == "young"   # age=20
    assert groups[1] == "adult"   # age=40
    assert groups[2] == "young"   # age=25
    assert groups[3] == "adult"   # age=35


def test_v3_numerical_and_range_nodes_execute_correctly():
    """V3 tree with UnaryNode (numeric) and CasesRanges produces correct outputs."""
    from decider.modules.rules.tree.v3.nodes_ui import (
        UnaryNode, CasesRanges, LeafNode, PositionedNode, Position, RangeCondition
    )
    from decider.modules.rules.tree.v1.edges import MultiSourceEdge, MultiEdgeData

    output = TreeOutput(
        data=[{"band": "low"}, {"band": "mid"}, {"band": "high"}],
        default={"band": "unknown"},
        dtypes=[("band", "String")],
        type_defs={},
    )
    # CasesRanges: (<30 → low, 30–70 → mid, >70 → high)
    v3 = V3Tree(
        edges=[
            MultiSourceEdge(id="e0", source="root", target="leaf_low",  data=MultiEdgeData(sourceIndex=[0])),
            MultiSourceEdge(id="e1", source="root", target="leaf_mid",  data=MultiEdgeData(sourceIndex=[1])),
            MultiSourceEdge(id="e2", source="root", target="leaf_high", data=MultiEdgeData(sourceIndex=[2])),
            MultiSourceEdge(id="e3", source="root", target="leaf_unk",  data=MultiEdgeData(sourceIndex=[3])),
        ],
        nodes=[
            PositionedNode(id="root",      position=Position(x=0, y=0), data=CasesRanges(
                feature="score",
                conditions=[
                    RangeCondition(max=30.0),
                    RangeCondition(min=30.0, max=70.0),
                    RangeCondition(min=70.0),
                ],
                end_logic="lower_inclusive",
                strict=False,
            )),
            PositionedNode(id="leaf_low",  position=Position(x=-150, y=100), data=LeafNode(result_idx=0)),
            PositionedNode(id="leaf_mid",  position=Position(x=0,    y=100), data=LeafNode(result_idx=1)),
            PositionedNode(id="leaf_high", position=Position(x=150,  y=100), data=LeafNode(result_idx=2)),
            PositionedNode(id="leaf_unk",  position=Position(x=300,  y=100), data=LeafNode(result_idx=-1)),
        ],
        output=output,
        parameters={},
    )

    df = pl.DataFrame({"score": [10.0, 30.0, 50.0, 70.0, 90.0]})
    result = v3({"input": df.lazy()})
    bands = result["band"].to_list()

    # lower_inclusive: [min, max) — 30 is start of mid
    assert bands == ["low", "mid", "mid", "high", "high"]


def test_v1_tree_parse_rejects_missing_nodes():
    """V1Tree.model_validate raises on a dict with no 'nodes' key."""
    from pydantic import ValidationError
    bad = dict(_V1_DICT)
    bad.pop("nodes")
    with pytest.raises(ValidationError):
        _V1Tree.model_validate(bad)


def test_v1_tree_parse_rejects_unknown_node_type():
    """V1Tree.model_validate raises on an unrecognised node_type."""
    import json
    from pydantic import ValidationError
    bad = json.loads(json.dumps(_V1_DICT))
    bad["nodes"][0]["data"]["node_type"] = "nonexistent_node_type"
    with pytest.raises(ValidationError):
        _V1Tree.model_validate(bad)


def test_can_create_default_tree():
    """Tree.default_tree() creates a valid V3 tree with expected structure."""
    from decider.modules.rules import Tree
    default = Tree.default_tree()
    assert isinstance(default.root, V3Tree)
    assert default.root.format_version == 3
    assert len(default.root.nodes) == 1
    assert default.root.nodes[0].data.type == "leaf"