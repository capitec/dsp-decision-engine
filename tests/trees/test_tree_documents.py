import pydantic
import pytest

from decider.steps.trees import (
    CasesRanges,
    FlatRuleDocument,
    LeafNode,
    Node,
    PrioritizedFlatRuleDocument,
    RangeCondition,
    Rule,
    Tree,
    TreeDocument,
    TreeOutput,
    UnaryLessThan,
    UnaryNode,
    V3TreeDocument,
    detect_format,
    load_document,
)
from decider.steps.values import ParamRef

OUTPUT = {
    "data": [{"r": "low"}, {"r": "mid"}, {"r": "high"}, {"r": "adult"}],
    "default": {"r": "default"},
    "dtypes": [("r", "String")],
}

# Two levels: age < 30 -> score bands (low / mid / high / default), else adult.
V3_NESTED = {
    "type": "v3-tree",
    "formatVersion": 3,
    "nodes": [
        {"id": "root", "position": {"x": 0, "y": 0},
         "data": {"type": "unary", "condition": {"op": "<", "feature": "age", "threshold": 30.0}}},
        {"id": "buckets", "position": {"x": 10, "y": 50},
         "data": {"type": "cases", "op": "ranges", "feature": "score", "strict": False,
                  "conditions": [{"max": 40.0}, {"min": 40.0, "max": 70.0}, {"min": 70.0}]}},
        {"id": "low", "data": {"type": "leaf", "result_idx": 0}},
        {"id": "mid", "data": {"type": "leaf", "result_idx": 1}},
        {"id": "high", "data": {"type": "leaf", "result_idx": 2}},
        {"id": "none", "data": {"type": "leaf", "result_idx": -1}},
        {"id": "adult", "data": {"type": "leaf", "result_idx": 3}},
    ],
    "edges": [
        {"id": "e0", "source": "root", "target": "buckets", "data": {"sourceIndex": [0]}},
        {"id": "e1", "source": "root", "target": "adult", "data": {"sourceIndex": [1]}},
        {"source": "buckets", "target": "low", "data": {"sourceIndex": [0]}},
        {"source": "buckets", "target": "mid", "data": {"sourceIndex": [1]}},
        {"source": "buckets", "target": "high", "data": {"sourceIndex": [2]}},
        {"source": "buckets", "target": "none", "data": {"sourceIndex": [3]}},
    ],
    "output": OUTPUT,
}

FLAT_NESTED = {
    "type": "flat_rule",
    "rule": {"meta": {}, "rule": {
        "type": "unary", "id": "root",
        "condition": {"op": "<", "feature": "age", "threshold": 30.0},
        "then": {
            "type": "cases", "op": "ranges", "id": "buckets", "feature": "score", "strict": False,
            "conditions": [
                {"when": {"max": 40.0}, "then": 0},
                {"when": {"min": 40.0, "max": 70.0}, "then": 1},
                {"when": {"min": 70.0}, "then": 2},
            ],
            "otherwise": 3,
            "branches": [
                {"type": "leaf", "id": "low", "result_idx": 0},
                {"type": "leaf", "id": "mid", "result_idx": 1},
                {"type": "leaf", "id": "high", "result_idx": 2},
                {"type": "leaf", "id": "none", "result_idx": -1},
            ],
        },
        "otherwise": {"type": "leaf", "id": "adult", "result_idx": 3, "meta": {"position": {"x": 1, "y": 2}}},
    }},
    "output": OUTPUT,
}


def _unary(nid, threshold=30.0):
    return {"id": nid, "data": {"type": "unary", "condition": {"op": "<", "feature": "x", "threshold": threshold}}}


def _leaf(nid, idx=0):
    return {"id": nid, "data": {"type": "leaf", "result_idx": idx}}


def _edge(source, target, *index):
    return {"source": source, "target": target, "data": {"sourceIndex": list(index)}}


def _flat(rule, **extra):
    return {"type": "flat_rule", "rule": {"rule": rule}, "output": OUTPUT, **extra}


def _flat_unary(nid=None, then=None, otherwise=None, threshold=30.0):
    rule = {"type": "unary", "condition": {"op": "<", "feature": "x", "threshold": threshold}}
    if nid:
        rule["id"] = nid
    return {**rule, "then": then, "otherwise": otherwise}


# --- one internal model for both formats -------------------------------------------------


def test_a_flat_rule_and_the_equivalent_v3_tree_normalise_to_equal_trees():
    assert load_document(FLAT_NESTED).to_tree() == load_document(V3_NESTED).to_tree()


def test_the_normalised_tree_holds_nodes_by_id_with_children_by_branch():
    tree = load_document(V3_NESTED).to_tree()
    assert tree.rules == (Rule(root="root"),)
    assert tree.nodes["root"].children == ("buckets", "adult")
    assert tree.nodes["buckets"].children == ("low", "mid", "high", "none")
    assert tree.nodes["low"] == Node(data=LeafNode(result_idx=0))
    assert isinstance(tree.nodes["buckets"].data, CasesRanges)
    assert tree.required_features() == {"age", "score"}


def test_node_ids_inside_node_data_do_not_leak_into_the_tree():
    doc = load_document({"nodes": [{"id": "only", "data": {"type": "leaf", "id": "other", "result_idx": 0}}]})
    assert doc.to_tree().nodes["only"].data.id is None


def test_a_document_round_trips_through_its_dump():
    for source in (V3_NESTED, FLAT_NESTED):
        doc = load_document(source)
        again = load_document(doc.model_dump(by_alias=True))
        assert again.to_tree() == doc.to_tree()
        assert load_document(doc.model_dump()).to_tree() == doc.to_tree()


def test_a_tree_document_is_a_usable_pydantic_field():
    class Holder(pydantic.BaseModel):
        tree: TreeDocument

    assert isinstance(Holder(tree=V3_NESTED).tree, V3TreeDocument)
    assert isinstance(Holder(tree=FLAT_NESTED).tree, FlatRuleDocument)
    assert Holder.model_validate(Holder(tree=FLAT_NESTED).model_dump()).tree.to_tree() == load_document(
        FLAT_NESTED).to_tree()


# --- format detection and deprecated versions --------------------------------------------


@pytest.mark.parametrize(
    "doc, expected",
    [
        ({"formatVersion": 3, "nodes": []}, "v3"),
        ({"format_version": 2, "nodes": []}, "v2"),
        ({"type": "v1-tree", "nodes": []}, "v1"),
        ({"formatVersion": 1, "type": "v1-tree"}, "v1"),
        ({"nodes": {"a": {}}}, "v0"),
        ({"nodes": []}, "v3"),
        ({"type": "flat_rule"}, "flat_rule"),
        ({"type": "prioritized_flat_rule", "formatVersion": 2}, "prioritized_flat_rule"),
    ],
)
def test_the_format_is_detected_from_the_document(doc, expected):
    assert detect_format(doc) == expected


@pytest.mark.parametrize(
    "doc, version",
    [
        ({"formatVersion": 1, "nodes": [], "edges": [], "features": []}, "v1"),
        ({"type": "v1-tree", "nodes": []}, "v1"),
        ({"type": "v2-tree", "formatVersion": 2, "nodes": [], "edges": []}, "v2"),
        ({"format_version": 2, "nodes": []}, "v2"),
        ({"nodes": {"n1": {"node_type": "leaf"}}}, "v0"),
        ({"formatVersion": 7, "nodes": []}, "v7"),
    ],
)
def test_an_unsupported_version_raises_a_deprecation_error_naming_it(doc, version):
    with pytest.raises(pydantic.ValidationError, match=f"{version} tree documents are deprecated"):
        load_document(doc)


def test_decider_old_v1_fixture_is_rejected_as_v1():
    v1 = {
        "formatVersion": 1,
        "features": ["age"],
        "edges": [{"id": "e1", "source": "num_root", "target": "leaf_young", "data": {"sourceIndex": [0]}}],
        "nodes": [
            {"id": "num_root", "position": {"x": 0, "y": 0}, "data": {
                "node_type": "numerical_test_node", "split_feature_id": 0, "comparison_op": "<", "threshold": 30.0}},
            {"id": "leaf_young", "position": {"x": -100, "y": 100},
             "data": {"node_type": "leaf", "output_data": {"group": "young"}}},
        ],
        "subtrees": [{"rootNodeId": "num_root", "name": "age_split", "order": 0}],
        "outputSchema": {"fields": [{"id": "f1", "field_name": "group", "field_type": "string"}]},
    }
    with pytest.raises(pydantic.ValidationError, match="v1 tree documents are deprecated"):
        load_document(v1)


def test_v3_parse_rejects_missing_nodes():
    with pytest.raises(pydantic.ValidationError):
        load_document({"name": "x", "edges": []})


def test_v3_parse_rejects_unknown_node_type():
    with pytest.raises(pydantic.ValidationError):
        load_document({"name": "x", "nodes": [{"id": "n1", "data": {"type": "bogus_type"}}], "edges": []})


# --- v3 structure -------------------------------------------------------------------------


def test_an_unconnected_branch_selects_the_default_row():
    tree = load_document({"nodes": [_unary("root"), _leaf("yes")], "edges": [_edge("root", "yes", 0)]}).to_tree()
    assert tree.nodes["root"].children == ("yes", None)


def test_one_edge_can_carry_several_branches_and_a_bare_index():
    doc = load_document({
        "nodes": [_unary("root"), _leaf("both")],
        "edges": [{"source": "root", "target": "both", "data": {"sourceIndex": 0}},
                  _edge("root", "both", 1)],
    })
    assert doc.to_tree().nodes["root"].children == ("both", "both")


def test_the_first_listed_subtree_is_the_root_and_other_subtrees_are_dropped():
    doc = load_document({
        "nodes": [_unary("a"), _leaf("a_leaf"), _unary("b"), _leaf("b_leaf")],
        "edges": [_edge("a", "a_leaf", 0), _edge("b", "b_leaf", 0)],
        "subtrees": [{"id": "b"}, {"id": "a"}],
    })
    tree = doc.to_tree()
    assert tree.rules == (Rule(root="b"),)
    assert set(tree.nodes) == {"b", "b_leaf"}


def test_without_subtrees_the_first_root_listed_wins():
    doc = load_document({"nodes": [_leaf("a"), _leaf("b")]})
    assert doc.to_tree().rules == (Rule(root="a"),)


@pytest.mark.parametrize(
    "doc, message",
    [
        ({"nodes": []}, "no nodes"),
        ({"nodes": [_leaf("a"), _leaf("a")]}, "duplicate node id"),
        ({"nodes": [_leaf("a")], "edges": [_edge("a", "ghost", 0)]}, "unknown node 'ghost'"),
        ({"nodes": [_unary("a"), _unary("b")], "edges": [_edge("a", "b", 0), _edge("b", "a", 0)]}, "no root"),
        ({"nodes": [_unary("r"), _unary("a"), _unary("b")],
          "edges": [_edge("r", "a", 0), _edge("a", "b", 0), _edge("b", "a", 0)]}, "cycle"),
    ],
)
def test_a_malformed_v3_graph_is_rejected(doc, message):
    with pytest.raises(pydantic.ValidationError, match=message):
        load_document(doc)


def test_a_shared_subtree_is_one_node():
    doc = load_document({
        "nodes": [_unary("r"), _unary("a"), _leaf("shared")],
        "edges": [_edge("r", "a", 0), _edge("r", "shared", 1), _edge("a", "shared", 0, 1)],
    })
    tree = doc.to_tree()
    assert tree.nodes["r"].children == ("a", "shared")
    assert tree.nodes["a"].children == ("shared", "shared")


# --- flat rules ---------------------------------------------------------------------------


def test_flat_nodes_without_ids_get_ids_from_their_path():
    tree = load_document(_flat(_flat_unary(then=_flat_unary(then={"type": "leaf", "result_idx": 0})))).to_tree()
    assert tree.rules == (Rule(root="0"),)
    assert tree.nodes["0"].children == ("0.0", None)
    assert tree.nodes["0.0"].children == ("0.0.0", None)
    assert tree.nodes["0.0.0"].data == LeafNode(result_idx=0)


def test_flat_cases_route_each_condition_to_its_branch_index():
    cases = {
        "type": "cases", "op": "isin", "feature": "code",
        "conditions": [{"when": {"values": [1, 2]}, "then": 1}, {"when": {"values": [3]}, "then": 1},
                       {"when": {"values": [4]}, "then": 0}],
        "otherwise": 2,
        "branches": [{"type": "leaf", "result_idx": 0}, {"type": "leaf", "result_idx": 1},
                     {"type": "leaf", "result_idx": -1}],
    }
    tree = load_document(_flat(cases)).to_tree()
    assert tree.nodes["0"].children == ("0.1", "0.1", "0.0", "0.2")
    assert [c.values for c in tree.nodes["0"].data.conditions] == [[1, 2], [3], [4]]


def test_a_flat_branch_index_out_of_range_is_rejected():
    cases = {"type": "cases", "op": "ranges", "feature": "x", "strict": False,
             "conditions": [{"when": {"max": 1.0}, "then": 3}], "otherwise": 0,
             "branches": [{"type": "leaf"}]}
    with pytest.raises(pydantic.ValidationError, match="out of range"):
        load_document(_flat(cases))


def test_flat_cases_ranges_are_validated_like_v3_ones():
    cases = {"type": "cases", "op": "ranges", "feature": "x",
             "conditions": [{"when": {"max": 10.0}, "then": 0}, {"when": {"min": 20.0}, "then": 0}],
             "otherwise": 0, "branches": [{"type": "leaf"}]}
    with pytest.raises(pydantic.ValidationError, match="not continuous"):
        load_document(_flat(cases))


def test_two_different_flat_nodes_cannot_share_an_id():
    rule = _flat_unary("n", then={"type": "leaf", "id": "n", "result_idx": 0})
    with pytest.raises(pydantic.ValidationError, match="share the id 'n'"):
        load_document(_flat(rule))


def test_a_repeated_identical_flat_subtree_is_one_node():
    leaf = {"type": "leaf", "id": "same", "result_idx": 0}
    tree = load_document(_flat(_flat_unary(then=leaf, otherwise=leaf))).to_tree()
    assert tree.nodes["0"].children == ("same", "same")


def test_prioritized_rules_keep_their_order_names_and_mode():
    doc = load_document({
        "type": "prioritized_flat_rule",
        "mode": "all",
        "rules": [
            {"meta": {"name": "under50"}, "rule": _flat_unary(threshold=50.0, then={"type": "leaf", "result_idx": 0})},
            {"meta": {"name": "under100"}, "rule": _flat_unary(threshold=100.0, then={"type": "leaf", "result_idx": 1})},
        ],
        "output": OUTPUT,
    })
    assert isinstance(doc, PrioritizedFlatRuleDocument)
    tree = doc.to_tree()
    assert tree.mode == "all"
    assert tree.rules == (Rule(root="0", name="under50"), Rule(root="1", name="under100"))
    assert tree.nodes["1"].data.condition.threshold == 100.0


def test_prioritized_rules_default_to_first_match():
    doc = load_document({"type": "prioritized_flat_rule", "rules": [{"rule": {"type": "leaf"}}], "output": OUTPUT})
    assert doc.to_tree().mode == "first_match"


@pytest.mark.parametrize("field", ["output_fn", "post_process_fn", "format_prioritized_fn"])
def test_python_function_references_are_rejected(field):
    doc = {"type": "prioritized_flat_rule", "rules": [{"rule": {"type": "leaf"}}], "output": OUTPUT,
           field: {"module_name": "m", "function_name": "f"}}
    with pytest.raises(pydantic.ValidationError, match=field):
        load_document(doc)


# --- params -------------------------------------------------------------------------------


def test_a_parameters_block_gives_key_refs_their_defaults():
    params = {"thresh": {"type": "Float64", "default_value": 50.0}}
    flat = load_document(_flat(_flat_unary(threshold={"key": "thresh"}), parameters=params)).to_tree()
    assert flat.nodes["0"].data.condition.threshold == ParamRef(param="thresh", default=50.0)
    v3 = load_document({"nodes": [_unary("root", {"key": "thresh"})], "parameters": params}).to_tree()
    assert v3.nodes["root"].data.condition.threshold == ParamRef(param="thresh", default=50.0)


def test_a_parameters_block_reaches_nested_conditions():
    rule = {"type": "composite", "op": "and", "conditions": [
        {"type": "composite", "op": "or", "conditions": [{"op": "between", "feature": "x", "min": {"key": "lo"}}]},
    ]}
    tree = load_document(_flat(rule, parameters={"lo": {"default_value": 1}})).to_tree()
    assert tree.nodes["0"].data.conditions[0].conditions[0].min == ParamRef(param="lo", default=1)


def test_a_parameters_block_leaves_explicit_and_shared_refs_alone():
    params = {"a": {"default_value": 1.0}, "b": {"default_value": 2.0}}
    rule = {"type": "composite", "op": "and", "conditions": [
        {"op": "<", "feature": "x", "threshold": {"param": "a", "default": 9.0}},
        {"op": "<", "feature": "x", "threshold": {"param": "b", "shared": True}},
    ]}
    conditions = load_document(_flat(rule, parameters=params)).to_tree().nodes["0"].data.conditions
    assert conditions[0].threshold == ParamRef(param="a", default=9.0)
    assert conditions[1].threshold == ParamRef(param="b", shared=True)


# --- the Tree model itself ----------------------------------------------------------------


def test_tree_output_rows_must_have_every_declared_column():
    with pytest.raises(pydantic.ValidationError, match="missing declared column"):
        TreeOutput(data=[{"a": 1}], dtypes=[("a", "Int64"), ("b", "Int64")])
    with pytest.raises(pydantic.ValidationError, match="default is missing"):
        TreeOutput(data=[], default={}, dtypes={"a": "Int64"})


def test_a_tree_checks_its_node_references():
    leaf = Node(data=LeafNode())
    unary = UnaryNode(condition=UnaryLessThan(feature="x", threshold=1))
    with pytest.raises(pydantic.ValidationError, match="not a node"):
        Tree(rules=[Rule(root="missing")], nodes={"a": leaf})
    with pytest.raises(pydantic.ValidationError, match="unknown node"):
        Tree(rules=[Rule(root="a")], nodes={"a": Node(data=unary, children=("ghost", None))})
    with pytest.raises(pydantic.ValidationError, match="expected 2"):
        Tree(rules=[Rule(root="a")], nodes={"a": Node(data=unary, children=(None,))})
    with pytest.raises(pydantic.ValidationError, match="cycle"):
        Tree(rules=[Rule(root="a")], nodes={"a": Node(data=unary, children=("a", None))})


def test_a_cases_node_has_one_branch_per_condition_plus_otherwise():
    node = CasesRanges(feature="x", conditions=[RangeCondition(max=1.0), RangeCondition(min=1.0)])
    Tree(rules=[Rule(root="c")], nodes={"c": Node(data=node, children=(None, None, None))})
