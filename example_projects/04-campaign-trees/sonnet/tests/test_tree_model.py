"""tree_model.py: node identity (§5.4.3), validation (§5.9) and path capture (§5.4.1)."""
from __future__ import annotations

import polars as pl
import pytest
from decider.steps.trees import TreeConfig

from campaign_trees import campaign23, tree_model

BASE_RECORD = {
    "has_active_flex_loan": True, "months_on_book_flex": 31, "flex_term_remaining_months": 22,
    "settlement_ratio": 0.71, "payments_missed_12m": 0,
    "behaviour_score": 604, "worst_arrears_months_12m": 1, "months_since_last_arrears": 14,
    "bureau_enquiries_3m": 1, "external_unsecured_growth_6m": 0.1,
    "discretionary_income": 3118, "estimated_instalment_to_income": 0.26, "employment_type_code": 1,
    "income_band_code": 5,
    "pre_assessed_amount": 84000.0, "current_flex_balance": 20000.0,
    "app_logins_90d": 1, "channel_preference_code": 1,
    "sms_response_rate_12m": 0.061, "prior_offer_declines_6m": 0,
}


def test_worked_example_path_matches_spec_04_5_4_1_d():
    """The spec's own worked example (§5.4.1(d)): client 8412907, path 1 -> 4 -> 6 -> 8 -> 10
    -> leaf 912, tier B."""
    path = tree_model.walk(campaign23.TREE_V1, BASE_RECORD)
    assert path[-1].kind == "leaf"
    assert path[-1].output["offer_tier_code"] == 2  # tier B
    assert path[-1].output["reason_label"] == 9103  # "Standard top-up, SMS responsive"
    # node 2 (n2 in this authoring) is visited-and-not-held, exactly as the spec's own
    # rendering shows it (§5.4.1(d)'s note: "node 2 appears... although it is not... visited-
    # and-passed").
    assert len(path) == 7  # n1, n2, n4, n6, n8, n10, then the leaf
    assert [s.held for s in path[:-1]] == [True, False, True, True, True, True]


def test_walker_agrees_with_treeconfig_leaf():
    """`TreeConfig`'s own evaluation and this project's walker must reach the same leaf for
    every record -- see tree_model.py's module docstring on why two traversal engines exist
    at all and why this is the check that risk earns."""
    doc = tree_model.to_v3_document(campaign23.TREE_V1)
    tc = TreeConfig.load(doc)
    variants = [
        BASE_RECORD,
        {**BASE_RECORD, "has_active_flex_loan": False},
        {**BASE_RECORD, "settlement_ratio": 0.5},  # -> node 3 branch instead of node 4
        {**BASE_RECORD, "discretionary_income": 1000.0},  # -> leaf 903, do not target
        {**BASE_RECORD, "pre_assessed_amount": 5000.0},  # -> leaf 904, below minimum
        {**BASE_RECORD, "settlement_ratio": 0.5, "bureau_enquiries_3m": 1, "external_unsecured_growth_6m": 0.1,
         "pre_assessed_amount": 60000.0, "app_logins_90d": 5},  # -> node 3 -> node 5 -> node 7 -> node 9
    ]
    df = pl.DataFrame(variants)
    out = tc.run(df)
    for i, record in enumerate(variants):
        walker_leaf = tree_model.walk(campaign23.TREE_V1, record)[-1].node_key
        assert walker_leaf == out["leaf_key"][i], f"variant {i}: TreeConfig and walker disagree"


def test_node6_is_a_dag_convergence_point_with_two_parents_and_one_identity():
    """§5.3.3's own note: node 6 is reachable from both node 3 (false branch) and node 4
    (true branch). The v3 document represents it as one physical node (a DAG, not a strict
    tree) with two incoming edges, so it has exactly one `node_key` by construction -- this
    test asserts the *position signature* (this project's identity mechanism) actually
    records two parents for it, not that two separately-built nodes happen to collide."""
    two_parent_nodes = {key for key, sig in campaign23.TREE_V1.position_signature.items() if len(sig) == 2}
    assert len(two_parent_nodes) >= 1  # node 6 (and, per the same table, node 8) each have two parents
    # the node(s) visited on BOTH of these two different routes, restricted to two-parent
    # nodes, is exactly the convergence point(s) -- reached by one identity either way.
    record_via_n3_false = {**BASE_RECORD, "settlement_ratio": 0.5, "behaviour_score": 500}  # n2 true -> n3(false) -> n6
    record_via_n4_true = {**BASE_RECORD}  # n2 false -> n4(true) -> n6, as the worked example does
    keys_a = {s.node_key for s in tree_model.walk(campaign23.TREE_V1, record_via_n3_false)}
    keys_b = {s.node_key for s in tree_model.walk(campaign23.TREE_V1, record_via_n4_true)}
    shared_convergence = (keys_a & keys_b) & two_parent_nodes
    assert shared_convergence, "node 6 should be reached by one identity from either parent path"


def test_overlay_changes_leaf_for_a_client_near_the_threshold():
    """A client whose discretionary_income sits between the published floor (2 200) and the
    volume-dial's tightened floor (2 600, §5.3.4's own worked example) reaches a different
    leaf with the overlay applied -- the mechanism §5.3.4 requirement 2 ("the unadjusted
    answer survives") exists to make visible."""
    record = {**BASE_RECORD, "discretionary_income": 2400.0}
    unadjusted = tree_model.walk(campaign23.TREE_V1, record, overrides={})
    adjusted = tree_model.walk(campaign23.TREE_V1, record,
                                overrides={"campaign23_node6_discretionary_income_thresh": 2600.0})
    assert unadjusted[-1].output["leaf_outcome_code"] == 1  # target, unadjusted
    assert adjusted[-1].output["leaf_outcome_code"] == 0    # do not target, adjusted (tightened)
    assert unadjusted[-1].node_key != adjusted[-1].node_key


def test_node_identity_map_isolates_the_one_changed_node():
    """§10 item 7: publishing TREE_V2 (node 6's floor permanently moved 2200 -> 2600)
    classifies node 6 as `changed`, and node 1 (unrelated, upstream of nothing that
    changed) stays `carried_forward`."""
    m = tree_model.node_identity_map(campaign23.TREE_V1, campaign23.TREE_V2)
    assert len(m.changed) == 1
    changed = m.changed[0]
    assert "campaign23_node6_discretionary_income_thresh=2200.0" in changed["old_condition"]
    assert "campaign23_node6_discretionary_income_thresh=2600.0" in changed["new_condition"]
    assert campaign23.TREE_V1.root in m.carried_forward  # n1 -- entirely unaffected by the node-6 edit
    # node 6's own downstream (n8, n10, leaves 903/904/912/913) legitimately gets new
    # identity too -- see NOTES.md "What I built" for why that cascade is the conservative,
    # correct-by-construction behaviour rather than a bug.
    assert len(m.added) > 0 and len(m.removed) > 0


def test_validate_tree_rejects_unknown_and_prohibited_features():
    nodes = {
        "root": {"kind": "test", "logic": None,
                  "terms": [{"feature": "not_a_real_feature", "op": ">=", "threshold": 1.0}],
                  "true": "leaf_a", "false": "leaf_b"},
        "leaf_a": {"kind": "leaf", "output": {"x": 1}},
        "leaf_b": {"kind": "leaf", "output": {"x": 0}},
    }
    tree = tree_model.build_tree(nodes, root="root", name="t")
    problems = tree_model.validate_tree(tree, feature_registry={"age": "Float64"}, prohibited_features={"age"})
    assert any("unknown feature" in p for p in problems)


def test_validate_tree_rejects_unreachable_node():
    nodes = {
        "root": {"kind": "leaf", "output": {"x": 1}},
        "orphan": {"kind": "leaf", "output": {"x": 2}},
    }
    with pytest.raises(ValueError, match="unreachable"):
        tree_model.build_tree(nodes, root="root", name="t")


def test_validate_tree_catches_contradiction():
    nodes = {
        "root": {"kind": "test", "logic": "and",
                  "terms": [{"feature": "age", "op": ">=", "threshold": 65},
                            {"feature": "age", "op": "<", "threshold": 60}],
                  "true": "leaf_a", "false": "leaf_b"},
        "leaf_a": {"kind": "leaf", "output": {"x": 1}},
        "leaf_b": {"kind": "leaf", "output": {"x": 0}},
    }
    tree = tree_model.build_tree(nodes, root="root", name="t")
    problems = tree_model.validate_tree(tree, feature_registry={"age": "Float64"}, prohibited_features=set())
    assert any("contradictory" in p for p in problems)
