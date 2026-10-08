"""Campaign 23's tree as published: the paths clients take, and what an edit or an overlay changes."""
import json

import polars as pl
import pytest
from decider import TreeConfig

from campaign_trees.trees import changed_nodes, problems

# A client who reaches node 5 and, with a big enough pre-assessed amount, the digital-engagement node 9.
STRONG = {
    "has_active_flex_loan": True, "months_on_book_flex": 30, "flex_term_remaining_months": 20, "settlement_ratio": 0.5,
    "payments_missed_12m": 0, "behaviour_score": 700, "worst_arrears_months_12m": 0, "months_since_last_arrears": 24,
    "bureau_enquiries_3m": 0, "external_unsecured_growth_6m": 0.0, "discretionary_income": 5000.0,
    "estimated_instalment_to_income": 0.2, "income_band_code": 7, "employment_type_code": 1,
    "pre_assessed_amount": 90_000.0, "current_flex_balance": 10_000.0, "app_logins_90d": 5, "channel_preference_code": 1,
    "sms_response_rate_12m": 0.1, "prior_offer_declines_6m": 0,
}
# The sample client: a settlement ratio over 0.65 sends them through node 4, which sends them on to node 6.
SAMPLE = {**STRONG, "settlement_ratio": 0.71, "behaviour_score": 604, "worst_arrears_months_12m": 1,
          "months_since_last_arrears": 14, "discretionary_income": 3118.0, "estimated_instalment_to_income": 0.26,
          "pre_assessed_amount": 20_800.0, "current_flex_balance": 20_000.0, "income_band_code": 5,
          "app_logins_90d": 1, "sms_response_rate_12m": 0.061}


def walk(tree, record, **params):
    out = tree.run(pl.DataFrame([record]), params={"campaign_23": params} if params else None)
    return out["tree_path"][0], out["reason_label"][0]


@pytest.mark.parametrize("record, path, reason_label", [
    (SAMPLE, "n1>n2>n4>n6>n8>n10>leaf_912", 9103),
    ({**STRONG}, "n1>n2>n3>n5>n7>n9>leaf_910", 9101),
    ({**STRONG, "app_logins_90d": 0}, "n1>n2>n3>n5>n7>n9>leaf_911", 9102),
    ({**STRONG, "sms_response_rate_12m": 0.01, "pre_assessed_amount": 30_000.0}, "n1>n2>n3>n5>n7>n10>leaf_913", 9104),
    ({**STRONG, "has_active_flex_loan": False}, "n1>leaf_901", 9201),
    ({**SAMPLE, "behaviour_score": 560}, "n1>n2>n4>leaf_902", 9202),
    ({**SAMPLE, "employment_type_code": 3}, "n1>n2>n4>n6>leaf_903", 9203),
    ({**SAMPLE, "pre_assessed_amount": 9_000.0}, "n1>n2>n4>n6>n8>leaf_904", 9204),
])
def test_each_client_walks_to_its_leaf_and_the_path_says_how(tree, record, path, reason_label):
    assert walk(tree, record) == (path, reason_label)


def test_node_6_is_reached_from_both_node_3_and_node_4(tree):
    via_node_3 = {**STRONG, "discretionary_income": 3000.0, "estimated_instalment_to_income": 0.3}
    assert walk(tree, via_node_3)[0].split(">")[:4] == ["n1", "n2", "n3", "n5"]
    through_3_false = {**STRONG, "behaviour_score": 610}
    assert walk(tree, through_3_false)[0].split(">")[:5] == ["n1", "n2", "n3", "n6", "n8"]
    assert walk(tree, SAMPLE)[0].split(">")[:4] == ["n1", "n2", "n4", "n6"]


def test_node_7_compares_the_pre_assessed_amount_with_the_balance(tree):
    heavy_balance = {**STRONG, "current_flex_balance": 70_000.0}  # 90 000 < 1.5 * 70 000
    assert walk(tree, heavy_balance)[0] == "n1>n2>n3>n5>n7>n10>leaf_912"


def test_a_missing_feature_takes_the_otherwise_branch_instead_of_failing(tree):
    out = tree.run(pl.DataFrame([{**SAMPLE, "behaviour_score": None}]))
    assert out["tree_path"][0] == "n1>n2>n4>leaf_902"


def test_a_param_moves_a_threshold_without_changing_the_node_ids(tree):
    published = walk(tree, {**SAMPLE, "discretionary_income": 2400.0})
    dialled = walk(tree, {**SAMPLE, "discretionary_income": 2400.0}, n6_discretionary_income_floor=2600.0)
    assert published[0] == "n1>n2>n4>n6>n8>n10>leaf_912"
    assert dialled[0] == "n1>n2>n4>n6>leaf_903"
    assert published[0].split(">")[:4] == dialled[0].split(">")[:4]


def test_the_published_tree_is_complete(tree):
    assert problems(tree) == []


def test_a_node_nothing_reaches_and_a_branch_to_nowhere_are_found(tree_document):
    document = json.loads(json.dumps(tree_document))
    document["tree"]["nodes"].append({"id": "orphan", "data": {"type": "leaf", "result_idx": 0}})
    document["tree"]["edges"] = [e for e in document["tree"]["edges"] if not (e["source"] == "n8" and e["target"] == "leaf_904")]
    assert problems(TreeConfig.load(document)) == [
        "node leaf_904 is not reachable from the root", "node orphan is not reachable from the root",
        "node n8 has a branch that leads nowhere"]


def test_raising_node_6_s_floor_in_the_document_changes_only_node_6(tree, tree_document):
    document = json.loads(json.dumps(tree_document))
    condition = next(n for n in document["tree"]["nodes"] if n["id"] == "n6")["data"]["conditions"][0]
    condition["threshold"]["default"] = 2600.0
    edited = TreeConfig.load(document)
    assert changed_nodes(tree, edited) == {"added": [], "removed": [], "changed": ["n6"]}
    assert walk(edited, {**SAMPLE, "discretionary_income": 2400.0})[0] == "n1>n2>n4>n6>leaf_903"
