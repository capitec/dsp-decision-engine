"""Campaign 23, Flex Loan pre-approved top-up -- spec 04 §5.3.3's worked example, authored
faithfully (same ten internal nodes, seven leaves, same conditions and thresholds) as this
project's one hand-built tree. The other ~59 live campaigns are generated at scale instead
(`generate_trees.py`) -- see NOTES.md "What I built" for why this one is hand-built and the
rest are not.

Two published versions are built here:

  - `TREE_V1` -- node 6's `discretionary_income` floor at its original 2 200 (§5.3.3's own
    number), with two thresholds marked overlay-tunable (`param_threshold`): node 6's
    `discretionary_income` floor (the volume dial worked example, §5.3.4) and node 3's
    `behaviour_score` floor (standing in for the cut-off shift worked example, which the
    spec phrases in terms of `risk_grade`; this tree's worked example has no `risk_grade`
    node to shift, so the same *mechanism* is demonstrated against the risk floor it does
    have -- see NOTES.md "Spec problems").
  - `TREE_V2` -- the *same* tree republished with node 6's floor permanently moved to 2 600
    (not an overlay -- a real new version), for the node identity map demo (§5.4.3,
    acceptance §10 item 7) and change scenario 1/15 ("the March-versus-April comparison of
    node 6 must become visibly incomparable").

Node 6 is deliberately reachable from both node 3 (its false branch) and node 4 (its true
branch) -- §5.3.3's own note ("node 6 is reachable from both node 3 and node 4") -- to
exercise `tree_model`'s DAG convergence handling for real, not as a synthetic edge case.
"""
from __future__ import annotations

from campaign_trees import vocab
from campaign_trees.tree_model import Tree, build_tree, param_threshold

CAMPAIGN_ID = 23
NODE6_DISCRETIONARY_INCOME_PARAM = "campaign23_node6_discretionary_income_thresh"
NODE3_BEHAVIOUR_SCORE_PARAM = "campaign23_node3_behaviour_score_thresh"

TIER_A_CEILING = 250_000.0
TIER_B_CEILING = 120_000.0
TIER_C_CEILING = 60_000.0


def _leaf(**output) -> dict:
    return {"kind": "leaf", "output": output}


def _authored_nodes(node6_discretionary_income_default: float) -> dict:
    return {
        "n1": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "has_active_flex_loan", "op": "is_true"},
                {"feature": "months_on_book_flex", "op": ">=", "threshold": 6},
                {"feature": "flex_term_remaining_months", "op": ">=", "threshold": 4},
            ],
            "true": "n2", "false": "leaf901",
        },
        "n2": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "settlement_ratio", "op": "<=", "threshold": 0.65},
                {"feature": "payments_missed_12m", "op": "==", "threshold": 0},
            ],
            "true": "n3", "false": "n4",
        },
        "n3": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "behaviour_score", "op": ">=",
                 "threshold": param_threshold(NODE3_BEHAVIOUR_SCORE_PARAM, 620.0)},
                {"feature": "worst_arrears_months_12m", "op": "==", "threshold": 0},
                {"feature": "bureau_enquiries_3m", "op": "<=", "threshold": 2},
                {"feature": "external_unsecured_growth_6m", "op": "<=", "threshold": 0.20},
            ],
            "true": "n5", "false": "n6",
        },
        "n4": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "behaviour_score", "op": ">=", "threshold": 580.0},
                {"feature": "months_since_last_arrears", "op": ">=", "threshold": 9},
                {"feature": "worst_arrears_months_12m", "op": "<=", "threshold": 1},
            ],
            "true": "n6", "false": "leaf902",
        },
        "n5": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "discretionary_income", "op": ">=", "threshold": 3500.0},
                {"feature": "estimated_instalment_to_income", "op": "<=", "threshold": 0.28},
                {"feature": "income_band_code", "op": "isin", "values": [5, 6, 7, 8, 9]},
            ],
            "true": "n7", "false": "n8",
        },
        "n6": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "discretionary_income", "op": ">=",
                 "threshold": param_threshold(NODE6_DISCRETIONARY_INCOME_PARAM, node6_discretionary_income_default)},
                {"feature": "estimated_instalment_to_income", "op": "<=", "threshold": 0.33},
                {"feature": "employment_type_code", "op": "isin", "values": [1, 2, 4]},
            ],
            "true": "n8", "false": "leaf903",
        },
        "n7": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "pre_assessed_amount", "op": ">=", "threshold": 40_000.0},
                {"expr": "pre_assessed_amount - 1.5 * current_flex_balance", "op": ">=", "threshold": 0.0},
            ],
            "true": "n9", "false": "n10",
        },
        "n8": {
            "kind": "test", "logic": None,
            "terms": [{"feature": "pre_assessed_amount", "op": ">=", "threshold": 15_000.0}],
            "true": "n10", "false": "leaf904",
        },
        "n9": {
            "kind": "test", "logic": "or",
            "terms": [
                {"feature": "app_logins_90d", "op": ">=", "threshold": 3},
                {"feature": "channel_preference_code", "op": "==", "threshold": 2},
            ],
            "true": "leaf910", "false": "leaf911",
        },
        "n10": {
            "kind": "test", "logic": "and",
            "terms": [
                {"feature": "sms_response_rate_12m", "op": ">=", "threshold": 0.04},
                {"feature": "prior_offer_declines_6m", "op": "<=", "threshold": 1},
            ],
            "true": "leaf912", "false": "leaf913",
        },
        # Every output row must carry the same columns (`TreeOutput` requires it) -- a "do
        # not target" leaf's offer fields are the declared sentinel 0/0.0, never null, so a
        # warehouse join never has to special-case a decline row's shape.
        "leaf901": _leaf(leaf_outcome_code=vocab.DO_NOT_TARGET, offer_tier_code=0, tier_ceiling=0.0,
                          channel_1=0, channel_2=0, priority_weight=0.0, reason_label=9201),
        "leaf902": _leaf(leaf_outcome_code=vocab.DO_NOT_TARGET, offer_tier_code=0, tier_ceiling=0.0,
                          channel_1=0, channel_2=0, priority_weight=0.0, reason_label=9202),
        "leaf903": _leaf(leaf_outcome_code=vocab.DO_NOT_TARGET, offer_tier_code=0, tier_ceiling=0.0,
                          channel_1=0, channel_2=0, priority_weight=0.0, reason_label=9203),
        "leaf904": _leaf(leaf_outcome_code=vocab.DO_NOT_TARGET, offer_tier_code=0, tier_ceiling=0.0,
                          channel_1=0, channel_2=0, priority_weight=0.0, reason_label=9204),
        "leaf910": _leaf(leaf_outcome_code=vocab.TARGET, offer_tier_code=vocab.TIER_A, tier_ceiling=TIER_A_CEILING,
                          channel_1=vocab.CHANNEL_IN_APP, channel_2=vocab.CHANNEL_EMAIL,
                          priority_weight=0.86, reason_label=9101),
        "leaf911": _leaf(leaf_outcome_code=vocab.TARGET, offer_tier_code=vocab.TIER_A, tier_ceiling=TIER_A_CEILING,
                          channel_1=vocab.CHANNEL_CALL, channel_2=vocab.CHANNEL_SMS,
                          priority_weight=0.81, reason_label=9102),
        "leaf912": _leaf(leaf_outcome_code=vocab.TARGET, offer_tier_code=vocab.TIER_B, tier_ceiling=TIER_B_CEILING,
                          channel_1=vocab.CHANNEL_SMS, channel_2=vocab.CHANNEL_IN_APP,
                          priority_weight=0.58, reason_label=9103),
        "leaf913": _leaf(leaf_outcome_code=vocab.TARGET, offer_tier_code=vocab.TIER_C, tier_ceiling=TIER_C_CEILING,
                          channel_1=vocab.CHANNEL_IN_APP, channel_2=0,
                          priority_weight=0.34, reason_label=9104),
    }


def build(version_name: str, node6_discretionary_income_default: float) -> Tree:
    return build_tree(_authored_nodes(node6_discretionary_income_default), root="n1", name=version_name)


TREE_V1 = build("campaign23_v1", node6_discretionary_income_default=2200.0)
TREE_V2 = build("campaign23_v2", node6_discretionary_income_default=2600.0)
