"""Synthetic tree generation at real scale (spec 04 §4.2: "60 trees, 20-400 nodes each,
~9 200 nodes in total"; SCOPE.md rule 1: "the difficulty is trees and paths, so keep the
tree count and size near real... generate large tables synthetically at their full
dimensions").

Campaign 23 (`campaign23.py`) is the one hand-built tree, faithful to spec §5.3.3's worked
example. The other ~59 live campaigns are generated here: a proper (cycle-free,
every-path-terminates-in-a-leaf, no-unreachable-node) binary tree over a synthetic feature
pool, sized and shaped to the spec's own declared ranges (depth 4-12, mean path length 7.4,
1-6 conditions per node, mean 2.3). Generation is deterministic (`random.Random(seed)`), so
the same seed always reproduces the same campaign -- required for a re-run to be a re-run
(§10 item 4) and for `tests/test_generate_trees.py` to assert against fixed output.

This exists to prove the framework holds at the stated volume (validation timing over ~9 200
nodes, the node identity map at scale, a synthetic-population batch run) -- not to be served.
`pipeline.py` serves campaign 23 only, exactly as 00 and 03's own demo pipelines serve one
representative slice rather than everything they built (see their NOTES.md "What I built").
"""
from __future__ import annotations

import random

from campaign_trees import vocab
from campaign_trees.tree_model import Tree, build_tree, validate_tree

# A representative slice of the ~600-feature client mart (§4.1), enough to generate
# realistic-looking multi-condition nodes without declaring all 600.
_NUMERIC_FEATURES = [
    "behaviour_score", "discretionary_income", "estimated_instalment_to_income", "settlement_ratio",
    "months_on_book_flex", "bureau_enquiries_3m", "external_unsecured_growth_6m", "worst_arrears_months_12m",
    "months_since_last_arrears", "app_logins_90d", "sms_response_rate_12m", "prior_offer_declines_6m",
    "pre_assessed_amount", "current_flex_balance", "age_years", "tenure_months", "salary_credit_count_3m",
    "average_balance_6m", "turnover_12m", "propensity_score", "probability_of_default",
]
_BOOLEAN_FEATURES = ["has_active_flex_loan", "has_credit_card", "is_salaried", "opted_out_email"]
_SET_FEATURES = {
    "income_band_code": list(range(1, 10)),
    "employment_type_code": [1, 2, 3, 4, 5],
    "province_code": list(range(1, 10)),
    "segment_code": [1, 2, 3, 4],
}

ALL_FEATURES = set(_NUMERIC_FEATURES) | set(_BOOLEAN_FEATURES) | set(_SET_FEATURES)


def _random_term(rng: random.Random) -> dict:
    kind = rng.choice(["numeric", "boolean", "set"])
    if kind == "numeric":
        feat = rng.choice(_NUMERIC_FEATURES)
        op = rng.choice([">=", "<=", ">", "<"])
        threshold = round(rng.uniform(0.1, 1000.0), 2)
        return {"feature": feat, "op": op, "threshold": threshold}
    if kind == "boolean":
        feat = rng.choice(_BOOLEAN_FEATURES)
        return {"feature": feat, "op": rng.choice(["is_true", "is_false"])}
    feat = rng.choice(list(_SET_FEATURES))
    values = rng.sample(_SET_FEATURES[feat], k=rng.randint(1, len(_SET_FEATURES[feat])))
    return {"feature": feat, "op": "isin", "values": sorted(values)}


def _random_node_terms(rng: random.Random) -> tuple[list[dict], str | None]:
    """1-6 conditions per node, mean 2.3 (§5.3.1) -- each on a *different* feature, so a
    generated node can never contradict itself the way a spec-authored one could."""
    n = min(rng.choices([1, 2, 3, 4, 5, 6], weights=[35, 30, 15, 10, 6, 4])[0], 6)
    used_features: set[str] = set()
    terms = []
    for _ in range(n):
        for _attempt in range(5):
            term = _random_term(rng)
            if term["feature"] not in used_features:
                used_features.add(term["feature"])
                terms.append(term)
                break
    logic = None if len(terms) == 1 else rng.choice(["and", "and", "and", "or"])  # AND dominates, per §5.3.1
    return terms, logic


def _leaf_output(rng: random.Random) -> dict:
    if rng.random() < 0.45:
        return {"leaf_outcome_code": vocab.DO_NOT_TARGET, "offer_tier_code": 0, "tier_ceiling": 0.0,
                "channel_1": 0, "channel_2": 0, "priority_weight": 0.0,
                "reason_label": rng.choice([9201, 9202, 9203, 9204])}
    tier = rng.choice([vocab.TIER_A, vocab.TIER_B, vocab.TIER_C, vocab.TIER_D, vocab.TIER_E])
    return {
        "leaf_outcome_code": vocab.TARGET, "offer_tier_code": tier,
        "tier_ceiling": round(300_000.0 / tier, -2), "channel_1": rng.choice([1, 2, 3, 4]),
        "channel_2": rng.choice([0, 1, 2, 3, 4]), "priority_weight": round(rng.uniform(0.1, 0.95), 2),
        "reason_label": rng.choice([9101, 9102, 9103, 9104]),
    }


def generate_tree(campaign_id: int, target_nodes: int, seed: int) -> Tree:
    """A cycle-free binary tree with roughly `target_nodes` internal test nodes, built
    top-down so every node is reachable by construction (§5.9 items 1-2 hold trivially)."""
    rng = random.Random(seed)
    nodes: dict[str, dict] = {}
    counter = [0]

    def new_id() -> str:
        counter[0] += 1
        return f"c{campaign_id}_n{counter[0]}"

    def build(depth: int, remaining: int) -> str:
        # Stop at a leaf once out of node budget, at max depth (12), or by a shrinking
        # chance as depth grows -- keeps mean path length near the spec's 7.4 (§5.3.1)
        # rather than always hitting the hard caps.
        stop = remaining <= 0 or depth >= 12 or (depth >= 4 and rng.random() < 0.22)
        if stop:
            leaf_id = new_id()
            nodes[leaf_id] = {"kind": "leaf", "output": _leaf_output(rng)}
            return leaf_id
        terms, logic = _random_node_terms(rng)
        node_id = new_id()
        remaining -= 1
        true_id = build(depth + 1, remaining // 2)
        false_id = build(depth + 1, remaining - remaining // 2)
        nodes[node_id] = {"kind": "test", "logic": logic, "terms": terms, "true": true_id, "false": false_id}
        return node_id

    root = build(depth=1, remaining=target_nodes)
    return build_tree(nodes, root=root, name=f"campaign{campaign_id}_v1")


def generate_valid_tree(campaign_id: int, target_nodes: int, seed: int, *, max_attempts: int = 25) -> Tree:
    """`generate_tree`, retried under a bumped seed until §5.9's structural checks pass.

    Left un-retried, the generator legitimately produces self-contradictory nodes (it picks
    a random feature per condition without regard to what ancestors already constrained) --
    that is a *feature*, not a bug: it is what exercises `validate_tree`'s contradiction
    check for real (`tests/test_generate_trees.py::test_validator_catches_generated_contradictions`
    keeps one such tree on record deliberately). This wrapper is for callers that want a
    clean, publishable registry instead."""
    for attempt in range(max_attempts):
        tree = generate_tree(campaign_id, target_nodes, seed + attempt * 7919)
        if not validate_tree(tree, feature_registry={f: "Float64" for f in ALL_FEATURES}, prohibited_features=set()):
            return tree
    raise RuntimeError(f"campaign {campaign_id}: no valid tree found in {max_attempts} attempts")


def generate_registry(n_campaigns: int = 59, seed: int = 2026, start_id: int = 1, *, valid_only: bool = True,
                       ) -> dict[int, Tree]:
    """~59 trees (campaign 23 is the 60th, hand-built), 20-400 nodes each (§4.2), sized on a
    log-ish distribution so most are mid-sized and a few sit near each end -- the same shape
    the spec's own range implies, not a uniform draw. `valid_only=True` (the default) retries
    any generated tree that fails §5.9's structural validation, so this is the registry a
    scale test or a batch run can use as-is; `valid_only=False` returns the raw, sometimes-
    invalid draw, for testing the validator itself."""
    rng = random.Random(seed)
    trees: dict[int, Tree] = {}
    campaign_id = start_id
    for _ in range(n_campaigns):
        if campaign_id == 23:  # reserved for the hand-built worked example
            campaign_id += 1
        size = int(round(20 * (400 / 20) ** rng.random()))
        build_fn = generate_valid_tree if valid_only else generate_tree
        trees[campaign_id] = build_fn(campaign_id, size, seed * 1000 + campaign_id)
        campaign_id += 1
    return trees
