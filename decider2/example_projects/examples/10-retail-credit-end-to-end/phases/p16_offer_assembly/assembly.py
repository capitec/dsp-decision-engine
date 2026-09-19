"""P16 offer assembly and cross-product arbitration — 74 decision points.
Spec 5.17. Elided from phases/__init__.py "for length"; this package is the
real file. "No isolated flow owns this phase" — each thing it arbitrates has
its own notion of "best"; composing them needs a FOURTH notion, owned by
four teams (T7, T8, T9, T11) jointly.

Campaign-tree evaluation captures the PATH TAKEN through each of the 44 trees
as first-class output (spec 5.17(d)) — the input to §13 Q17's "is a tree node
a decision point" question this project answers in README §11 and
deadlogic/candidates.py.
"""

from __future__ import annotations

from decider2 import module, param, Collection

def candidate_assembly(solve_offers: list[dict], consolidation_scenarios: list[dict],
                       proposed_limit: dict | None, reprice_offer: dict | None,
                       campaign_leaves: list[dict]) -> list[dict]:
    """14 decision points. Up to 8 term-amount offers per eligible product, up
    to 3 consolidation scenarios, a proposed limit, a re-price, and — entry
    point 4 only — up to 44 campaign leaves."""
    pass  # union everything the flow produced for this decision into one candidate list

def suppression(candidate_assembly: list[dict], consent_state: dict,
                do_not_target_hit: bool,
                min_offer_amount: float = param(2_000_00, ge=0),
                min_instalment: float = param(185_00, ge=0),
                ear_suppression_ceiling: float = param(0.58, ge=0)) -> list[dict]:
    """16 decision points. Minimum viable offer thresholds, 26 contact-fatigue
    rules on entry point 4, cooling-off windows, consent-derived channel
    restrictions, the do-not-target register. Every suppression is RECORDED
    with its reason."""
    pass  # filter candidate_assembly, recording the suppressing rule per removed offer

def deduplication(suppression: list[dict],
                  instalment_tolerance_pct: float = param(0.025, ge=0),
                  amount_tolerance: float = param(1_000_00, ge=0)) -> list[dict]:
    """9 decision points. Two offers within 2.5% instalment and R1 000 amount
    are the same offer; the shorter term survives."""
    pass  # merge near-duplicate offers, keeping the shorter term

def campaign_trees(deduplication: list[dict], client_features: dict) -> Collection:
    """21 decision points. 44 trees, ~6 800 nodes, ~2 900 leaves. The PATH
    TAKEN is first-class output, not a debugging aid — README §11 explains
    why leaves are counted as decision points and internal nodes are not."""
    pass  # evaluate each applicable tree; capture the taken path per tree

def cross_product_arbitration(deduplication: list[dict], campaign_trees: Collection,
                              ranking_basis: str = param("client_outcome")) -> list[dict]:
    """14 decision points. The phase's namesake: a fourth notion of "best"
    across incommensurable units (a term-loan offer, a card limit, a
    consolidation scenario), owned by four teams for one number."""
    pass  # rank the combined candidate set by ranking_basis, mark is_recommended

Assemble = module(candidate_assembly, suppression, deduplication, campaign_trees,
                  cross_product_arbitration, name="assembly",
                  taps=["campaign_trees"])
