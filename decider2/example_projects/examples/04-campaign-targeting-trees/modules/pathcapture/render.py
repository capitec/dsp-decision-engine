"""Rendering a stored path — spec §5.4.1(d), §9.1.

Inputs, all of them stored artefacts, none of them a running system:

    paths row            (client_id, campaign_id, cycle_id, tree_version,
                          route_digest, overlay_stack_id, variant, leaf_key)
    route_dictionary     campaigns/<c>/routes/v<n>.routes.csv
    node_meta            campaigns/<c>/node_meta/v<n>.node_meta.csv
    feature snapshot     the frozen stage-1 snapshot for that cycle
    threshold values     the tree document's slots, plus the overlay delta for
                         `overlay_stack_id`

Output, for client 8 412 907, cycle 2026-09, campaign 23, tree version 11:

    node n_9f2c41a7b0e35d18   "L1 facility gate"
        has_active_flex_loan = true
        months_on_book_flex = 31 (>= 6)
        flex_term_remaining_months = 22 (>= 4)                  HELD     -> node 2
    node n_4b71e0cd5a92f36c   "L2 settlement/arrears gate"
        settlement_ratio = 0.71 (<= 0.65 FAILS)
        payments_missed_12m = 0 (= 0)                           NOT HELD -> node 4
    node n_2e6f9b14a7c308d5   "L3 risk gate, relaxed"
        behaviour_score = 604 (>= 580)
        months_since_last_arrears = 14 (>= 9)
        worst_arrears_months_12m = 1 (<= 1)                      HELD     -> node 6
    node n_a1f45e9c2b70d863   "L4 affordability gate, standard"
        discretionary_income = R3 118 (>= 2 200)
        estimated_instalment_to_income = 0.26 (<= 0.33)
        employment_type_code = 1 (in {1,2,4})                    HELD     -> node 8
    node n_e83017bd6f2c94a5   "L5 amount floor, standard"
        pre_assessed_amount = R84 000 (>= 15 000)                HELD     -> node 10
    node n_6b4d2f9138ea07c5   "L6 channel split, SMS"
        sms_response_rate_12m = 0.061 (>= 0.04)
        prior_offer_declines_6m = 0 (<= 1)                       HELD     -> leaf
    leaf l_cf2839a54e0b761d   TARGET, tier B, R84 000, SMS then in-app,
                              priority 0.58, "Standard top-up, SMS responsive"

    overlay stack 4471100:  OV-2026-114 volume dial, node
                            n_a1f45e9c2b70d863 threshold 2 200 (unchanged this
                            cycle; the dial moved it to 2 600 from 2026-10-01)
    control: no.  variant: champion.  unadjusted leaf: same.

Note the second node: visited, condition did not hold, and printed.  The stored
route holds it (modules/tree/routes.py explains why), so the rendering is a
lookup and not a reconstruction.

THE IMPORT LIST IS THE POINT
------------------------------
    import csv, json, datetime          # and the artefact's portable interpreter

No decider2, no numba, no polars, no database driver.  The staff member producing
the §9.1 answer within five business days is not an engineer and does not run the
decision system, and 18 months from now the decision system will not be the same
software.  What must still exist is the CSV, the JSON and 200 lines of Python
that shipped with the tree.
"""

from __future__ import annotations


def render_path(path_row: dict, route_dict: str, node_meta: str,
                snapshot_row: dict, thresholds: dict) -> str:
    """Produce the text above.  Pure lookup plus the portable interpreter for the
    per-atom HELD/FAILS annotations."""
    pass


def rederive_leaf(snapshot_row: dict, tree_document: dict, thresholds: dict) -> str:
    """Spec §5.4.2(4): re-derive the leaf on a laptop, from the snapshot and the
    document, with no decision system available.  Asserting that this agrees with
    the stored `leaf_key` for a sample of every cycle is a standing test, and it
    is the fourth rung of the equivalence ladder (DEMANDS #12)."""
    pass


def plain_language_reason(leaf_key: str, node_meta: str, reason_labels: str) -> str:
    """Spec §5.8: the contact centre needs a plain-language statement of why this
    client was selected, *derived from the reason label and the path*, not written
    by hand per campaign.

    Derivation: the reason label supplies the sentence; the path supplies the two
    or three conditions with the largest population discrimination on the route,
    named from `node_meta.readable_condition`.  "You hold a Flex Loan in good
    standing and have made every payment for the last 12 months."  Sixty campaigns
    x 190 leaves is 11 400 hand-written sentences nobody will maintain; 220 reason
    labels plus a path is maintainable.
    """
    pass
