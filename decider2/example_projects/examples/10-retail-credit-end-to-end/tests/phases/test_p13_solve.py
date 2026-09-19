"""Per-phase unit set, P13. One of eighteen such files (spec 5.30.2: 18 sets,
1 200-31 000 cases each, ~164 000 total). Runs on every change to P13.

This test encodes spec 5.14's worked failure verbatim: the band-edge
inversion at grade 8 / 60 months, where the feasible set is
{R2 000..R58 250} union {R60 000} and the correct answer is R60 000, not
R58 250. A search that assumes affordability is monotone in the amount fails
this test; a `Partition`-based search over declared breakpoints
(phases/p13_solve/search.py) does not, because the breakpoint at the R60 000
band edge is DECLARED, not discovered by scanning.
"""

from __future__ import annotations

from phases.p13_solve.search import SolvePerTerm


def test_band_edge_inversion_grade8_60_months():
    """Spec 5.14's worked failure. max_affordable_instalment R1 820.00; card
    prices R60 000-69 999 at 17.40%, bands below R60 000 at 18.95%."""
    result = SolvePerTerm.score(
        product_minimum=2_000_00, requested_amount=95_000_00, amount_cap=120_750_00,
        product_maximum=500_000_00, product_code=10, term_months=60, risk_grade=8,
        max_affordable_instalment=1_820_00, decision_date="2027-06-15",
    )
    assert result["offer_or_no_offer"]["amount"] == 60_000_00
    assert result["offer_or_no_offer"]["binding_constraint"] == "BIND-AFF"
    # A binary search over the raw domain returns 58_250_00 here -- the
    # regression this test exists to catch is exactly that answer.


def test_evaluation_ceiling_is_not_negotiable():
    """correctness_parameters=("evaluations_per_term",) on the P13 envelope:
    no caller may raise the 19-per-term ceiling to buy more accuracy under
    budget pressure. §11 scenario 9's "cut the budget by 40%" negotiation
    must never touch this number."""
    result = SolvePerTerm.score(
        product_minimum=2_000_00, requested_amount=500_000_00, amount_cap=500_000_00,
        product_maximum=500_000_00, product_code=10, term_months=84, risk_grade=1,
        max_affordable_instalment=50_000_00, decision_date="2027-06-15",
    )
    assert result["partitioned_search"]["evaluation_count"] <= 19


def test_re_check_never_fails_from_scratch():
    """Spec 5.14 requirement 7: mandatory, zero-tolerance regression. This one
    case stands in for the 180 000-application re-check regression
    (tests/golden/manifest.toml) that runs on every change to P09-P13."""
    offer = SolvePerTerm.score(
        product_minimum=2_000_00, requested_amount=60_000_00, amount_cap=120_750_00,
        product_maximum=500_000_00, product_code=10, term_months=60, risk_grade=8,
        max_affordable_instalment=1_820_00, decision_date="2027-06-15",
    )["offer_or_no_offer"]
    rechecked = SolvePerTerm.score(
        product_minimum=2_000_00, requested_amount=offer["amount"], amount_cap=120_750_00,
        product_maximum=500_000_00, product_code=10, term_months=60, risk_grade=8,
        max_affordable_instalment=1_820_00, decision_date="2027-06-15",
    )["offer_or_no_offer"]
    assert rechecked["amount"] == offer["amount"]
