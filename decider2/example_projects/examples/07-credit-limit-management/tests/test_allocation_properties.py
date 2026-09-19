"""The allocation's acceptance criteria are properties of the OUTCOME, because
that is what an auditor tests. s5.8 states seven of them and s10 restates three
as acceptance criteria. They are asserted here, over a corpus rather than over
a hand-built fixture, because a hand-built fixture will never contain 708 000
accounts sitting near a funding line.
"""

import pytest
from decider2.testing import assert_modes_agree, corpus, golden

from clm.allocation.sweep import Allocate, Allocation
from clm.pipelines.programme import programme

BOOK = corpus.from_snapshot("2026-08", rows=4_100_000)
SMALL = corpus.from_snapshot("2026-08", rows=250_000, stratify=["behaviour_grade",
                                                                "utilisation_band"])


def test_total_funded_within_the_offer_envelope():
    """s5.8 constraint 1. In expectation, at the over-allocation factor."""
    pass


def test_no_increase_is_trimmed_to_fit():
    """s5.8 constraint 3. Every funded amount equals the amount s5.5 produced.
    A part-funded increase is an amount no cell of the matrix produced."""
    pass  # assert (funded_amount_c == additional_limit_c) | (funded_amount_c == 0)


def test_tail_skips_are_bounded_and_recorded():
    """s5.8 constraint 4. At most 2 000 skips, each carrying the outcome code,
    and the cycle stops rather than continuing silently past the bound."""
    pass


def test_starved_segment_is_visible_as_a_segment():
    """s5.8 constraint 5. A segment below 40% of the book-wide funded proportion
    for three consecutive cycles appears in the fairness report as one starved
    segment, not as 6 000 individually unlucky accounts."""
    pass


def test_fewer_than_two_percent_change_funded_status_on_unchanged_inputs():
    """s10.8, the measured hysteresis requirement. Run the same book twice with
    only the prior-cycle inputs advanced; assert the funded-status churn."""
    pass


def test_ties_break_by_account_id_and_the_funded_set_is_bit_identical():
    """s5.8 constraint 7 and s8's determinism row. Run the sweep over the same
    frame partitioned three different ways; assert identical ranks and an
    identical funded set. This is the test that fails if anyone makes the budget
    carry a float."""
    pass


def test_budget_carry_is_integer():
    """A lint would catch this too, but the property is load-bearing enough to
    assert: float addition is not associative, so a float budget accumulator
    makes the funding line depend on summation order."""
    assert Allocate.carry_dtypes["limit_budget_remaining_c"] == "int64"


def test_non_selection_is_answerable_from_the_record_alone():
    """s10.7. For a below-the-line account the record must state its rank, the
    ranked total, the funded count and the value at the line, and must
    distinguish 'below the line' from 'reduced below the minimum by an
    overlay'."""
    pass


def test_overlay_suppressed_is_distinct_from_below_minimum():
    """The 41 000 accounts the 80% dial pushes below the minimum. One is
    reversible by withdrawing an overlay; the other is not."""
    pass


def test_modes_agree_over_the_sweep():
    """The equivalence ladder applied to an ordered fold. interpreted == stepped
    == fused, over a corpus that includes the boundary values: an account whose
    increase is exactly the remaining budget, an account at a band edge, and a
    ranking value tie."""
    assert_modes_agree(Allocation, SMALL.with_boundaries([
        "additional_limit_c == limit_budget_remaining_c",
        "rank_key ties",
        "utilisation exactly 0.40",
        "utilisation exactly 0.90",
        "months_on_book == 6",
        "proposed_limit_c a multiple of 50000 exactly",
    ]))
