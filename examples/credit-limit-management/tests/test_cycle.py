"""The whole monthly cycle over a synthetic book: the per-account pipeline compiled, then the allocation."""
from datetime import date

import polars as pl
import pytest

from conftest import switch_off_every_overlay
from limit_mgmt import reasons
from limit_mgmt.allocation import BudgetInstruction
from limit_mgmt.simulation import run_cycle, swap_set
from limit_mgmt.vocab import AllocationOutcome as Outcome
from population import make_book

N = 2_000
# About half of the book's R834 500 of proposed increases fits in the offer envelope of R345 000.
TIGHT = BudgetInstruction(applied_limit_budget=250_000.0, fairness_reserve_pct=0.0)
AMPLE = BudgetInstruction(applied_limit_budget=1e12, rwa_envelope=1e15, expected_loss_envelope=1e15)


@pytest.fixture(scope="module")
def snapshot():
    return make_book(N)


@pytest.fixture(scope="module")
def production(snapshot):
    return run_cycle(snapshot, budget=TIGHT)


def test_every_account_gets_a_decision_excluded_ones_included(production):
    book = production.book
    assert book.height == N and book["account_id"].n_unique() == N
    excluded = book.filter(pl.col("is_excluded"))
    assert excluded.height > 100 and (excluded["funded_amount"] == 0).all() and (excluded["final_proposed_limit"] > 0).all()


def test_the_budget_binds_so_some_qualifying_increases_are_not_funded(production):
    cycle, book = production.summary, production.book
    assert cycle["which_envelope_bound"] == "limit" and cycle["tail_skips"] > 0
    assert 100 < cycle["funded_count"] < cycle["total_ranked"]
    assert cycle["envelope_consumed"] == book["funded_amount"].sum() <= cycle["offer_envelope"]
    assert cycle["offer_envelope"] - cycle["envelope_consumed"] < 10_000  # the budget is used almost to the rand


def test_the_funded_set_is_exactly_what_walking_the_ranking_against_the_budget_gives(production):
    ranked = production.book.filter(pl.col("allocation_rank") > 0).sort("allocation_rank")
    assert ranked["allocation_rank"].to_list() == list(range(1, ranked.height + 1))
    assert ranked["ranking_value"].is_sorted(descending=True)
    spent, expected = 0.0, []
    for increase in (ranked["final_proposed_limit"] - ranked["current_limit"]).to_list():
        fits = spent + increase <= production.summary["offer_envelope"] * 1.0
        expected.append(fits)
        spent += increase if fits else 0.0
    assert (ranked["allocation_outcome_code"] == Outcome.FUNDED).to_list() == expected


def test_a_funded_account_is_given_exactly_its_proposed_increase(production):
    funded = production.book.filter(pl.col("allocation_outcome_code") == Outcome.FUNDED)
    assert (funded["funded_amount"] == funded["final_proposed_limit"] - funded["current_limit"]).all()
    assert (funded["granted_limit"] == funded["final_proposed_limit"]).all()


def test_every_unfunded_account_says_why(production):
    unfunded = production.book.filter(pl.col("allocation_outcome_code") != Outcome.FUNDED)
    assert unfunded.filter(pl.col("allocation_reason_code").is_null() & (pl.col("increase_path_code") != 2)
                           & (pl.col("change_type_code") != 3)).height == 0
    ranked_out = unfunded.filter(pl.col("allocation_rank") > 0)
    assert set(ranked_out["allocation_reason_code"]) <= {reasons.R_TAIL_SKIPPED, reasons.R_BELOW_FUNDING_LINE}


def test_a_client_with_an_immediate_decrease_has_no_increase_funded_on_any_account(production):
    book = production.book
    decreased = book.filter(pl.col("change_type_code") == 3)["client_id"]
    suppressed = book.filter(pl.col("client_id").is_in(decreased.implode()) & (pl.col("change_type_code") != 3))
    assert decreased.len() > 20 and (suppressed["funded_amount"] == 0).all()
    assert (suppressed["allocation_outcome_code"] == Outcome.SUPPRESSED).sum() > 0


def test_a_smaller_budget_funds_fewer_accounts_and_less_limit(snapshot, production):
    half = run_cycle(snapshot, budget=BudgetInstruction(applied_limit_budget=125_000.0, fairness_reserve_pct=0.0))
    assert half.summary["funded_count"] < production.summary["funded_count"]
    assert half.summary["envelope_consumed"] < production.summary["envelope_consumed"]


def test_running_the_cycle_again_reproduces_every_rank_and_amount(snapshot, production):
    replay = run_cycle(snapshot, budget=TIGHT)
    assert (swap_set(production, replay)["swap_class"] == "unaffected").all()
    ranks = production.book.select("account_id", "allocation_rank").join(replay.book.select("account_id", "allocation_rank"),
                                                                         on="account_id", suffix="_replay")
    assert (ranks["allocation_rank"] == ranks["allocation_rank_replay"]).all()


def test_the_fairness_reserve_is_spent_within_its_slice_and_what_it_cannot_lift_is_reported_as_starved(snapshot):
    result = run_cycle(snapshot, budget=BudgetInstruction(applied_limit_budget=250_000.0))
    cycle, segment = result.summary, ["product_code", "behaviour_grade", "mob_band"]
    assert 0 < cycle["reserve_consumed"] <= cycle["offer_envelope"] * 0.15
    capped = result.book.filter(pl.col("allocation_outcome_code") == Outcome.FAIRNESS_CAPPED)
    assert capped.height > 0 and set(capped.select(segment).iter_rows()) == set(cycle["starved_segments"])
    assert cycle["envelope_consumed"] <= cycle["offer_envelope"]
    assert result.book["funded_amount"].sum() == pytest.approx(cycle["envelope_consumed"])


def test_a_cycle_cap_trims_the_increases_and_lets_accounts_below_the_line_in(snapshot, production):
    cap = {"limit_mgmt": {"matrix": {"apply_cycle_cap": {"cycle_cap_enabled": True, "cycle_cap_amount": 3_000.0}}}}
    capped = run_cycle(snapshot, params=cap, budget=TIGHT)
    swap = swap_set(production, capped)
    assert (swap.filter(pl.col("swap_class") == "loser").height, swap.filter(pl.col("swap_class") == "gainer").height) > (0, 0)
    assert capped.summary["funded_count"] > production.summary["funded_count"]  # smaller increases, same money, more clients


def test_with_ample_budget_a_cycle_cap_only_creates_losers(snapshot):
    cap = {"limit_mgmt": {"matrix": {"apply_cycle_cap": {"cycle_cap_enabled": True, "cycle_cap_amount": 3_000.0}}}}
    swap = swap_set(run_cycle(snapshot, budget=AMPLE), run_cycle(snapshot, params=cap, budget=AMPLE))
    assert swap.filter(pl.col("swap_class") == "gainer").height == 0 and swap.filter(pl.col("swap_class") == "loser").height > 0


def test_switching_the_overlays_off_changes_the_whole_book_and_only_upwards(snapshot, params):
    on, off = run_cycle(snapshot, budget=AMPLE), run_cycle(snapshot, params=switch_off_every_overlay(params), budget=AMPLE)
    swap = swap_set(on, off)
    assert swap.filter(pl.col("swap_class") == "loser").height == 0 and swap.filter(pl.col("swap_class") == "gainer").height > 0
    assert (off.scored["matrix_multiplier"] == off.scored["matrix_multiplier_unadjusted"]).all()
    assert (on.scored["matrix_multiplier"] <= on.scored["matrix_multiplier_unadjusted"]).all()


def test_the_dial_can_drop_an_increase_below_the_minimum_and_the_reason_says_so(snapshot, params):
    on = run_cycle(snapshot, budget=AMPLE).book["allocation_reason_code"]
    off = run_cycle(snapshot, params=switch_off_every_overlay(params), budget=AMPLE).book["allocation_reason_code"]
    assert (on == reasons.R_REDUCED_BELOW_MINIMUM_BY_OVERLAY).sum() > 0 and (off == reasons.R_REDUCED_BELOW_MINIMUM_BY_OVERLAY).sum() == 0


def test_an_overlay_past_its_review_date_stops_the_cycle(snapshot):
    late = snapshot.head(10).with_columns(pl.lit(date(2026, 10, 20)).alias("decision_date"))
    with pytest.raises(ValueError, match="ADJ-07-DIAL-2026-03"):
        run_cycle(late)
