"""The allocation on hand-built books, where every funded set can be worked out by hand."""
import polars as pl
import pytest

from limit_mgmt import reasons
from limit_mgmt.allocation import BudgetInstruction, allocate
from limit_mgmt.vocab import AllocationOutcome as Outcome

# A card account with a R1 000 increase (the smallest worth offering), PD 2%, qualifying for the automatic path.
ACCOUNT = dict(client_id=0, product_code=20, current_limit=10_000.0, increase=1_000.0, probability_of_default=0.02,
               behaviour_grade=2, mob_band=4, is_excluded=False, increase_path_code=1, change_type_code=1,
               decrease_notice_class_code=4, evidence_sufficiency_code=0)

# Offers equal the applied budget, no reserve and ample risk envelopes: only the limit envelope can run out.
PLAIN = dict(over_allocation_factor=1.0, fairness_reserve_pct=0.0, rwa_envelope=1e12, expected_loss_envelope=1e12)


def book(*accounts: dict) -> pl.DataFrame:
    rows = []
    for i, changes in enumerate(accounts, start=1):
        row = {"account_id": i, **ACCOUNT, **changes}
        row["final_proposed_limit"] = row["current_limit"] + row.pop("increase")
        row.setdefault("policy_proposed_limit_unadjusted", row["final_proposed_limit"])
        rows.append(row)
    return pl.DataFrame(rows)


def outcomes(result: pl.DataFrame) -> list[int]:
    return result.sort("account_id")["allocation_outcome_code"].to_list()


def test_accounts_are_funded_best_first_until_the_limit_envelope_is_full():
    scored = book({"probability_of_default": 0.04}, {"probability_of_default": 0.01}, {"probability_of_default": 0.03}, {"probability_of_default": 0.02})
    result, cycle = allocate(scored, BudgetInstruction(applied_limit_budget=2_500.0, **PLAIN))
    ordered = result.sort("allocation_rank")
    assert ordered["account_id"].to_list() == [2, 4, 3, 1]            # lowest PD earns most per rand of limit
    assert ordered["allocation_outcome_code"].to_list() == [Outcome.FUNDED, Outcome.FUNDED, Outcome.TAIL_SKIPPED, Outcome.TAIL_SKIPPED]
    assert ordered["funded_amount"].to_list() == [1_000.0, 1_000.0, 0.0, 0.0]
    assert (cycle["total_ranked"], cycle["funded_count"], cycle["envelope_consumed"], cycle["which_envelope_bound"]) == (4, 2, 2_000.0, "limit")
    assert cycle["funding_line_ranking_value"] == ordered["ranking_value"][1]


def test_an_increase_that_does_not_fit_is_skipped_and_a_smaller_one_further_down_is_still_funded():
    scored = book({"probability_of_default": 0.01}, {"probability_of_default": 0.02, "increase": 2_000.0}, {"probability_of_default": 0.03})
    result, cycle = allocate(scored, BudgetInstruction(applied_limit_budget=2_500.0, **PLAIN))
    assert outcomes(result) == [Outcome.FUNDED, Outcome.TAIL_SKIPPED, Outcome.FUNDED]
    assert cycle["tail_skips"] == 1


def test_no_increase_is_ever_trimmed_to_fit():
    scored = book(*[{"increase": 1_000.0 * (1 + i % 3), "probability_of_default": 0.01 + i / 1000} for i in range(12)])
    result, _ = allocate(scored, BudgetInstruction(applied_limit_budget=9_000.0, **PLAIN))
    for row in result.iter_rows(named=True):
        assert row["funded_amount"] in (0.0, row["final_proposed_limit"] - row["current_limit"])
    assert result["funded_amount"].sum() <= 9_000.0 and result["funded_amount"].sum() > 0


def test_after_the_tail_skip_limit_the_cycle_stops_even_for_accounts_that_would_fit():
    scored = book({"probability_of_default": 0.01}, {"probability_of_default": 0.02, "increase": 5_000.0},
                  {"probability_of_default": 0.03, "increase": 5_000.0}, {"probability_of_default": 0.04})
    result, cycle = allocate(scored, BudgetInstruction(applied_limit_budget=2_000.0, max_tail_skips=1, **PLAIN))
    assert outcomes(result) == [Outcome.FUNDED, Outcome.TAIL_SKIPPED, Outcome.BELOW_LINE, Outcome.BELOW_LINE]  # account 4 fitted
    assert cycle["tail_skips"] == 1
    assert result.sort("account_id")["allocation_reason_code"].to_list() == [None, reasons.R_TAIL_SKIPPED,
                                                                              reasons.R_BELOW_FUNDING_LINE, reasons.R_BELOW_FUNDING_LINE]


@pytest.mark.parametrize("envelope, bound", [({"rwa_envelope": 700.0}, "rwa"), ({"expected_loss_envelope": 10.0}, "expected_loss")])
def test_a_risk_envelope_can_bind_before_the_limit_envelope_and_the_cycle_says_which(envelope, bound):
    scored = book(*[{"probability_of_default": 0.01 * i} for i in range(1, 5)])
    # Each R1 000 increase draws R420, weighs R315 and expects a loss of R420 x PD x 0.74 (R3.11, R6.22, R9.32, ...): two fit R700 of RWA or R10 of loss, three do not.
    result, cycle = allocate(scored, BudgetInstruction(applied_limit_budget=1e9, **{**PLAIN, **envelope}))
    assert cycle["which_envelope_bound"] == bound and cycle["funded_count"] == 2 and cycle["envelope_consumed"] == 2_000.0


def test_nothing_is_reported_as_binding_when_every_ranked_increase_fits():
    result, cycle = allocate(book({}, {}), BudgetInstruction(applied_limit_budget=1e9, **PLAIN))
    assert (cycle["which_envelope_bound"], cycle["tail_skips"], cycle["funded_count"]) == (None, 0, 2)


def test_a_tie_in_ranking_value_goes_to_the_lower_account_id_whatever_the_row_order():
    scored = book({}, {}, {}).with_columns(pl.Series("account_id", [30, 10, 20]))
    result, _ = allocate(scored, BudgetInstruction(applied_limit_budget=1_000.0, **PLAIN))
    assert result.filter(pl.col("funded_amount") > 0)["account_id"].to_list() == [10]
    assert result.sort("allocation_rank").filter(pl.col("allocation_rank") > 0)["account_id"].to_list() == [10, 20, 30]


def test_the_objective_changes_who_comes_first():
    big = {"increase": 5_000.0, "probability_of_default": 0.02, "behaviour_grade": 5, "mob_band": 6}
    small = {"increase": 1_000.0, "probability_of_default": 0.01, "behaviour_grade": 1, "mob_band": 1}

    def first(objective):
        result, _ = allocate(book(big, small), BudgetInstruction(objective=objective, **PLAIN))
        return result.sort("allocation_rank").filter(pl.col("allocation_rank") > 0)["account_id"][0]

    assert [first(o) for o in ("risk_adjusted_return", "expected_value", "policy_priority")] == [2, 1, 2]
    with pytest.raises(ValueError, match="objective"):
        first("whim")


def test_a_segment_the_ranking_starves_is_topped_up_from_the_reserve():
    healthy = [{"probability_of_default": 0.01, "behaviour_grade": 2} for _ in range(10)]
    starved = [{"probability_of_default": 0.05, "behaviour_grade": 6} for _ in range(4)]
    # Main pool R10 000 funds the ten healthy accounts; the R2 500 reserve lifts the starved segment to 40% of the book rate.
    budget = BudgetInstruction(applied_limit_budget=12_500.0, over_allocation_factor=1.0, fairness_reserve_pct=0.2,
                               rwa_envelope=1e12, expected_loss_envelope=1e12)
    result, cycle = allocate(book(*healthy, *starved), budget)
    assert outcomes(result) == [Outcome.FUNDED] * 10 + [Outcome.FUNDED, Outcome.FUNDED, Outcome.TAIL_SKIPPED, Outcome.TAIL_SKIPPED]
    assert (cycle["funded_count"], cycle["envelope_consumed"], cycle["starved_segments"]) == (12, 12_000.0, [])


def test_a_segment_the_reserve_cannot_lift_is_marked_fairness_capped():
    healthy = [{"probability_of_default": 0.01, "behaviour_grade": 2} for _ in range(10)]
    starved = [{"probability_of_default": 0.05, "behaviour_grade": 6, "increase": 1_800.0} for _ in range(4)]
    budget = BudgetInstruction(applied_limit_budget=12_500.0, over_allocation_factor=1.0, fairness_reserve_pct=0.2,
                               rwa_envelope=1e12, expected_loss_envelope=1e12)
    result, cycle = allocate(book(*healthy, *starved), budget)
    assert outcomes(result)[10:] == [Outcome.FUNDED, Outcome.FAIRNESS_CAPPED, Outcome.FAIRNESS_CAPPED, Outcome.FAIRNESS_CAPPED]
    assert cycle["starved_segments"] == [(20, 6, 4)] and cycle["envelope_consumed"] == 11_800.0
    assert set(result.filter(pl.col("allocation_outcome_code") == Outcome.FAIRNESS_CAPPED)["allocation_reason_code"]) == {reasons.R_FAIRNESS_CAPPED}


def test_an_immediate_decrease_on_one_account_suppresses_the_clients_other_increases():
    scored = book({"client_id": 1, "change_type_code": 3, "decrease_notice_class_code": 1, "is_excluded": True, "increase": -4_000.0},
                  {"client_id": 1}, {"client_id": 2})
    result, cycle = allocate(scored, BudgetInstruction(**PLAIN))
    assert outcomes(result)[1:] == [Outcome.SUPPRESSED, Outcome.FUNDED]
    suppressed = result.filter(pl.col("account_id") == 2).row(0, named=True)
    assert (suppressed["allocation_rank"], suppressed["funded_amount"], suppressed["allocation_reason_code"]) == (0, 0.0, reasons.R_SUPPRESSED_BY_CLIENT_DECREASE)
    assert cycle["total_ranked"] == 1


def test_a_decrease_with_a_notice_period_does_not_suppress_the_clients_other_increases():
    scored = book({"client_id": 1, "change_type_code": 3, "decrease_notice_class_code": 2, "increase": -4_000.0}, {"client_id": 1})
    result, _ = allocate(scored, BudgetInstruction(**PLAIN))
    assert outcomes(result)[1] == Outcome.FUNDED


def test_a_granted_limit_is_the_current_limit_plus_what_was_funded_or_the_decrease_target():
    scored = book({}, {"probability_of_default": 0.9}, {"change_type_code": 3, "decrease_notice_class_code": 1, "client_id": 9, "increase": -4_000.0})
    result, _ = allocate(scored, BudgetInstruction(applied_limit_budget=1_000.0, **PLAIN))
    assert result.sort("account_id")["granted_limit"].to_list() == [11_000.0, 10_000.0, 6_000.0]


def test_every_account_not_funded_says_why():
    scored = book(
        {"is_excluded": True},                                            # 1: excluded
        {"increase_path_code": 3},                                        # 2: client cannot carry the increase
        {"increase_path_code": 3, "evidence_sufficiency_code": 4},        # 3: affordability not established
        {"increase_path_code": 2},                                        # 4: waits for the client's income confirmation
        {"increase": 700.0, "policy_proposed_limit_unadjusted": 11_200.0},  # 5: the dial took it under R1 000
        {"increase": 700.0},                                              # 6: under R1 000 anyway
        {"increase": 0.0},                                                # 7: the matrix and caps left nothing
    )
    result, _ = allocate(scored, BudgetInstruction(**PLAIN))
    reason = dict(zip(result["account_id"].to_list(), result["allocation_reason_code"].to_list()))
    assert reason == {1: reasons.R_EXCLUDED, 2: reasons.R_AFFORDABILITY_FAIL, 3: reasons.R_AFFORDABILITY_UNESTABLISHED, 4: None,
                      5: reasons.R_REDUCED_BELOW_MINIMUM_BY_OVERLAY, 6: reasons.R_BELOW_MINIMUM_MEANINGFUL_INCREASE,
                      7: reasons.R_NO_INCREASE_AFTER_CAPS}
    assert set(result["allocation_outcome_code"]) == {Outcome.NOT_RANKED}


def test_an_empty_ranking_is_a_valid_cycle():
    result, cycle = allocate(book({"is_excluded": True}), BudgetInstruction(**PLAIN))
    assert (cycle["total_ranked"], cycle["funded_count"], cycle["funding_line_ranking_value"]) == (0, 0, None)
    assert result["allocation_outcome_code"].to_list() == [Outcome.NOT_RANKED]
