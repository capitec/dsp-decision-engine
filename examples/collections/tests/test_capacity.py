import copy

import polars as pl
import pytest

from collections_treatment import capacity, vocab as v
from collections_treatment.capacity import allocate_capacity


def account(account_id, *, blocker=0, pool=v.POOL_EARLY_AGENTS, supply=1_000, share=0.0, balance_band=3, recovery=1000.0,
            cost=100.0, priority=30.0):
    return dict(account_id=account_id, selection_blocker_code=blocker, pool_code=pool, pool_supply=float(supply),
                pool_reserved_share=share, balance_band_code=balance_band, recovery_estimate=recovery,
                cost_to_collect=cost, allocation_priority=priority)


def allocate(accounts, **params):
    return allocate_capacity(pl.DataFrame(accounts), **params).sort("account_id")


def test_accounts_past_the_cutoff_get_reason_250_with_their_rank_and_the_cutoff_rank():
    out = allocate([account(i, supply=2, recovery=1000.0 - i) for i in range(5)])
    assert out["allocated"].to_list() == [True, True, False, False, False]
    assert out["allocation_rank"].to_list() == [1, 2, 3, 4, 5]
    assert out["allocation_cutoff_rank"].to_list() == [2] * 5
    assert out["non_selection_reason_code"].to_list() == [None, None, 250, 250, 250]


def test_the_higher_expected_recovery_net_of_cost_is_allocated_first():
    out = allocate([account(1, supply=1, recovery=500.0, cost=400.0), account(2, supply=1, recovery=300.0, cost=10.0)])
    assert out["allocated"].to_list() == [False, True]


def test_an_account_stopped_before_ranking_keeps_its_reason_and_takes_no_capacity():
    out = allocate([account(1, blocker=v.NS_SUSPENDED, supply=1), account(2, supply=1)])
    assert out["allocated"].to_list() == [False, True]
    assert out["non_selection_reason_code"].to_list() == [v.NS_SUSPENDED, None]
    assert out["allocation_rank"].to_list() == [None, 1]


def test_ties_break_on_account_id_so_a_rerun_gives_the_same_allocation_whatever_the_input_order():
    accounts = [account(i, supply=3, recovery=500.0) for i in range(6)]
    forward, shuffled = allocate(accounts), allocate(accounts[3:] + accounts[:3])
    assert forward["allocation_rank"].to_list() == shuffled["allocation_rank"].to_list() == [1, 2, 3, 4, 5, 6]
    assert forward["allocated"].to_list() == [True] * 3 + [False] * 3


def test_each_pool_is_rationed_on_its_own_supply():
    out = allocate([account(1, pool=v.POOL_SMS, supply=1), account(2, pool=v.POOL_SMS, supply=1),
                    account(3, pool=v.POOL_LEGAL, supply=1)])
    assert out["allocated"].to_list() == [True, False, True]


def test_high_balance_accounts_keep_their_reserved_share_against_low_balance_volume():
    low_balance_high_value = [account(100 + i, balance_band=2, recovery=9_999.0, supply=5, share=0.4) for i in range(5)]
    high_balance_low_value = [account(i, balance_band=6, recovery=10.0, supply=5, share=0.4) for i in range(2)]
    out = allocate(low_balance_high_value + high_balance_low_value)
    assert out.filter(pl.col("balance_band_code") == 6)["allocated"].to_list() == [True, True]
    assert out["allocated"].sum() == 5


def test_without_a_reserved_share_the_same_high_balance_accounts_are_crowded_out():
    out = allocate([account(100 + i, balance_band=2, recovery=9_999.0, supply=5) for i in range(5)]
                   + [account(i, balance_band=6, recovery=10.0, supply=5) for i in range(2)])
    assert out.filter(pl.col("balance_band_code") == 6)["allocated"].to_list() == [False, False]


def test_the_priority_basis_ranks_by_policy_not_by_value_and_records_which_basis_was_used():
    accounts = [account(1, supply=1, recovery=9_000.0, priority=30.0), account(2, supply=1, recovery=10.0, priority=1_030.0)]
    assert allocate(accounts)["allocated"].to_list() == [True, False]
    out = allocate(accounts, ranking_basis="priority")
    assert out["allocated"].to_list() == [False, True]
    assert out["allocation_ranking_basis"].to_list() == ["priority"] * 2


def test_an_unknown_ranking_basis_is_refused():
    with pytest.raises(ValueError, match="ranking_basis"):
        allocate([account(1)], ranking_basis="whim")


def test_a_treatment_with_no_pool_is_never_rationed():
    out = allocate([account(1, pool=v.NO_POOL, supply=0)])
    assert out["allocated"].to_list() == [True] and out["non_selection_reason_code"].to_list() == [None]


def test_an_empty_day_allocates_nothing():
    assert allocate([account(1)]).head(0).height == 0


@pytest.mark.parametrize("fields, reason", [
    (dict(suppression_adjustments_applied=["ADJ-1"]), v.NS_OVERLAY_SUPPRESSED),
    (dict(overlaid_treatment_code=v.NO_ACTION), v.NS_MATRIX_NO_ACTION),
    (dict(suspension_blocked=True, suspension_codes=[101]), v.NS_SUSPENDED),
    (dict(suspension_blocked=True, suspension_codes=[119]), v.NS_PROMISE_IN_FORCE),
    (dict(interval_or_cap_detail="cooling_off", interval_or_cap_blocked=True), v.NS_COOLING_OFF),
    (dict(interval_or_cap_detail="daily_cap", interval_or_cap_blocked=True), v.NS_INTERVAL_OR_CAP),
    (dict(), 0),
])
def test_the_reason_an_account_is_stopped_before_ranking(fields, reason):
    inputs = dict(matrix_treatment_code=5, overlaid_treatment_code=5, suspension_blocked=False,
                  suppression_adjustments_applied=[], suspension_codes=[], interval_or_cap_blocked=False,
                  interval_or_cap_detail="") | fields
    assert capacity.selection_blocker_code(**inputs) == reason


def test_the_overlay_reason_is_reported_before_a_suspension_on_the_same_account():
    assert capacity.selection_blocker_code(
        5, v.NO_ACTION, True, ["ADJ-1"], [101], False, "") == v.NS_OVERLAY_SUPPRESSED


# Through the whole pipeline: the supply is a row of the params document, so a revised supply needs no rebuild.
def population(parse, sample, size):
    return pl.DataFrame([parse(sample | {"account_id": 900_000 + i, "consent_withdrawn_channels": []})
                         for i in range(size)])


def with_supply(params, pool, supply):
    revised = copy.deepcopy(params)
    for row in revised["collections_treatment"]["capacity"]["pools"]:
        if row["pool_code"] == pool:
            row["pool_supply"] = supply
    return revised


def test_a_revised_pool_supply_in_the_params_document_changes_who_is_allocated(pipeline, params, parse, sample):
    day = population(parse, sample, 4)
    out = pipeline.run(day, params=with_supply(params, v.POOL_VOICE_MESSAGE, 2)).sort("account_id")
    assert out["pool_code"].to_list() == [v.POOL_VOICE_MESSAGE] * 4
    assert out["allocated"].to_list() == [True, True, False, False]
    assert out["non_selection_reason_code"].to_list() == [None, None, v.NS_BELOW_CUTOFF, v.NS_BELOW_CUTOFF]
    assert pipeline.run(day, params=params)["allocated"].to_list() == [True] * 4


def test_the_ranking_basis_is_switched_in_the_params_document(pipeline, params, parse, sample):
    day = pl.concat([population(parse, sample | {"days_since_last_treatment": 12}, 1),
                     population(parse, sample | {"days_since_last_treatment": 12, "is_new_bucket3_entrant": False,
                                                 "outstanding_balance": 60_000.0, "arrears_amount": 100.0}, 1)
                     .with_columns(account_id=pl.lit(900_001))], how="diagonal_relaxed")
    tight = with_supply(params, v.POOL_VOICE_MESSAGE, 1)
    by_value = pipeline.run(day, params=tight).sort("account_id")
    switched = copy.deepcopy(tight)
    switched["collections_treatment"]["allocate_capacity"]["ranking_basis"] = "priority"
    by_priority = pipeline.run(day, params=switched).sort("account_id")
    assert by_value["allocation_ranking_basis"].to_list() == ["value"] * 2
    assert by_priority["allocation_ranking_basis"].to_list() == ["priority"] * 2
    assert by_priority["allocated"].to_list() == [True, False]  # the new bucket-3 entrant outranks the richer account
