"""§5.9: rank and ration, deterministically, with a per-account non-selection reason
for everyone left out, and a fairness floor for high-balance accounts."""
import polars as pl

from collections_treatment import capacity_alloc as ca
from collections_treatment import vocab


def _account(account_id, treatment_code=vocab.AGENT_CALL_LOW, balance_band_code=3,
             recovery_estimate=1000.0, cost_to_collect=100.0, **overrides):
    row = dict(
        account_id=account_id, treatment_code=treatment_code, arrears_bucket_code=3,
        balance_band_code=balance_band_code, recovery_estimate=recovery_estimate,
        cost_to_collect=cost_to_collect, suspended_blocks_all=False,
        suppression_adjustments_applied=[], interval_or_cap_blocked=False,
        interval_or_cap_detail="", suspension_codes=[], pre_prescription_flag=False,
        is_new_bucket3_entrant=False,
    )
    row.update(overrides)
    return row


def test_below_capacity_cutoff_is_reported_with_rank_and_cutoff(monkeypatch):
    monkeypatch.setitem(ca.CAPACITY_SUPPLY, "early_agents", (2, 0.0))
    rows = [_account(i, recovery_estimate=1000.0 - i) for i in range(5)]
    df = ca.allocate(pl.DataFrame(rows), ranking_basis=ca.BASIS_VALUE)
    allocated = df.filter(pl.col("allocated")).sort("account_id")
    not_allocated = df.filter(~pl.col("allocated"))
    assert allocated["account_id"].to_list() == [0, 1]  # highest value, deterministic
    assert set(not_allocated["non_selection_reason_code"].to_list()) == {vocab.NS_BELOW_CUTOFF}
    assert not_allocated["allocation_cutoff_rank"].to_list() == [2, 2, 2]


def test_matrix_no_action_is_never_ranked():
    df = ca.allocate(pl.DataFrame([_account(1, treatment_code=vocab.NO_ACTION)]))
    row = df.row(0, named=True)
    assert row["allocated"] is False
    assert row["non_selection_reason_code"] == vocab.NS_MATRIX_NO_ACTION


def test_suspended_account_is_never_ranked_even_with_capacity_to_spare(monkeypatch):
    monkeypatch.setitem(ca.CAPACITY_SUPPLY, "early_agents", (10, 0.0))
    df = ca.allocate(pl.DataFrame([_account(1, suspended_blocks_all=True)]))
    row = df.row(0, named=True)
    assert row["allocated"] is False
    assert row["non_selection_reason_code"] == vocab.NS_SUSPENDED


def test_determinism_same_input_same_ranks_every_run(monkeypatch):
    monkeypatch.setitem(ca.CAPACITY_SUPPLY, "early_agents", (3, 0.0))
    rows = [_account(i, recovery_estimate=500.0) for i in range(6)]  # tied value -> account_id tie-break
    first = ca.allocate(pl.DataFrame(rows)).sort("account_id")
    second = ca.allocate(pl.DataFrame(rows)).sort("account_id")
    assert first["allocation_rank"].to_list() == second["allocation_rank"].to_list()
    # Rank is recorded for everyone, allocated or not -- §5.9 code 250's own example
    # ("you were 91 412th of 186 000") is a rank on an *unallocated* account.
    assert first["allocation_rank"].to_list() == [1, 2, 3, 4, 5, 6]
    assert first["allocated"].to_list() == [True, True, True, False, False, False]


def test_high_balance_reserved_share_is_not_crowded_out_by_low_value(monkeypatch):
    """§5.9: "at least 8% of agent capacity is reserved for balance bands 6-7" -- without
    the reservation, low-band high-value accounts would take every slot."""
    monkeypatch.setitem(ca.CAPACITY_SUPPLY, "early_agents", (5, 0.4))  # 40% reserved for this test
    rows = [_account(i, balance_band_code=6, recovery_estimate=10.0) for i in range(2)]  # low value, high balance
    rows += [_account(100 + i, balance_band_code=2, recovery_estimate=9999.0) for i in range(5)]  # high value, low balance
    df = ca.allocate(pl.DataFrame(rows))
    allocated_high_balance = df.filter((pl.col("balance_band_code") == 6) & pl.col("allocated"))
    assert len(allocated_high_balance) == 2  # both high-balance accounts got in via the reserve
