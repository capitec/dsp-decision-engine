"""Which accounts can enter a settlement set, and the rule that decided it."""
from consolidation import vocab


def codes(out):
    return {a["account_ref"]: a["settleability_code"] for a in out["classified_accounts"]}


def test_the_sample_inventory_is_classified_by_the_first_matching_rule(assess):
    out = assess()
    assert codes(out) == {
        1: vocab.INTERNAL, 2: vocab.PARTIAL_REVOLVING, 3: vocab.QUOTED, 4: vocab.PARTIAL_REVOLVING,
        5: vocab.PARTIAL_REVOLVING, 6: vocab.QUOTABLE, 7: vocab.PARTIAL_REVOLVING, 8: vocab.BLOCKED_BY_PROVIDER,
        9: vocab.SECURITY_RELEASE}
    assert out["settleable_account_refs"] == [1, 2, 3, 4, 5, 6, 7, 9]
    assert out["unknown_count"] == 0


def test_an_internal_revolving_account_is_partially_settleable_not_internal(assess):
    rules = {a["account_ref"]: a["settleability_rule_id"] for a in assess()["classified_accounts"]}
    assert rules[2] == "CON-SETTLE-REVOLVING" and rules[1] == "CON-SETTLE-INTERNAL"


def test_a_disputed_account_is_blocked_by_status_whatever_else_is_true(assess, sample, account):
    out = assess(accounts=[account(1, is_disputed=True), *sample["accounts"][1:]])
    assert codes(out)[1] == vocab.BLOCKED_BY_STATUS
    assert {a["account_ref"]: a["settleability_rule_id"] for a in out["classified_accounts"]}[1] == "CON-INT-11"


def test_an_account_opened_within_the_policy_window_is_blocked_by_policy(assess, sample, account):
    out = assess(accounts=[account(6, opened_date="2026-08-01"), *[a for a in sample["accounts"] if a["account_ref"] != 6]])
    assert codes(out)[6] == vocab.BLOCKED_BY_POLICY


def test_the_recently_opened_window_is_a_param(assess, sample, account, params):
    accounts = [account(6, opened_date="2026-08-01"), *[a for a in sample["accounts"] if a["account_ref"] != 6]]
    changed = {**params, "consolidation": {**params["consolidation"], "inventory": {"classify": {
        **params["consolidation"]["inventory"]["classify"], "recently_opened_months": 1}}}}
    assert codes(assess(accounts=accounts, params_document=changed))[6] == vocab.QUOTABLE


def test_a_provider_the_table_does_not_know_is_unknown_and_stays_out_of_every_set(assess, sample, account):
    out = assess(accounts=[account(6, provider_code=777), *[a for a in sample["accounts"] if a["account_ref"] != 6]])
    assert codes(out)[6] == vocab.UNKNOWN and out["unknown_count"] == 1
    assert 6 not in out["settleable_account_refs"]
    assert all(6 not in s["settlement_set"] for s in out["scenarios"])


def test_a_provider_that_does_not_quote_third_parties_blocks_its_accounts(assess):
    account = next(a for a in assess()["classified_accounts"] if a["account_ref"] == 8)
    assert (account["settleability_code"], account["settleability_cell_id"]) == (
        vocab.BLOCKED_BY_PROVIDER, "settleability@settleability-2026.09#901")


def test_a_secured_account_is_settleable_only_with_its_security_release(assess):
    nine = next(a for a in assess()["classified_accounts"] if a["account_ref"] == 9)
    assert (nine["settleability_code"], nine["security_release_days"]) == (vocab.SECURITY_RELEASE, 10)


def test_a_held_unexpired_quotation_beats_a_quotable_account_and_an_expired_one_does_not(assess, sample, account):
    assert codes(assess())[3] == vocab.QUOTED
    expired = assess(accounts=[account(3, quotation_expiry_date="2026-09-01"), *[a for a in sample["accounts"] if a["account_ref"] != 3]])
    assert codes(expired)[3] == vocab.QUOTABLE


def test_a_client_exclusion_removes_the_account_from_every_set_and_is_recorded(assess):
    out = assess(client_excluded_settle=[2])
    assert out["excluded_account_refs"] == [2] and 2 not in out["settleable_account_refs"]
    assert all(2 not in s["settlement_set"] for s in out["scenarios"])


def test_a_nominated_account_is_in_every_set(assess):
    out = assess(client_nominated_settle=[5])
    assert out["mandatory_account_refs"] == [5]
    assert out["scenarios"] and all(5 in s["settlement_set"] for s in out["scenarios"])


def test_a_nominated_account_that_cannot_be_settled_is_a_conflict_for_the_consultant(assess):
    out = assess(client_nominated_settle=[8])
    assert out["nomination_conflicts"] == [8]
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [2007])
    assert out["scenarios"] == []


def test_the_settleable_share_of_the_client_instalments_is_reported(assess):
    assert 0.0 < assess()["settleable_obligation_proportion"] < 1.0
