from datetime import date

from consolidation import inventory, vocab


def _account(**kw):
    base = dict(
        account_ref=1, account_type_code=10, provider_code=101, is_internal=False, balance=1000.0,
        credit_limit=5000.0, instalment=100.0, nominal_annual_rate=0.2, remaining_term_months=None,
        months_in_arrears=0, opened_date=date(2020, 1, 1), is_secured=False, security_type_code=None,
        account_status_code=0, is_disputed=False, quotation_reference="", quotation_expiry_date=date(2000, 1, 1),
    )
    base.update(kw)
    return base


def test_internal_account_is_settle_internal():
    out = inventory.classify_settleability([_account(is_internal=True)], date(2026, 9, 24), [], [])
    assert out[0]["settleability_code"] == vocab.SETTLE_INTERNAL


def test_disputed_account_is_blocked_regardless_of_other_attributes():
    out = inventory.classify_settleability([_account(is_disputed=True)], date(2026, 9, 24), [], [])
    assert out[0]["settleability_code"] == vocab.SETTLE_BLOCKED_STATUS


def test_secured_account_needs_release():
    out = inventory.classify_settleability([_account(is_secured=True)], date(2026, 9, 24), [], [])
    assert out[0]["settleability_code"] == vocab.SETTLE_SECURITY_RELEASE


def test_recently_opened_account_is_blocked_by_policy():
    out = inventory.classify_settleability(
        [_account(opened_date=date(2026, 9, 1))], date(2026, 9, 24), [], [], recently_opened_months=3,
    )
    assert out[0]["settleability_code"] == vocab.SETTLE_BLOCKED_POLICY


def test_held_quotation_beats_quotable():
    out = inventory.classify_settleability(
        [_account(account_type_code=20, quotation_reference="Q-1", quotation_expiry_date=date(2027, 1, 1))],
        date(2026, 9, 24), [], [],
    )
    assert out[0]["settleability_code"] == vocab.SETTLE_QUOTED


def test_revolving_account_is_partially_settleable():
    out = inventory.classify_settleability([_account(account_type_code=10)], date(2026, 9, 24), [], [])
    assert out[0]["settleability_code"] == vocab.SETTLE_PARTIAL_REVOLVING


def test_missing_provider_on_external_account_is_unknown():
    out = inventory.classify_settleability([_account(provider_code=None)], date(2026, 9, 24), [], [])
    assert out[0]["settleability_code"] == vocab.SETTLE_UNKNOWN


def test_client_exclusion_is_recorded_and_removed_from_candidacy():
    row = {"accounts": [_account(account_ref=7, account_type_code=20)], "decision_date": date(2026, 9, 24),
           "client_nominated_settle": [], "client_excluded_settle": [7]}
    result = inventory._process(row, recently_opened_months=3)
    assert 7 in result["client_excluded_account_refs"]
    assert 7 not in result["settleable_account_refs"]


def test_mandatory_account_recorded_when_settleable():
    row = {"accounts": [_account(account_ref=3, account_type_code=20)], "decision_date": date(2026, 9, 24),
           "client_nominated_settle": [3], "client_excluded_settle": []}
    result = inventory._process(row, recently_opened_months=3)
    assert result["mandatory_account_refs"] == [3]
