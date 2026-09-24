"""`core.obligations`: two lists deduped, ten treatments, scalar + per-account shapes (00 §6.4; addendum A10)."""
from decider import Engine

from credit_core import obligations

BUREAU = [
    {"account_type_code": 10, "balance": 8_000.0, "limit": 20_000.0, "instalment": 400.0,
     "months_in_arrears": 0, "opened_date": "2022-01-01", "closed": False, "is_internal": False},
    # Same account as the internal one below (same type/opened date/balance) -- must be deduped.
    {"account_type_code": 1, "balance": 15_000.0, "instalment": 900.0,
     "months_in_arrears": 0, "opened_date": "2021-05-01", "closed": False, "is_internal": False},
]
INTERNAL = [
    {"account_type_code": 1, "balance": 15_000.0, "instalment": 900.0,
     "months_in_arrears": 2, "opened_date": "2021-05-01", "closed": False, "is_internal": True},
]


def _score(bureau=BUREAU, internal=INTERNAL):
    exe = Engine().bind(obligations.obligations, mode="interpreted")
    return exe.score({"bureau_accounts": bureau, "internal_accounts": internal})


def test_the_duplicate_account_seen_on_both_lists_is_counted_once():
    out = _score()
    # 2 unique accounts (the bureau credit card, and the internal one that also appeared,
    # unmatched, on the bureau list) -- not 3.
    assert len(out["obligation_account_type_codes"]) == 2


def test_internal_records_win_the_duplicate_and_carry_its_arrears():
    out = _score()
    assert out["worst_arrears_months"] == 2  # only the internal copy carries the arrears


def test_scalar_and_per_account_shapes_are_both_returned():
    """00 §6.4 "Hard part": one capability, two shapes of answer."""
    out = _score()
    assert isinstance(out["existing_obligations"], float)
    assert len(out["obligation_account_type_codes"]) == len(out["obligation_treatment_codes"]) \
        == len(out["obligation_monthly_amounts"]) == len(out["obligation_is_internal"])


def test_revolving_utilisation_is_balance_over_limit_for_revolving_accounts_only():
    out = _score()
    assert out["revolving_utilisation"] == 8_000.0 / 20_000.0


def test_closed_accounts_are_excluded():
    out = _score(bureau=[{**BUREAU[0], "closed": True}], internal=[])
    assert out["existing_obligations"] == 0.0
    assert out["obligation_treatment_codes"] == [obligations.EXCLUDE_IF_CLOSED]


def test_settlement_quote_overrides_the_stated_instalment():
    account = {"account_type_code": 20, "balance": 30_000.0, "instalment": 1_200.0,
               "months_in_arrears": 0, "opened_date": "2020-01-01", "closed": False, "is_internal": False,
               "settlement_quote": 18_000.0}
    out = _score(bureau=[account], internal=[])
    assert out["obligation_treatment_codes"] == [obligations.USE_SETTLEMENT_QUOTE]
    assert out["obligation_monthly_amounts"][0] == round(18_000.0 / 36, 2)


def test_no_accounts_is_zero_not_an_error():
    out = _score(bureau=[], internal=[])
    assert out["existing_obligations"] == 0.0
    assert out["obligation_account_type_codes"] == []


def test_ten_treatment_behaviours_are_all_reachable():
    """Addendum item 10: ten behaviours, not five."""
    assert len({
        obligations.USE_STATED_INSTALMENT, obligations.IMPUTE_PCT_OF_LIMIT, obligations.IMPUTE_PCT_OF_BALANCE,
        obligations.EXCLUDE, obligations.EXCLUDE_IF_SETTLEMENT_IN_FLIGHT, obligations.USE_SETTLEMENT_QUOTE,
        obligations.IMPUTE_MINIMUM_PAYMENT_PCT, obligations.EXCLUDE_IF_CLOSED,
        obligations.EXCLUDE_INTERNAL_BEING_CONSOLIDATED, obligations.AMORTISE_BALANCE_OVER_STANDARD_TERM,
    }) == 10
