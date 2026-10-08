from datetime import date

import pytest

from credit_core import obligations as ob
from credit_core.obligations import assess, obligations


def account(**fields):
    return {"account_type_code": 20, "balance": 0.0, "instalment": 0.0, "is_internal": False, **fields}


TREATMENTS = [
    # what the account looks like, expected monthly cost, treatment code
    (account(instalment=1200.0), 1200.0, ob.USE_STATED_INSTALMENT),
    (account(account_type_code=10, limit=20000.0, balance=5000.0), 600.0, ob.IMPUTE_PCT_OF_LIMIT),  # 3% of the limit
    (account(account_type_code=12, balance=10000.0), 500.0, ob.IMPUTE_PCT_OF_BALANCE),  # 5% of the balance
    (account(account_type_code=11, balance=4000.0), 100.0, ob.IMPUTE_MINIMUM_PAYMENT_PCT),  # 2.5% of the balance
    (account(account_type_code=21, balance=36000.0), 1000.0, ob.AMORTISE_BALANCE_OVER_STANDARD_TERM),  # over 36 months
    (account(account_type_code=31, instalment=900.0), 0.0, ob.EXCLUDE),
    (account(account_type_code=30, settlement_quote=9000.0), 0.0, ob.EXCLUDE_IF_SETTLEMENT_IN_FLIGHT),
    (account(account_type_code=30, instalment=800.0), 800.0, ob.USE_STATED_INSTALMENT),  # no settlement in flight
    (account(settlement_quote=18000.0, instalment=1200.0), 500.0, ob.USE_SETTLEMENT_QUOTE),  # quote over 36 months
    (account(closed=True, instalment=1200.0), 0.0, ob.EXCLUDE_IF_CLOSED),
    (account(account_type_code=40, is_internal=True, instalment=700.0), 0.0, ob.EXCLUDE_INTERNAL_BEING_CONSOLIDATED),
    (account(account_type_code=40, is_internal=False, instalment=700.0), 700.0, ob.USE_STATED_INSTALMENT),
]


@pytest.mark.parametrize("acc,cost,treatment", TREATMENTS)
def test_each_account_is_costed_by_its_treatment(acc, cost, treatment):
    result = assess([acc], [])
    assert result.obligation_monthly_amounts == [cost]
    assert result.obligation_treatment_codes == [treatment]
    assert result.existing_obligations == cost


def test_the_ten_treatments_are_all_exercised():
    assert {t for _, _, t in TREATMENTS} == set(range(1, 11))


BUREAU_CARD = account(account_type_code=10, balance=8000.0, limit=20000.0, instalment=400.0, opened_date=date(2022, 1, 1))
LOAN = {"account_type_code": 1, "balance": 15000.0, "instalment": 900.0, "opened_date": date(2021, 5, 1)}


def test_an_account_on_both_lists_is_counted_once_and_the_internal_record_wins():
    result = assess([BUREAU_CARD, {**LOAN, "is_internal": False, "months_in_arrears": 0}],
                    [{**LOAN, "is_internal": True, "months_in_arrears": 2}])
    assert result.obligation_account_type_codes == [1, 10]
    assert result.obligation_is_internal == [True, False]
    assert result.existing_obligations == 900.0 + 600.0
    assert result.worst_arrears_months == 2  # only the internal copy carries the arrears


def test_same_type_and_date_but_another_balance_is_a_different_account():
    other = {**LOAN, "balance": 14000.0, "is_internal": False}
    assert len(assess([other], [{**LOAN, "is_internal": True}]).obligation_account_type_codes) == 2


def test_totals_split_by_where_the_account_is_held():
    result = assess([BUREAU_CARD], [{**LOAN, "is_internal": True}])
    assert (result.obligations_internal, result.obligations_external, result.existing_obligations) == (900.0, 600.0, 1500.0)
    assert result.total_exposure == 23000.0


def test_utilisation_counts_revolving_accounts_only():
    result = assess([BUREAU_CARD], [{**LOAN, "is_internal": True}])
    assert result.revolving_utilisation == 0.4  # 8 000 of 20 000; the loan has no limit and is excluded


def test_arrears_are_summarised():
    result = assess([account(months_in_arrears=1), account(months_in_arrears=3, opened_date=date(2020, 1, 1)), account()], [])
    assert (result.worst_arrears_months, result.accounts_in_arrears_count) == (3, 2)


def test_no_accounts_costs_nothing():
    result = assess([], [])
    assert result.existing_obligations == 0.0 and result.obligation_account_type_codes == []
    assert result.worst_arrears_months == 0 and result.revolving_utilisation == 0.0


def test_ninety_five_accounts_are_summed():
    accounts = [account(instalment=10.0, balance=float(i)) for i in range(95)]
    assert assess(accounts, []).existing_obligations == 950.0


def test_the_step_returns_both_shapes_and_defaults_missing_lists_to_empty(score):
    out = score(obligations, {"bureau_accounts": [BUREAU_CARD], "internal_accounts": [{**LOAN, "is_internal": True}]})
    assert out["existing_obligations"] == 1500.0
    assert out["obligation_monthly_amounts"] == [900.0, 600.0]
    assert score(obligations, {})["existing_obligations"] == 0.0
