"""The household's existing debts, treated account by account by `credit_core.obligations`."""
from __future__ import annotations

from decider import allow_fallback, missing_as, step

from credit_core import obligations
from credit_core.obligations import Account

from assessment.modes import SCENARIO

# Account types the treatment matrix cannot cost mechanically; the verdict is indeterminate until someone does.
REFER_ACCOUNT_TYPES = frozenset({99})


def _same_account(account: Account) -> tuple:
    return account.get("account_type_code"), account.get("opened_date"), round(account.get("balance") or 0.0, 2)


def _once(accounts: list[Account]) -> list[Account]:
    """An account reported on both applicants' profiles counts once."""
    return list({_same_account(a): a for a in accounts}.values())


def _for_mode(accounts: list[Account], assessment_mode_code: int) -> list[Account]:
    """A scenario assumes accounts with a settlement quote are paid off, so they leave the list; any other mode costs them as usual."""
    if assessment_mode_code == SCENARIO:
        return [a for a in accounts if a.get("settlement_quote") is None]
    return [{**a, "settlement_quote": None} for a in accounts]


@allow_fallback
@step(outputs=(*obligations.Obligations._fields, "has_refer_account"))
def household_obligations(
    assessment_mode_code: int,
    is_joint_application: bool = missing_as(False),
    applicant1_bureau_accounts: list[Account] = missing_as([]), applicant1_internal_accounts: list[Account] = missing_as([]),
    applicant2_bureau_accounts: list[Account] = missing_as([]), applicant2_internal_accounts: list[Account] = missing_as([]),
) -> tuple[float, float, float, int, int, float, float, list[int], list[int], list[float], list[bool], bool]:
    """`credit_core.obligations.assess` over both applicants' lists, plus whether any account needs a manual look.

    Example: a card on both applicants' bureau profiles adds its monthly cost once.
    """
    bureau = applicant1_bureau_accounts + (applicant2_bureau_accounts if is_joint_application else [])
    internal = applicant1_internal_accounts + (applicant2_internal_accounts if is_joint_application else [])
    bureau, internal = (_for_mode(_once(accounts), assessment_mode_code) for accounts in (bureau, internal))
    refer = any(a.get("account_type_code") in REFER_ACCOUNT_TYPES for a in bureau + internal)
    return (*obligations.assess(bureau, internal), refer)
