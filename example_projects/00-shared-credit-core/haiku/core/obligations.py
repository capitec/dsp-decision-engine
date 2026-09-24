from __future__ import annotations
"""Existing debt obligations (00 §6.4, 00-ADDENDUM A10)."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Account:
    """One existing credit account."""
    account_id: str
    account_type: str  # "instalment", "revolving", "mortgage"
    monthly_payment: float
    outstanding_balance: float
    status: str  # "active", "delinquent", "settled"


@dataclass
class ObligationsResult:
    """Result of obligations calculation."""
    existing_obligations: float  # Total monthly cost
    obligations_scalar: float  # Same as existing_obligations
    per_account_obligations: list[dict]  # Detail per account
    account_count: int
    deduped_count: int


def calculate_obligations(
    bureau_accounts: Optional[list[Account]] = None,
    internal_accounts: Optional[list[Account]] = None
) -> ObligationsResult:
    """
    Convert a variable-length list of existing credit accounts into monthly cost.
    Supports two account lists (bureau and internal) with de-duplication.
    Implements 09 §5.15 item 8: mutable state captured.
    """
    bureau_accounts = bureau_accounts or []
    internal_accounts = internal_accounts or []

    # De-duplicate: if an account appears in both lists, keep internal version
    internal_ids = {a.account_id for a in internal_accounts}
    deduped_bureau = [a for a in bureau_accounts if a.account_id not in internal_ids]

    # Combine
    all_accounts = internal_accounts + deduped_bureau

    # Calculate total obligations (only active accounts)
    total_obligations = 0.0
    per_account = []

    for account in all_accounts:
        if account.status in ("active", "delinquent"):
            monthly = account.monthly_payment
            total_obligations += monthly
            per_account.append({
                "account_id": account.account_id,
                "type": account.account_type,
                "monthly_payment": monthly,
                "outstanding": account.outstanding_balance,
                "status": account.status
            })

    return ObligationsResult(
        existing_obligations=total_obligations,
        obligations_scalar=total_obligations,  # Per-account annotation
        per_account_obligations=per_account,  # Implements 00 §6.4 requirement
        account_count=len(all_accounts),
        deduped_count=len(deduped_bureau)
    )
