"""P14(a) settleability — 14 decision points. Spec 5.15(a). Per account: is
it settleable, by whom, at what cost, within what window.

Written for P14 and consumed by P15's decrease path (spec Q1 reuse — the
seam this project tests). Reads the settleability matrix T4 owns and T12
does not (OWNERS.toml `depends_on`: "T12 owns the phase and owns almost
nothing the phase depends on").
"""

from __future__ import annotations

from decider2 import module, Table

class SettleabilityMatrixTable(Table):
    key: tuple[int, str]        # (provider_id, attribute)  — 240 x 6
    value: float

def settleable(per_account_obligation_annotation: list[dict],
               settleability_matrix: SettleabilityMatrixTable) -> list[bool]:
    pass  # per account: quotation availability from the 240-provider matrix

def early_settlement_charge(per_account_obligation_annotation: list[dict],
                            settleability_matrix: SettleabilityMatrixTable) -> list[float]:
    pass  # per settleable account, from the matrix

def notice_period_days(per_account_obligation_annotation: list[dict],
                       settleability_matrix: SettleabilityMatrixTable) -> list[int]:
    pass  # per settleable account

def post_settlement_state(per_account_obligation_annotation: list[dict],
                          settleability_matrix: SettleabilityMatrixTable) -> list[dict]:
    """Whether the facility closes, and whether the LIMIT survives. A settled
    card with its limit intact is a re-accumulation waiting to happen, so this
    — not "settled: true/false" — is what obligation re-derivation must see."""
    pass  # per account: {"closes": bool, "limit_survives": bool}

Classify = module(settleable, early_settlement_charge, notice_period_days,
                  post_settlement_state, name="settleability")
