"""What it costs to close each settleable account, and until when that figure is true.

The settlement amount is not the balance. It is the capital at the assumed settlement date, the interest accrued to
that date, unpaid fees, any early settlement charge and security release cost, less the rebate of unearned fees. A
quotation in hand overrides the derivation: it is the provider's binding figure.
"""
from __future__ import annotations

from datetime import date, timedelta

from decikit import ParamTable, allow_fallback, param, param_table, step

from consolidation import vocab
from consolidation.inventory import num
from consolidation.records import Settlement, SettlementBufferPolicy

# early_settlement_rule_code -> months of interest charged at or above the large-agreement threshold
EARLY_SETTLEMENT_RULES = param_table({"early_settlement_rule_code": int, "months_of_interest": float}, default=[
    {"early_settlement_rule_code": 0, "months_of_interest": 0.0},
    {"early_settlement_rule_code": 1, "months_of_interest": 1.0},
    {"early_settlement_rule_code": 2, "months_of_interest": 3.0},
])


def settlement_buffer(settlement_total: float, buffer_pct: float, buffer_cap: float) -> float:
    """The cushion added to an advance for interest that accrues before the money arrives: a share of the total, capped.

    Example: `settlement_buffer(100_000, 0.015, 2_500)` is 1 500.0, and `settlement_buffer(400_000, 0.015, 2_500)` is 2 500.0.
    """
    return round(min(settlement_total * buffer_pct, buffer_cap), 2)


def derive_settlement(account: dict, decision_date: date, disbursement_lag_days: int, large_agreement_threshold: float,
                      security_release_cost: float, estimate_tolerance_pct: float, months_of_interest: dict[int, float]) -> Settlement:
    """One account's settlement amount with every component, its basis and the date the figure is for."""
    # the rules table has one turnaround per provider, used as both the quotation and the payment turnaround
    assumed = decision_date + timedelta(days=int(num(account, "quotation_turnaround_days")) + disbursement_lag_days)
    balance, rate = num(account, "balance"), num(account, "nominal_annual_rate")
    days = int(num(account, "days_since_statement")) + (assumed - decision_date).days
    out = {"account_ref": account["account_ref"], "assumed_settlement_date": assumed.isoformat(),
           "quotation_reference": account.get("quotation_reference"),
           "quotation_expiry_date": account["quotation_expiry_date"].isoformat() if account.get("quotation_held") else None}
    if account.get("quotation_held"):
        return {**out, "settlement_amount": round(num(account, "quotation_amount"), 2), "amount_basis": vocab.QUOTED_BASIS,
                "estimation_tolerance": 0.0, "components": {"quoted_amount": round(num(account, "quotation_amount"), 2)}}
    charge_months = months_of_interest.get(int(num(account, "early_settlement_rule_code")), 0.0)
    components = {
        "capital": balance,
        "accrued_interest": round(balance * rate / 365.0 * days, 2),
        "fees_unpaid": num(account, "fees_unpaid"),
        "early_settlement_charge": (round(balance * rate / 12.0 * charge_months, 2)
                                    if balance >= large_agreement_threshold else 0.0),
        "security_release_cost": security_release_cost if account["settleability_code"] == vocab.SECURITY_RELEASE else 0.0,
        "rebate": 0.0 - num(account, "unearned_rebate"),
    }
    amount = round(sum(components.values()), 2)
    internal = bool(account["is_internal"])
    return {**out, "settlement_amount": amount, "components": components,
            "amount_basis": vocab.DERIVED_BASIS if internal else vocab.ESTIMATED_BASIS,
            "estimation_tolerance": 0.0 if internal else round(amount * estimate_tolerance_pct, 2)}


@allow_fallback
@step(outputs=("settlements", "settlement_buffer_policy"))
def settlements(
    classified_accounts: list[dict], decision_date: date,
    disbursement_lag_days: int = param(2, ge=0, le=12), large_agreement_threshold: float = param(250_000.0, ge=0.0),
    security_release_cost: float = param(500.0, ge=0.0), estimate_tolerance_pct: float = param(0.03, ge=0.0, le=0.5),
    buffer_pct: float = param(0.015, ge=0.0, le=0.1), buffer_cap: float = param(2_500.0, ge=0.0),
    early_settlement_rules: ParamTable = EARLY_SETTLEMENT_RULES,
) -> tuple[list[Settlement], SettlementBufferPolicy]:
    """The settlement amount of every account that can be settled, and the buffer policy scenarios will apply.

    A client exclusion is applied before this step: an excluded account is not costed.
    """
    months = dict(zip(early_settlement_rules.early_settlement_rule_code.tolist(),
                      early_settlement_rules.months_of_interest.tolist()))
    costed = [derive_settlement(a, decision_date, disbursement_lag_days, large_agreement_threshold, security_release_cost,
                                estimate_tolerance_pct, months)
              for a in classified_accounts if a["settleability_code"] in vocab.SETTLEABLE and not a["is_excluded"]]
    return costed, {"buffer_pct": buffer_pct, "buffer_cap": buffer_cap}
