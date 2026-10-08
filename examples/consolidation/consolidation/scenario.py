"""One scenario, evaluated: obligations re-derived, the instalment priced and tested, the interventions applied.

A scenario is a settlement set carried by one product over one term. Everything that does not depend on the set (the
income and expense determination, the baseline) is read, never recomputed; only the obligations and what follows from
them are.
"""
from __future__ import annotations

from dataclasses import dataclass

from assessment import verdict
from credit_core.instalment import instalment_before_fees

from consolidation import interventions, reasons
from consolidation.capacity import Buffers, Capacity, case_record
from consolidation.inventory import num
from consolidation.settlement import settlement_buffer

SCENARIO_MEASURES = ("new_money_ratio", "instalment_relief_ratio", "total_cost_increase_ratio", "bank_value_ratio",
                     "accounts_exited_proportion", "weighted_rate_reduction", "providers_exited_proportion")


def remaining_cost(account: dict, paydown_horizon_months: int) -> float:
    """What the client still pays on an account: its remaining instalments, or for a revolving account the cost of
    amortising the balance over an assumed horizon (a revolving account has no natural term)."""
    if account.get("remaining_term_months"):
        return num(account, "instalment") * account["remaining_term_months"]
    return instalment_before_fees(num(account, "balance"), paydown_horizon_months, num(account, "nominal_annual_rate")) * paydown_horizon_months


def remaining_term(account: dict, paydown_horizon_months: int) -> float:
    """Months left on an account; a revolving account is taken to have the paydown horizon."""
    return account.get("remaining_term_months") or float(paydown_horizon_months)


def weighted_rate(accounts: list[dict]) -> float:
    balance = sum(num(a, "balance") for a in accounts)
    return sum(num(a, "balance") * num(a, "nominal_annual_rate") for a in accounts) / balance if balance else 0.0


@dataclass
class Assessment:
    """What every scenario of one assessment shares."""

    request: dict  # decision_date, channel_code, segment_code, risk_grade, applicant_age_years, requested_amount, ...
    accounts: list[dict]  # every account, with its settleability and settlement facts
    invariants: dict
    baseline: dict
    policies: dict[int, dict]
    products: dict
    capacity: Capacity
    buffers: Buffers
    buffer_policy: dict
    economics: dict  # cost_of_funds, expected_loss_rate
    max_consolidations_24m: int
    stack_enabled: bool

    def __post_init__(self):
        self.by_ref = {a["account_ref"]: a for a in self.accounts}
        self.horizon = self.baseline["paydown_horizon_months"]

    def evaluate(self, scenario_id: int, refs: tuple[int, ...], rule: str, product_code: int, term_months: int,
                 promotional_months: int) -> dict:
        r, policy, product = self.request, self.policies[product_code], self.products[product_code]
        settled = [self.by_ref[ref] for ref in refs]
        amounts = {a["account_ref"]: a["settlement_amount"] for a in settled}
        total = round(sum(amounts.values()), 2)
        buffer = settlement_buffer(total, self.buffer_policy["buffer_pct"], self.buffer_policy["buffer_cap"])
        new_money = round(min(r["requested_amount"] or 0.0, total * policy["new_money_pct"], policy["new_money_cap"]), 2)
        external = round(sum(a["settlement_amount"] for a in settled if not a["is_internal"]) / total, 4) if total else 0.0
        scenario = {"scenario_id": scenario_id, "settlement_set": list(refs), "generating_rule": rule,
                    "product_code": product_code, "term_months": term_months, "promotional_months": promotional_months,
                    "settlement_total": total, "settlement_buffer": buffer, "new_money": new_money,
                    "external_proportion": external, "rejection_reason_codes": [], "interventions": []}

        quote = product.quote(policy, total + buffer + new_money, term_months, promotional_months, r["risk_grade"],
                              applicant_age_years=r["applicant_age_years"], external_proportion=external)
        if isinstance(quote, str):
            scenario["unpriced_reason"] = quote
            code = reasons.NOT_PRICED if quote == "no_rate_cell" else reasons.OUTSIDE_PRODUCT_LIMITS
            return {**scenario, "viable": False, "rejection_reason_codes": [code]}

        case = case_record(r["decision_date"], product_code, r["channel_code"], r["segment_code"], r["court_ordered_deductions"],
                           self.invariants, self.buffers(product_code, r["risk_grade"]))
        after = self.capacity(case, self.accounts, amounts, quote.tested_payment, self.stack_enabled)
        base = self.baseline
        retained = [a for a in self.accounts if a["account_ref"] not in amounts]
        settled_instalments = sum(num(a, "instalment") for a in settled)
        relief = round(settled_instalments - quote.instalment, 2)
        commitment_after = base["total_instalment"] - relief
        settled_remaining = sum(remaining_cost(a, self.horizon) for a in settled)
        cost_delta = round(quote.total_cost_of_credit - settled_remaining, 2)
        retained_balance = sum(num(a, "balance") for a in retained)
        new_rate_blend = (quote.amount * quote.nominal_annual_rate + retained_balance * weighted_rate(retained)) / (quote.amount + retained_balance)
        facts = {
            "accounts": len(refs), "relief_pct": round(relief / settled_instalments, 4) if settled_instalments else 0.0,
            "cost_increase_pct": round(cost_delta / settled_remaining, 4) if settled_remaining else 0.0,
            "term_extension_months": max(0.0, term_months - max(remaining_term(a, self.horizon) for a in settled)),
            "new_rate": quote.nominal_annual_rate, "settled_rate": round(weighted_rate(settled), 4), "new_money": new_money,
            "settlement_total": total, "external_proportion": external,
            "dsr": round((after["existing_obligations"] + quote.tested_payment) / self.invariants["net_monthly_income"], 4),
            "discretionary_after": after["discretionary_income_after"],
        }
        checks = interventions.evaluate(facts, policy, r["decision_date"], r["consolidations_last_24_months"],
                                        self.max_consolidations_24m, self.stack_enabled)
        rejections = [1000 + c["intervention"] for c in checks if c["status"] == interventions.FAILED]
        if after["affordability_verdict_code"] != verdict.PASS:
            rejections.append(reasons.AFFORDABILITY_FAIL)

        econ = self.economics
        bank_value = quote.amount * max(0.0, quote.nominal_annual_rate - econ["cost_of_funds"] - econ["expected_loss_rate"]) - sum(
            num(a, "balance") * max(0.0, num(a, "nominal_annual_rate") - econ["cost_of_funds"]) for a in settled if a["is_internal"])
        providers_left = {a.get("provider_code") for a in retained}
        after_position = {
            "monthly_instalment": round(commitment_after, 2), "weighted_average_rate": round(new_rate_blend, 4),
            "longest_remaining_term_months": float(max([remaining_term(a, self.horizon) for a in retained] + [term_months])),
            "total_cost": round(base["total_remaining_cost"] + cost_delta, 2), "accounts": base["account_count"] - len(refs),
            "providers": len(providers_left)}
        measures = {
            "new_money_ratio": round(new_money / r["requested_amount"], 4) if r["requested_amount"] else 0.0,
            "instalment_relief_ratio": round(relief / base["total_instalment"], 4) if base["total_instalment"] else 0.0,
            "total_cost_increase_ratio": round(cost_delta / base["total_remaining_cost"], 4) if base["total_remaining_cost"] else 0.0,
            "bank_value_ratio": round(bank_value / total, 4) if total else 0.0,
            "accounts_exited_proportion": round(len(refs) / base["account_count"], 4),
            "weighted_rate_reduction": round(base["weighted_average_rate"] - new_rate_blend, 4),
            "providers_exited_proportion": round((base["provider_count"] - len(providers_left)) / base["provider_count"], 4),
        }
        return {
            **scenario, "viable": not rejections, "rejection_reason_codes": sorted(set(rejections)), "interventions": checks,
            "offered_amount": quote.amount, "nominal_annual_rate": quote.nominal_annual_rate, "instalment": quote.instalment,
            "tested_payment": quote.tested_payment, "total_cost_of_credit": quote.total_cost_of_credit,
            "effective_annual_rate": quote.effective_annual_rate, "rate_cell_id": quote.rate_cell_id,
            "rate_card_version": quote.rate_card_version, "fees": quote.fees,
            "existing_obligations_after": after["existing_obligations"], "discretionary_income": after["discretionary_income"],
            "max_affordable_instalment": after["max_affordable_instalment"],
            "affordability_verdict_code": after["affordability_verdict_code"],
            "instalment_relief": relief, "total_cost_delta": cost_delta, "bank_expected_value": round(bank_value, 2),
            "measures": measures, "after": after_position,
        }
