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
from consolidation.capacity import Buffers, Capacity, case_record, debt_service_ratio
from consolidation.inventory import num
from consolidation.products import Quote
from consolidation.settlement import settlement_buffer


def remaining_cost(account: dict, paydown_horizon_months: int) -> float:
    """What the client still pays on an account: its remaining instalments, or for a revolving account the cost of
    amortising the balance over an assumed horizon (a revolving account has no natural term)."""
    if account.get("remaining_term_months"):
        return num(account, "instalment") * account["remaining_term_months"]
    cost = instalment_before_fees(num(account, "balance"), paydown_horizon_months, num(account, "nominal_annual_rate"))
    return cost * paydown_horizon_months


def remaining_term(account: dict, paydown_horizon_months: int) -> float:
    """Months left on an account; a revolving account is taken to have the paydown horizon."""
    return account.get("remaining_term_months") or float(paydown_horizon_months)


def weighted_rate(accounts: list[dict]) -> float:
    balance = sum(num(a, "balance") for a in accounts)
    return sum(num(a, "balance") * num(a, "nominal_annual_rate") for a in accounts) / balance if balance else 0.0


def ratio(part: float, whole: float) -> float:
    return round(part / whole, 4) if whole else 0.0


@dataclass
class ScenarioEvaluator:
    """Evaluates scenarios of one assessment against what they all share."""

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
        """The scenario as a record: priced, tested, with every intervention and a viability verdict."""
        r, policy = self.request, self.policies[product_code]
        settled = [self.by_ref[ref] for ref in refs]
        amounts = {a["account_ref"]: a["settlement_amount"] for a in settled}
        total = round(sum(amounts.values()), 2)
        buffer = settlement_buffer(total, self.buffer_policy["buffer_pct"], self.buffer_policy["buffer_cap"])
        new_money = round(min(r["requested_amount"], total * policy["new_money_pct"], policy["new_money_cap"]), 2)
        external = ratio(sum(a["settlement_amount"] for a in settled if not a["is_internal"]), total)
        record = {"scenario_id": scenario_id, "settlement_set": list(refs), "generating_rule": rule,
                  "product_code": product_code, "term_months": term_months, "promotional_months": promotional_months,
                  "settlement_total": total, "settlement_buffer": buffer, "new_money": new_money,
                  "external_proportion": external, "rejection_reason_codes": [], "interventions": []}

        quote = self.products[product_code].quote(
            policy, total + buffer + new_money, term_months, promotional_months, r["risk_grade"],
            applicant_age_years=r["applicant_age_years"], external_proportion=external)
        if isinstance(quote, str):
            code = reasons.NOT_PRICED if quote == "no_rate_cell" else reasons.OUTSIDE_PRODUCT_LIMITS
            return {**record, "viable": False, "rejection_reason_codes": [code], "unpriced_reason": quote}

        case = case_record(r["decision_date"], product_code, r["channel_code"], r["segment_code"],
                           r["court_ordered_deductions"], self.invariants, self.buffers(product_code, r["risk_grade"]))
        after = self.capacity(case, self.accounts, amounts, quote.tested_payment, self.stack_enabled)
        facts, position = self._position(settled, refs, term_months, quote, new_money, total, external, after)
        checks = interventions.evaluate(facts, policy, r["decision_date"], r["consolidations_last_24_months"],
                                        self.max_consolidations_24m, self.stack_enabled)
        rejections = [1000 + c["intervention"] for c in checks if c["status"] == interventions.FAILED]
        if after["affordability_verdict_code"] != verdict.PASS:
            rejections.append(reasons.AFFORDABILITY_FAIL)
        return {
            **record, "viable": not rejections, "rejection_reason_codes": sorted(rejections), "interventions": checks,
            "offered_amount": quote.amount, "nominal_annual_rate": quote.nominal_annual_rate, "instalment": quote.instalment,
            "tested_payment": quote.tested_payment, "total_cost_of_credit": quote.total_cost_of_credit,
            "effective_annual_rate": quote.effective_annual_rate, "rate_cell_id": quote.rate_cell_id,
            "rate_card_version": quote.rate_card_version, "fees": quote.fees,
            "existing_obligations_after": after["existing_obligations"], "discretionary_income": after["discretionary_income"],
            "max_affordable_instalment": after["max_affordable_instalment"],
            "affordability_verdict_code": after["affordability_verdict_code"], **position,
        }

    def _position(self, settled: list[dict], refs: tuple[int, ...], term_months: int, quote: Quote, new_money: float,
                  total: float, external: float, after: dict) -> tuple[dict, dict]:
        """`(the facts the interventions test, the scenario's outcome)`: relief, cost, rates, and the measures scored."""
        base, econ = self.baseline, self.economics
        retained = [a for a in self.accounts if a["account_ref"] not in refs]
        settled_instalments = sum(num(a, "instalment") for a in settled)
        relief = round(settled_instalments - quote.instalment, 2)
        settled_remaining = sum(remaining_cost(a, self.horizon) for a in settled)
        # new money is cash the client receives, not a cost of settling these debts
        cost_delta = round(quote.total_cost_of_credit - new_money - settled_remaining, 2)
        retained_balance = sum(num(a, "balance") for a in retained)
        blended_rate = (quote.amount * quote.nominal_annual_rate + retained_balance * weighted_rate(retained)) / (
            quote.amount + retained_balance)
        longest_settled = max(remaining_term(a, self.horizon) for a in settled)
        margin = quote.amount * max(0.0, quote.nominal_annual_rate - econ["cost_of_funds"] - econ["expected_loss_rate"])
        forgone = sum(num(a, "balance") * max(0.0, num(a, "nominal_annual_rate") - econ["cost_of_funds"])
                      for a in settled if a["is_internal"])
        providers_left = len({a.get("provider_code") for a in retained})
        facts = {
            "accounts": len(refs), "relief_pct": ratio(relief, settled_instalments),
            "cost_increase_pct": ratio(cost_delta, settled_remaining),
            "term_extension_months": max(0.0, term_months - longest_settled), "new_rate": quote.nominal_annual_rate,
            "settled_rate": round(weighted_rate(settled), 4), "new_money": new_money, "settlement_total": total,
            "external_proportion": external, "discretionary_after": after["discretionary_income_after"],
            "dsr": debt_service_ratio(after["existing_obligations"] + quote.tested_payment,
                                      self.invariants["net_monthly_income"]),
        }
        outcome = {
            "instalment_relief": relief, "total_cost_delta": cost_delta,
            "settled_longest_remaining_term_months": float(longest_settled),
            "bank_expected_value": round(margin - forgone, 2),
            "after": {
                "monthly_instalment": round(base["total_instalment"] - relief, 2),
                "weighted_average_rate": round(blended_rate, 4),
                "longest_remaining_term_months": float(max([remaining_term(a, self.horizon) for a in retained] + [term_months])),
                "total_cost": round(base["total_remaining_cost"] + cost_delta, 2), "accounts": base["account_count"] - len(refs),
                "providers": providers_left},
            "measures": {
                "new_money_ratio": ratio(new_money, self.request["requested_amount"]),
                "instalment_relief_ratio": ratio(relief, base["total_instalment"]),
                "total_cost_increase_ratio": ratio(cost_delta, base["total_remaining_cost"]),
                "bank_value_ratio": ratio(margin - forgone, total),
                "accounts_exited_proportion": ratio(len(refs), base["account_count"]),
                "weighted_rate_reduction": round(base["weighted_average_rate"] - blended_rate, 4),
                "providers_exited_proportion": ratio(base["provider_count"] - providers_left, base["provider_count"])},
        }
        return facts, outcome
