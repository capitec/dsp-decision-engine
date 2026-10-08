"""Product 20, Everyday Card balance transfer: a revolving limit with a promotional rate that reverts.

It has no term and no instalment, so the quote is built from a limit, a promotional rate for a fixed period, a
reversion rate, and a mandatory paydown. Affordability is tested against a stressed payment, never the promotional one.
"""
from __future__ import annotations

import bisect
import math

from credit_core.rate_card import rate_card_rate
from credit_core.rounding import round_instalment
from decider import DecisionTableConfig, ParamTable, allow_fallback, param, param_table, step

from consolidation import vocab
from consolidation.products import Quote
from consolidation.vocab import PRODUCT_EVERYDAY_CARD

CODE = PRODUCT_EVERYDAY_CARD
REJECT_LIMIT, REJECT_UNPRICED = "transfer_above_utilisation_cap", "no_rate_cell"
PROMOTIONAL_MONTHS = param_table({"promotional_months": int}, default=[{"promotional_months": m} for m in (0, 6, 12, 18, 24)])


@allow_fallback
@step(output="everyday_card_policy")
def everyday_card_policy(
    max_accounts: int = param(8, ge=3, le=15), min_limit: float = param(1_000.0, ge=0.0),
    max_limit: float = param(300_000.0, ge=0.0), max_utilisation: float = param(0.80, gt=0.0, le=1.0),
    transfer_fee_pct: float = param(0.025, ge=0.0, le=0.10), minimum_payment_floor: float = param(50.0, ge=0.0),
    minimum_payment_pct: float = param(0.03, ge=0.0, le=0.2), paydown_months: int = param(36, ge=1, le=84),
    stress_months: int = param(36, ge=1, le=84), min_relief: float = param(0.10, ge=0.0, le=0.30),
    anti_harm: float = param(0.15, ge=0.0, le=0.60), rate_ceiling: bool = param(True),
    dsr_ceiling: float = param(0.45, ge=0.30, le=0.60), min_discretionary: float = param(850.0, ge=500.0, le=2_500.0),
    max_revolving_in_arrears: int = param(1, ge=0), transfer_cooldown_months: int = param(12, ge=0),
    candidate_promotional_months: ParamTable = PROMOTIONAL_MONTHS,
) -> dict:
    """The limits and intervention thresholds product 20's team owns, as one record for the search."""
    return {
        "product_code": CODE, "min_accounts": 1, "max_accounts": max_accounts, "min_limit": min_limit,
        "max_limit": max_limit, "max_utilisation": max_utilisation, "transfer_fee_pct": transfer_fee_pct,
        "minimum_payment_floor": minimum_payment_floor, "minimum_payment_pct": minimum_payment_pct,
        "paydown_months": paydown_months, "stress_months": stress_months, "min_relief": min_relief,
        "anti_harm": anti_harm, "rate_ceiling": rate_ceiling, "new_money_pct": 0.0, "new_money_cap": 0.0,
        "dsr_ceiling": dsr_ceiling, "min_discretionary": min_discretionary,
        "max_revolving_in_arrears": max_revolving_in_arrears, "transfer_cooldown_months": transfer_cooldown_months,
        "candidate_promotional_months": candidate_promotional_months.promotional_months.tolist(),
    }


class CardRates:
    """The two small rate documents as sorted limit bands: `promotional(limit, months, grade)` and `reversion(limit, grade)`
    each give `(rate as a fraction, cell id)` or None."""

    def __init__(self, promotional: DecisionTableConfig, reversion: DecisionTableConfig):
        self.version = reversion.rows.df["cell_id"][0].split("@")[1].split("#")[0]
        self._promotional = self._index(promotional, ("duration", "grade"))
        self._reversion = self._index(reversion, ("grade",))

    @staticmethod
    def _index(table: DecisionTableConfig, keys: tuple[str, ...]) -> dict:
        groups: dict[tuple, list[dict]] = {}
        for row in table.rows.df.to_dicts():
            groups.setdefault(tuple(row[k] for k in keys), []).append(row)
        return {key: ([r["lim_lo"] for r in sorted(rows, key=lambda r: r["lim_lo"])], sorted(rows, key=lambda r: r["lim_lo"]))
                for key, rows in groups.items()}

    @staticmethod
    def _hit(index: dict, key: tuple, limit: float):
        if key not in index:
            return None
        starts, rows = index[key]
        row = rows[bisect.bisect_right(starts, limit) - 1]
        return rate_card_rate(row["rate"]), row["cell_id"]

    def promotional(self, limit: float, months: int, grade: int):
        return self._hit(self._promotional, (months, grade), limit)

    def reversion(self, limit: float, grade: int):
        return self._hit(self._reversion, (grade,), limit)


def paydown(opening: float, promotional_rate: float, promotional_months: int, reversion_rate: float, policy: dict) -> tuple[float, float]:
    """`(total paid, first payment)` over the mandatory paydown: the minimum payment or 1/n of the opening balance,
    whichever is larger, at the promotional rate and then at the reversion rate."""
    n = policy["paydown_months"]
    balance, total, first = opening, 0.0, None
    for month in range(1, n * 3 + 1):  # the minimum payment alone always clears the balance within this bound
        if balance <= 0.01:
            break
        monthly = (promotional_rate if month <= promotional_months else reversion_rate) / 12.0
        interest = balance * monthly
        minimum = max(policy["minimum_payment_floor"], balance * policy["minimum_payment_pct"] + interest)
        payment = min(round_instalment(max(minimum, opening / n)), balance + interest)
        first = payment if first is None else first
        balance += interest - payment
        total += payment
    return round(total, 2), round(first or 0.0, 2)


class EverydayCard:
    code = CODE

    def __init__(self, rates: CardRates):
        self.rates = rates

    def bind(self, pricing_context: dict) -> EverydayCard:
        return self

    def unavailable(self, client: dict, policy: dict) -> str | None:
        """Why this client cannot use the card at all, or None."""
        if client["revolving_in_arrears"] > policy["max_revolving_in_arrears"]:
            return "more than one revolving account in arrears"
        since = client.get("last_balance_transfer_date")
        if since is not None:
            months = (client["decision_date"].year - since.year) * 12 + client["decision_date"].month - since.month
            if months < policy["transfer_cooldown_months"]:
                return "a balance transfer within the last 12 months"
        return None

    def carries(self, accounts: list[dict], policy: dict) -> bool:
        return bool(accounts) and all(a["account_type_code"] in vocab.CARD_TYPES for a in accounts)

    def variants(self, policy: dict, longest_remaining_term: float) -> list[tuple[int, int]]:
        return [(policy["paydown_months"], m) for m in policy["candidate_promotional_months"]]

    def quote(self, policy: dict, proceeds: float, term_months: int, promotional_months: int, risk_grade: int, **_) -> Quote | str:
        """A limit big enough that the transfer is at most `max_utilisation` of it, priced for `promotional_months`."""
        limit = max(policy["min_limit"], min(policy["max_limit"], math.ceil(proceeds / policy["max_utilisation"] / 100.0) * 100.0))
        if proceeds > policy["max_utilisation"] * limit:
            return REJECT_LIMIT
        promotional, reversion = self.rates.promotional(limit, promotional_months, risk_grade), self.rates.reversion(limit, risk_grade)
        if promotional is None or reversion is None:
            return REJECT_UNPRICED
        fee = round(proceeds * policy["transfer_fee_pct"], 2)
        total, first = paydown(proceeds + fee, promotional[0], promotional_months, reversion[0], policy)
        monthly = reversion[0] / 12.0
        n = policy["stress_months"]
        stressed = limit * monthly / (1.0 - (1.0 + monthly) ** -n) if monthly else limit / n
        return Quote(limit, first, round_instalment(stressed), promotional[0], total, round((1.0 + reversion[0] / 12.0) ** 12 - 1.0, 4),
                     promotional[1], self.rates.version,
                     {"transfer_fee": fee, "reversion_rate": reversion[0], "reversion_cell_id": reversion[1],
                      "promotional_months": promotional_months})
