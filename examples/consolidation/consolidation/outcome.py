"""The decision, the conditions on it, and what the client and the back office are shown.

The eligibility route takes precedence over whatever the search found: a client under debt review is referred, never
declined, however good a scenario looks.
"""
from __future__ import annotations

from datetime import date, timedelta

from credit_core.vocab import (
    OUTCOME_APPROVE,
    OUTCOME_APPROVE_WITH_CONDITIONS,
    OUTCOME_DECLINE,
    OUTCOME_REFER,
)
from decikit import allow_fallback, missing_as, param, step

from consolidation import eligibility, reasons, vocab
from consolidation.inventory import num

_GATE_REASONS = (("CON-ELIG-02", reasons.ADMINISTRATION), ("CON-ELIG-03", reasons.SERIAL_CONSOLIDATION),
                 ("CON-ELIG-05", reasons.NO_INCOME), ("CON-ELIG-06", reasons.NO_BUREAU_VIEW),
                 ("CON-ELIG-07", reasons.RECKLESS_ALLEGATION))


def decide(route: str, fired: dict, nomination_conflicts: list[int], evidence_sufficiency_code: int, plain: bool,
           plain_offer: dict, requested_amount: float, chosen: dict, estimated: bool) -> tuple[int, list[int]]:
    """`(outcome_code, reason codes)`. `plain` says the flow completes as plain granting; `estimated` that the chosen
    scenario rests on a settlement amount that is an estimate, so the approval is conditional."""
    if route == eligibility.DEBT_REVIEW_PATH:
        return OUTCOME_REFER, [reasons.DEBT_REVIEW]
    if route in (eligibility.DECLINE, eligibility.REFER):
        codes = [code for gate, code in _GATE_REASONS if fired[gate]]
        codes += [reasons.NOMINATION_CONFLICT] * bool(nomination_conflicts)
        return (OUTCOME_DECLINE if route == eligibility.DECLINE else OUTCOME_REFER), codes
    if evidence_sufficiency_code:
        return OUTCOME_REFER, [reasons.EVIDENCE_INSUFFICIENT]
    if plain:
        if not plain_offer:
            return OUTCOME_DECLINE, [reasons.NOT_AFFORDABLE_AS_REQUESTED]
        return (OUTCOME_APPROVE if plain_offer["offered_amount"] == requested_amount else OUTCOME_APPROVE_WITH_CONDITIONS), []
    if not chosen:
        return OUTCOME_DECLINE, [reasons.NO_VIABLE_SCENARIO]
    return (OUTCOME_APPROVE_WITH_CONDITIONS if estimated else OUTCOME_APPROVE), []


def comparison(baseline: dict, longest_term: float, chosen: dict) -> dict:
    """Before and after, side by side. Where the total cost rises, it says so in rands and percent, with the reason."""
    after = chosen["after"]
    change = round(after["total_cost"] - baseline["total_remaining_cost"], 2)
    out = {
        "before": {"monthly_instalment": baseline["total_instalment"], "weighted_average_rate": baseline["weighted_average_rate"],
                   "longest_remaining_term_months": longest_term, "total_cost": baseline["total_remaining_cost"],
                   "accounts": baseline["account_count"], "providers": baseline["provider_count"]},
        "after": after, "total_cost_change": change,
        "total_cost_change_pct": round(change / baseline["total_remaining_cost"], 4) if baseline["total_remaining_cost"] else 0.0,
    }
    if change > 0:
        out["total_cost_note"] = (
            f"you will pay R{change:,.0f} more in total ({out['total_cost_change_pct']:.1%}) because the new facility "
            f"runs {chosen['term_months']} months against {chosen['settled_longest_remaining_term_months']:.0f} "
            "on the accounts it settles")
    return out


def conditions_of(settled: list[dict], accounts: dict[int, dict]) -> list[str]:
    """What must happen before the offer can be executed: quotations to obtain, cards to close, security to release."""
    conditions = [f"obtain a settlement quotation for account {s['account_ref']}" for s in settled
                  if s["amount_basis"] == vocab.ESTIMATED_BASIS]
    conditions += [f"close account {s['account_ref']} once settled" for s in settled
                   if accounts[s["account_ref"]]["account_type_code"] in vocab.REVOLVING_TYPES]
    conditions += [f"release the security on account {s['account_ref']} "
                   f"({num(accounts[s['account_ref']], 'security_release_days'):.0f} working days)" for s in settled
                   if accounts[s["account_ref"]]["settleability_code"] == vocab.SECURITY_RELEASE]
    return conditions


@allow_fallback
@step(outputs=("outcome_code", "decline_reason_codes", "chosen_offer", "conditions", "offer_valid_until", "before_and_after",
               "execution_package"))
def outcome(
    eligibility_route: str, eligibility_gate_verdicts: dict, nomination_conflicts: list[int], baseline: dict,
    short_circuit_applies: bool, plain_offer: dict, chosen_scenario: dict, classified_accounts: list[dict],
    settlements: list[dict], decision_date: date, requested_amount: float = missing_as(0.0),
    offer_validity_days: int = param(21, ge=1, le=90),
) -> tuple[int, list[int], dict, list[str], str | None, dict, list[dict]]:
    """The outcome code, the reasons that fired, and for an approval the offer in full.

    The reasons are ranked, and the primary one chosen, by the `client_reason_codes` step that follows.

    `chosen_offer` is the winning scenario with the cost of each account it settles. `offer_valid_until` is the
    earlier of the offer validity and the first quotation to expire. `before_and_after` is the comparison the client is
    shown, and `execution_package` the per-account settlement instructions for the back office.
    """
    plain = eligibility_route == eligibility.NOT_APPLICABLE or short_circuit_applies
    by_ref = {s["account_ref"]: s for s in settlements}
    chosen = {**chosen_scenario, "settlement_details": [by_ref[ref] for ref in chosen_scenario.get("settlement_set", [])]}
    estimated = any(s["amount_basis"] == vocab.ESTIMATED_BASIS for s in chosen["settlement_details"])
    code, fired = decide(eligibility_route, eligibility_gate_verdicts, nomination_conflicts,
                         baseline["evidence_sufficiency_code"], plain, plain_offer, requested_amount, chosen_scenario, estimated)
    approved = code in (OUTCOME_APPROVE, OUTCOME_APPROVE_WITH_CONDITIONS)
    if not (approved and chosen_scenario and not plain):
        offer = plain_offer if approved and plain else {}
        return code, fired, offer, [], None, {}, []
    accounts = {a["account_ref"]: a for a in classified_accounts}
    expiries = [date.fromisoformat(s["quotation_expiry_date"]) for s in chosen["settlement_details"]
                if s["quotation_expiry_date"]]
    valid_until = min(expiries + [decision_date + timedelta(days=offer_validity_days)])
    longest = max([num(a, "remaining_term_months") for a in classified_accounts] + [0.0])
    return (code, fired, chosen, conditions_of(chosen["settlement_details"], accounts), valid_until.isoformat(),
            comparison(baseline, longest, chosen), chosen["settlement_details"])
