"""The decision, the conditions on it, and what the client and the back office are shown.

The eligibility route takes precedence over whatever the search found: a client under debt review is referred, never
declined, however good a scenario looks.
"""
from __future__ import annotations

from datetime import date, timedelta

from credit_core.vocab import OUTCOME_APPROVE, OUTCOME_APPROVE_WITH_CONDITIONS, OUTCOME_DECLINE, OUTCOME_REFER
from decider import allow_fallback, missing_as, param, step

from consolidation import eligibility, reasons, vocab
from consolidation.inventory import num

_GATE_REASONS = {"CON-ELIG-02": reasons.ADMINISTRATION, "CON-ELIG-03": reasons.SERIAL_CONSOLIDATION, "CON-ELIG-05": reasons.NO_INCOME,
                 "CON-ELIG-06": reasons.NO_BUREAU_VIEW, "CON-ELIG-07": reasons.RECKLESS_ALLEGATION}


def decide(route: str, fired: dict, nomination_conflicts: list[int], evidence_sufficiency_code: int, uses_plain_offer: bool,
           plain_offer: dict, requested_amount: float, chosen: dict) -> tuple[int, list[int]]:
    """`(outcome_code, reason codes)`."""
    if route == eligibility.DEBT_REVIEW_PATH:
        return OUTCOME_REFER, [reasons.DEBT_REVIEW]
    codes = [code for gate, code in _GATE_REASONS.items() if fired[gate]] + ([reasons.NOMINATION_CONFLICT] * bool(nomination_conflicts))
    if route in (eligibility.DECLINE, eligibility.REFER):
        return (OUTCOME_DECLINE if route == eligibility.DECLINE else OUTCOME_REFER), codes
    if evidence_sufficiency_code:
        return OUTCOME_REFER, [reasons.EVIDENCE_INSUFFICIENT]
    if uses_plain_offer:
        if not plain_offer:
            return OUTCOME_DECLINE, [reasons.NOT_AFFORDABLE_AS_REQUESTED]
        return (OUTCOME_APPROVE if plain_offer["offered_amount"] == requested_amount else OUTCOME_APPROVE_WITH_CONDITIONS), []
    if not chosen:
        return OUTCOME_DECLINE, [reasons.NO_VIABLE_SCENARIO]
    return (OUTCOME_APPROVE_WITH_CONDITIONS if any(
        s["amount_basis"] == vocab.ESTIMATED_BASIS for s in chosen["settlement_details"]) else OUTCOME_APPROVE), []


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
        out["total_cost_note"] = (f"you will pay R{change:,.0f} more in total ({out['total_cost_change_pct']:.1%}) because the new "
                                  f"facility runs {chosen['term_months']} months against {chosen['settled_longest_remaining_term_months']:.0f} "
                                  f"on the accounts it settles")
    return out


@allow_fallback
@step(outputs=("outcome_code", "decline_reason_codes", "primary_reason_code", "reason_registry_version", "chosen_offer",
               "conditions", "offer_valid_until", "before_and_after", "execution_package"))
def outcome(
    eligibility_route: str, eligibility_gate_verdicts: dict, nomination_conflicts: list[int], baseline: dict,
    short_circuit_applies: bool, plain_offer: dict, chosen_scenario: dict, classified_accounts: list[dict], settlements: list[dict],
    decision_date: date, requested_amount: float = missing_as(0.0), offer_validity_days: int = param(21, ge=1, le=90),
) -> tuple[int, list[int], int | None, str, dict, list[str], str | None, dict, list[dict]]:
    """The outcome code, its reasons ranked most severe first, and for an approval the offer in full."""
    uses_plain = eligibility_route == eligibility.NOT_APPLICABLE or short_circuit_applies
    by_ref = {s["account_ref"]: s for s in settlements}
    chosen = dict(chosen_scenario)
    if chosen:
        chosen["settlement_details"] = [by_ref[ref] for ref in chosen["settlement_set"]]
    code, fired = decide(eligibility_route, eligibility_gate_verdicts, nomination_conflicts, baseline["evidence_sufficiency_code"],
                         uses_plain, plain_offer, requested_amount, chosen)
    ranked, primary = reasons.CLIENT_REASONS.rank(fired)
    if not chosen or uses_plain or code in (OUTCOME_DECLINE, OUTCOME_REFER):
        offer = plain_offer if uses_plain and code in (OUTCOME_APPROVE, OUTCOME_APPROVE_WITH_CONDITIONS) else {}
        return code, ranked, primary, reasons.CLIENT_REASONS.version, offer, [], None, {}, []
    accounts = {a["account_ref"]: a for a in classified_accounts}
    conditions = [f"obtain a settlement quotation for account {s['account_ref']}" for s in chosen["settlement_details"]
                  if s["amount_basis"] == vocab.ESTIMATED_BASIS]
    conditions += [f"close account {s['account_ref']} once settled" for s in chosen["settlement_details"]
                   if accounts[s["account_ref"]]["account_type_code"] in vocab.REVOLVING_TYPES]
    conditions += [f"release the security on account {s['account_ref']} ({num(accounts[s['account_ref']], 'security_release_days'):.0f} working days)"
                   for s in chosen["settlement_details"] if accounts[s["account_ref"]]["settleability_code"] == vocab.SECURITY_RELEASE]
    expiries = [date.fromisoformat(s["quotation_expiry_date"]) for s in chosen["settlement_details"] if s["quotation_expiry_date"]]
    valid_until = min(expiries + [decision_date + timedelta(days=offer_validity_days)])
    longest = max([num(a, "remaining_term_months") for a in classified_accounts] + [0.0])
    return (code, ranked, primary, reasons.CLIENT_REASONS.version, chosen, conditions, valid_until.isoformat(),
            comparison(baseline, longest, chosen), chosen["settlement_details"])
