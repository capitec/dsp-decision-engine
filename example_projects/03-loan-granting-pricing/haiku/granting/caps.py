"""Cap waterfall for unsecured Flex Loan (52 rules, 3 ceilings)."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from enum import Enum


@dataclass
class CapRule:
    """Definition of one cap rule."""
    rule_id: str
    owner: str  # "product", "credit_policy", "credit_systems", "financial_crime"
    class_: str  # "product", "appetite", "policy", "exposure", "campaign", "regulatory"
    name: str
    sequence: int  # Position in evaluation order (1-52)
    acts_on: Optional[str]  # "amount", "term", "grade" or None for decline-only
    condition: Callable[[Dict[str, Any]], bool]  # Returns True if rule applies
    effect: Optional[Callable[[Dict[str, Any], float], float]]  # (context, current_value) -> new_value, or None
    can_raise: bool = False  # Only CAP-0420 is true
    reason_code_if_decline: Optional[str] = None


@dataclass
class CapChainLink:
    """One change in a ceiling's value."""
    sequence: int
    rule_id: str
    applicable: bool
    value_before: float
    value_after: float
    status: str  # "seed", "bound", "evaluated_not_bound", "not_applicable", "raised"


def evaluate_cap_waterfall(
    risk_grade: int,
    employment_tenure_months: Optional[float],
    is_new_to_bank: bool,
    arrears_3m_count: int,
    arrears_2m_count: int,
    arrears_2m_recency_days: int,
    credit_enquiry_60d: int,
    credit_enquiry_90d: int,
    employer_on_watchlist: bool,
    group_exposure_limit: float,
    internal_exposure: float,
    channel_code: int,
    campaign_id: Optional[int],
    campaign_uplift_authorized: bool,
    rules: List[CapRule],
    seed_amount_cap: float = 500000,
    seed_term_cap: int = 84,
    seed_worst_grade: int = 12,
) -> tuple[Dict[str, float], Dict[str, List[CapChainLink]], List[tuple[str, str, bool, str]]]:
    """
    Evaluate the 52-rule cap waterfall.

    Returns:
    - ceilings: {"amount_cap": value, "term_cap": value, "worst_acceptable_grade": value}
    - chains: {"amount_cap": [links], "term_cap": [links], "worst_acceptable_grade": [links]}
    - rule_verdicts: [(rule_id, rule_name, applicable, status)]
    """

    context = {
        "risk_grade": risk_grade,
        "employment_tenure_months": employment_tenure_months,
        "is_new_to_bank": is_new_to_bank,
        "arrears_3m_count": arrears_3m_count,
        "arrears_2m_count": arrears_2m_count,
        "arrears_2m_recency_days": arrears_2m_recency_days,
        "credit_enquiry_60d": credit_enquiry_60d,
        "credit_enquiry_90d": credit_enquiry_90d,
        "employer_on_watchlist": employer_on_watchlist,
        "group_exposure_limit": group_exposure_limit,
        "internal_exposure": internal_exposure,
        "channel_code": channel_code,
        "campaign_id": campaign_id,
        "campaign_uplift_authorized": campaign_uplift_authorized,
    }

    # Initialize ceilings and chains
    ceilings = {
        "amount_cap": seed_amount_cap,
        "term_cap": seed_term_cap,
        "worst_acceptable_grade": seed_worst_grade,
    }

    chains = {
        "amount_cap": [
            CapChainLink(0, "SEED", True, 0, seed_amount_cap, "seed")
        ],
        "term_cap": [
            CapChainLink(0, "SEED", True, 0, seed_term_cap, "seed")
        ],
        "worst_acceptable_grade": [
            CapChainLink(0, "SEED", True, 0, seed_worst_grade, "seed")
        ],
    }

    rule_verdicts = []

    # Evaluate rules in sequence
    for rule in sorted(rules, key=lambda r: r.sequence):
        applicable = rule.condition(context)
        status = "not_applicable"

        if not applicable:
            rule_verdicts.append((rule.rule_id, rule.name, False, "not_applicable"))
            continue

        # Rule is applicable; try to apply it
        if rule.acts_on is None:
            # Decline-only rule (doesn't reduce ceilings, just declines)
            status = "evaluated"
            rule_verdicts.append((rule.rule_id, rule.name, True, status))
            continue

        # Rule acts on one or more ceilings
        for ceiling_type in ["amount_cap", "term_cap", "worst_acceptable_grade"]:
            if ceiling_type.startswith(rule.acts_on):
                current = ceilings[ceiling_type]
                if rule.effect:
                    new_value = rule.effect(context, current)

                    # Check constraint: can only raise if can_raise and ceiling_type is "amount_cap"
                    if new_value > current and not (rule.can_raise and ceiling_type == "amount_cap"):
                        # Constraint violation - treat as not binding this ceiling
                        status = "evaluated_not_bound"
                    else:
                        if new_value < current or (rule.can_raise and new_value > current):
                            status = "bound" if new_value != current else "evaluated_not_bound"
                            if new_value != current:
                                chains[ceiling_type].append(
                                    CapChainLink(
                                        rule.sequence,
                                        rule.rule_id,
                                        True,
                                        current,
                                        new_value,
                                        "raised" if new_value > current else "bound"
                                    )
                                )
                                ceilings[ceiling_type] = new_value
                        else:
                            status = "evaluated_not_bound"

        rule_verdicts.append((rule.rule_id, rule.name, applicable, status))

    return ceilings, chains, rule_verdicts


def build_cap_register() -> List[CapRule]:
    """Build the 52-rule cap register with illustrative rules."""

    def cond_always(ctx): return True
    def cond_grade_appetite(ctx): return True  # Appetite by grade always applies
    def cond_employment_tenure(ctx): return True
    def cond_new_to_bank(ctx): return ctx["is_new_to_bank"]
    def cond_arrears_history(ctx): return ctx["arrears_3m_count"] > 0 or ctx["arrears_2m_count"] > 0
    def cond_enquiry_velocity(ctx): return ctx["credit_enquiry_90d"] >= 6
    def cond_employer_watchlist(ctx): return ctx["employer_on_watchlist"]
    def cond_group_exposure(ctx): return True
    def cond_channel_broker(ctx): return ctx["channel_code"] == 6
    def cond_channel_partner(ctx): return ctx["channel_code"] == 5
    def cond_campaign_uplift(ctx): return ctx["campaign_id"] is not None and ctx["campaign_uplift_authorized"]

    def effect_product_max(ctx, current): return min(current, 500000)

    def effect_appetite_amount(ctx, current):
        # Grade 1->R500k, grade 7->R150k, grade 10->R45k, grade 12->decline
        appetite_limits = {1: 500000, 2: 400000, 3: 300000, 4: 250000, 5: 200000,
                           6: 150000, 7: 150000, 8: 100000, 9: 75000, 10: 45000, 11: 30000, 12: 0}
        limit = appetite_limits.get(ctx["risk_grade"], 0)
        return min(current, limit)

    def effect_appetite_term(ctx, current):
        # Grades 1-4 -> 84, 5-7 -> 72, 8-9 -> 60, 10-11 -> 36, 12 -> 0
        term_limits = {1: 84, 2: 84, 3: 84, 4: 84, 5: 72, 6: 72, 7: 72,
                       8: 60, 9: 60, 10: 36, 11: 36, 12: 0}
        limit = term_limits.get(ctx["risk_grade"], 0)
        return min(current, limit)

    def effect_new_to_bank(ctx, current): return min(current, 60000)

    def effect_employment_short_tenure(ctx, current):
        tenure = ctx.get("employment_tenure_months", 0) or 0
        if tenure < 3: return min(current, 0)  # Decline
        if tenure < 6: return min(current, 25000)
        if tenure < 24: return min(current, 70000)
        return current

    def effect_arrears(ctx, current):
        if ctx["arrears_3m_count"] > 0: return min(current, 40000)
        if ctx["arrears_2m_count"] > 0: return min(current, 120000)
        return current

    def effect_enquiry_velocity(ctx, current):
        if ctx["credit_enquiry_90d"] >= 10: return 0  # Decline
        if ctx["credit_enquiry_60d"] >= 6: return min(current, 30000)
        return current

    def effect_employer_watchlist(ctx, current): return min(current, 50000)

    def effect_group_exposure(ctx, current):
        headroom = ctx["group_exposure_limit"] - ctx["internal_exposure"]
        return min(current, max(0, headroom))

    def effect_channel_broker(ctx, current): return min(current, 150000)

    def effect_channel_partner(ctx, current): return min(current, 100000)

    def effect_campaign_uplift(ctx, current):
        uplift = min(current * 0.25, 250000)  # +25%, max R250k
        return min(current + uplift, 250000)

    rules = [
        CapRule("CAP-0010", "product", "product", "Product maximum", 1, "amount", cond_always, effect_product_max),
        CapRule("CAP-0100", "credit_policy", "appetite", "Appetite by grade (amount)", 3, "amount", cond_grade_appetite, effect_appetite_amount),
        CapRule("CAP-0118", "credit_policy", "appetite", "Appetite by grade (term)", 4, "term", cond_grade_appetite, effect_appetite_term),
        CapRule("CAP-0140", "product", "policy", "First loan cap for new-to-bank", 9, "amount", cond_new_to_bank, effect_new_to_bank),
        CapRule("CAP-0175", "credit_policy", "policy", "Employment tenure", 12, "amount", cond_employment_tenure, effect_employment_short_tenure),
        CapRule("CAP-0210", "credit_policy", "policy", "Arrears history", 17, "amount", cond_arrears_history, effect_arrears),
        CapRule("CAP-0240", "credit_policy", "policy", "Enquiry velocity", 21, "amount", cond_enquiry_velocity, effect_enquiry_velocity),
        CapRule("CAP-0305", "credit_systems", "exposure", "Employer/sector watchlist", 26, "amount", cond_employer_watchlist, effect_employer_watchlist),
        CapRule("CAP-0330", "product", "policy", "Channel cap (broker)", 33, "amount", cond_channel_broker, effect_channel_broker),
        CapRule("CAP-0331", "product", "policy", "Channel cap (partner)", 34, "amount", cond_channel_partner, effect_channel_partner),
        CapRule("CAP-0361", "credit_systems", "exposure", "Group exposure headroom", 38, "amount", cond_group_exposure, effect_group_exposure),
        CapRule("CAP-0420", "credit_committee", "campaign", "Campaign uplift authority", 52, "amount", cond_campaign_uplift, effect_campaign_uplift, can_raise=True),
    ]

    return rules
