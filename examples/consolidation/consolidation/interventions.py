"""The policy interventions a scenario must pass, each recorded with its actual and threshold values.

A violated intervention removes the scenario from selection (it is not scored down), all of them are evaluated rather
than only the first, and one that does not apply is `not_applicable`, never `passed`. The anti-harm ceiling can be
tightened by an approved overlay: the record then carries the overlay's id beside the base value it replaced.
"""
from __future__ import annotations

from datetime import date

from consolidation import overlays

PASSED, FAILED, NOT_APPLICABLE = "passed", "failed", "not_applicable"


def evaluate(facts: dict, policy: dict, decision_date: date, consolidations_24m: int, max_consolidations_24m: int,
             overlay_set: tuple, stack_enabled: bool = True) -> list[dict]:
    """All fourteen interventions for one priced scenario, by number.

    Numbers 2, 7, 11 and 12 are enforced earlier, when accounts are classified and the client is routed, so here they
    are `not_applicable`.

    `overlay_set` is `OverlaySet.packed()` of the overlays on the thresholds (`anti_harm`, scoped by `product_code`).

    `facts` holds the scenario's `accounts`, `relief_pct`, `cost_increase_pct`, `term_extension_months`, `new_rate`,
    `settled_rate`, `new_money`, `settlement_total`, `dsr`, `discretionary_after` and `external_proportion`.
    """
    anti_harm, anti_harm_overlays = overlays.apply(overlay_set, "anti_harm", policy["anti_harm"],
                                                   {"product_code": policy["product_code"]}, decision_date, stack_enabled)
    new_money_cap = min(facts["settlement_total"] * policy["new_money_pct"], policy["new_money_cap"])
    rows = [  # number, name, actual, threshold, passes, applies
        (1, "max_accounts_settled", facts["accounts"], policy["max_accounts"], facts["accounts"] <= policy["max_accounts"], True),
        (2, "recently_opened_account", None, None, True, False),
        (3, "min_instalment_reduction", facts["relief_pct"], policy["min_relief"],
         facts["relief_pct"] >= policy["min_relief"], True),
        (4, "anti_harm_ceiling", facts["cost_increase_pct"], anti_harm,
         facts["cost_increase_pct"] <= anti_harm, True),
        (5, "max_term_extension", facts["term_extension_months"], policy.get("max_term_extension"),
         facts["term_extension_months"] <= policy.get("max_term_extension", 0), "max_term_extension" in policy),
        (6, "rate_ceiling", facts["new_rate"], facts["settled_rate"], facts["new_rate"] <= facts["settled_rate"],
         policy["rate_ceiling"]),
        (7, "debt_review", None, None, True, False),
        (8, "new_money_cap", facts["new_money"], new_money_cap, facts["new_money"] <= new_money_cap, True),
        (9, "post_consolidation_dsr", facts["dsr"], policy["dsr_ceiling"], facts["dsr"] <= policy["dsr_ceiling"], True),
        (10, "min_external_proportion", facts["external_proportion"], policy.get("min_external"),
         facts["external_proportion"] >= policy.get("min_external", 0.0), "min_external" in policy),
        (11, "disputed_account", None, None, True, False),
        (12, "security_release", None, None, True, False),
        (13, "max_consolidations_24m", consolidations_24m, max_consolidations_24m,
         consolidations_24m < max_consolidations_24m, True),
        (14, "min_discretionary_income", facts["discretionary_after"], policy["min_discretionary"],
         facts["discretionary_after"] >= policy["min_discretionary"], True),
    ]
    return [{
        "intervention": number, "name": name, "status": (PASSED if passes else FAILED) if applies else NOT_APPLICABLE,
        "actual": actual if applies else None, "threshold": threshold if applies else None,
        "base_threshold": (policy["anti_harm"] if number == 4 else threshold) if applies else None,
        "overlay_id": anti_harm_overlays[0] if number == 4 and anti_harm_overlays else None,
    } for number, name, actual, threshold, passes, applies in rows]
