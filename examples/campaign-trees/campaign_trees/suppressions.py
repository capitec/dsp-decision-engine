"""Global suppressions: every rule that removes a client is recorded, not only the first."""
from __future__ import annotations

from decider import allow_fallback, missing_as, step

# Compliance owns which suppressions are absolute. An absolute one keeps the client out of the tree altogether; any
# other still runs the tree, so the analytics team can report who would have been targeted.
ABSOLUTE = frozenset({"S01", "S02", "S11", "S14"})


@allow_fallback
def suppression_codes(
    is_deceased: bool = missing_as(False), in_debt_review: bool = missing_as(False),
    marketing_opt_out: bool = missing_as(False), channel_permitted: bool = missing_as(True),
    fraud_marker: bool = missing_as(False), on_do_not_target_list: bool = missing_as(False),
    in_cooling_off: bool = missing_as(False), existing_product_at_or_above_tier: bool = missing_as(False),
    contacts_30d: int = missing_as(0), all_channel_fatigue_cap: int = missing_as(4),
) -> list[str]:
    """The code of every suppression that applies: S01 deceased, S02 debt review, S07 opt-out, S08 no channel consent,
    S11 fraud marker, S14 do-not-target list, S21 cooling-off, S23 product already held, S30 contact fatigue."""
    applies = {
        "S01": is_deceased, "S02": in_debt_review, "S07": marketing_opt_out, "S08": not channel_permitted,
        "S11": fraud_marker, "S14": on_do_not_target_list, "S21": in_cooling_off,
        "S23": existing_product_at_or_above_tier, "S30": contacts_30d >= all_channel_fatigue_cap,
    }
    return [code for code, hit in applies.items() if hit]


@allow_fallback
def suppressed_absolute(suppression_codes: list[str]) -> bool:
    return any(code in ABSOLUTE for code in suppression_codes)


steps = (step(suppression_codes), step(suppressed_absolute))
