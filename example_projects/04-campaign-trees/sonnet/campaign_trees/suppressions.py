"""Stage 2 -- global suppressions (spec 04 §5.2). A representative subset of the ~34-entry
registry (SCOPE.md's breadth cut), covering both suppression classes (§5.2 requirement 3):

  - **absolute** -- the client is removed and not evaluated further;
  - **measurement-relevant** -- the client is still evaluated through the tree and the path
    recorded, because the analytics team reports the population that *would* have been
    targeted (§5.2 requirement 3, §5.7 item 4).

Every suppression that applies is recorded, not merely the first (§5.2 requirement 1: "A
client suppressed by four different rules must show four").
"""
from __future__ import annotations

from decider import missing_as, step

from campaign_trees import vocab

# (code, description, class) -- a subset of §5.2's ~34-entry table, picked to cover both
# classes and both "all campaigns" and "per campaign" scope.
REGISTRY: dict[str, tuple[str, str]] = {
    "S01": ("Deceased or estate in progress", vocab.SUPPRESSION_ABSOLUTE),
    "S02": ("Debt review or debt counselling", vocab.SUPPRESSION_ABSOLUTE),
    "S07": ("Marketing opt-out", vocab.SUPPRESSION_MEASUREMENT_RELEVANT),
    "S08": ("Per-channel consent absent", vocab.SUPPRESSION_MEASUREMENT_RELEVANT),
    "S11": ("Fraud marker on client or device", vocab.SUPPRESSION_ABSOLUTE),
    "S14": ("Compliance do-not-target list", vocab.SUPPRESSION_ABSOLUTE),
    "S21": ("Recent decline, within cooling-off", vocab.SUPPRESSION_MEASUREMENT_RELEVANT),
    "S23": ("Product already held at or above the offered tier", vocab.SUPPRESSION_MEASUREMENT_RELEVANT),
    "S30": ("Contact fatigue, all-channel cap", vocab.SUPPRESSION_MEASUREMENT_RELEVANT),
}


def suppression_evaluation(
    is_deceased: bool = missing_as(False),
    in_debt_review: bool = missing_as(False),
    marketing_opt_out: bool = missing_as(False),
    consent_permitted: bool = missing_as(True),
    fraud_marker: bool = missing_as(False),
    on_do_not_target_list: bool = missing_as(False),
    in_cooling_off: bool = missing_as(False),
    existing_product_at_or_above_tier: bool = missing_as(False),
    contacts_30d: int = missing_as(0),
    all_channel_fatigue_cap: int = missing_as(4),
) -> tuple[list[str], bool]:
    """Every suppression code that applies, plus whether any of them is absolute. A client
    with any absolute code is not evaluated through the tree at all (`pipeline.py` branches
    on `suppressed_absolute`); everything else -- including a measurement-relevant
    suppression -- still is."""
    applied: list[str] = []
    if is_deceased:
        applied.append("S01")
    if in_debt_review:
        applied.append("S02")
    if marketing_opt_out:
        applied.append("S07")
    if not consent_permitted:
        applied.append("S08")
    if fraud_marker:
        applied.append("S11")
    if on_do_not_target_list:
        applied.append("S14")
    if in_cooling_off:
        applied.append("S21")
    if existing_product_at_or_above_tier:
        applied.append("S23")
    if contacts_30d >= all_channel_fatigue_cap:
        applied.append("S30")
    suppressed_absolute = any(REGISTRY[code][1] == vocab.SUPPRESSION_ABSOLUTE for code in applied)
    return applied, suppressed_absolute


suppression_evaluation_step = step(suppression_evaluation, outputs=("suppression_codes", "suppressed_absolute"))
