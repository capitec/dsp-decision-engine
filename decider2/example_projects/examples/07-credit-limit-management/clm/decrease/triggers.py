"""The fourteen decrease triggers (s5.7), as a panel with a different reducer.

Decreases are NOT subject to the s5.2 exclusions -- an account in arrears is
excluded from increases and is a prime candidate for a decrease -- and not
subject to the consent requirement, because a reduction is a unilateral act the
Bank is entitled to take with reasons and notice. So this is a separate
pipeline over the same book, not a branch inside the increase path.

Every trigger records its OBSERVED VALUE against its THRESHOLD AS APPLIED,
because s9.1 requires a branch consultant to answer "why was my limit reduced"
in one conversation with the trigger, the threshold, the observed value and the
date of observation. `evidence=["*", "observed_value", "threshold_applied"]`
is what makes that a row rather than a query.
"""

from decider2 import module, panel, param, shadow, table
from decider2.credit import overlay_point
from clm.vocabulary import DECREASE_TRIGGERS

DecreaseThresholds = table(
    "clm.decrease_thresholds",
    keys=("trigger_code", "product_code"),
    values={"threshold": float, "window_cycles": int, "notice_class_code": int,
            "reduction_rule_code": int, "reduction_parameter": float},
    domain={"trigger_code": range(1, 15), "product_code": [20, 21]},
    dense=True, effective_dated=True,
    owner="Credit Risk Policy + Collections",
    source="artefacts/decrease_triggers.csv",
)

ApplyCutoffOverlays = overlay_point(
    "decrease_cutoff_overlays",
    register="overlay_set",
    adjusts={"threshold_applied": "replace"},
    scope_keys=["trigger_code", "product_code", "behaviour_grade"],
    order="declared",
    evidence=["adjustments_applied"],
)
# The archetypal use: tightening D01 from 60 points to 45 for two quarters
# during a deteriorating cycle. It must expire, and a lapsed overlay stops the
# daily run exactly as it stops the monthly cycle.


def d01_score_deterioration(behaviour_score: float, behaviour_score_3m_ago: float,
                            behaviour_grade: int, threshold_applied: float) -> bool:
    """Drop of at least 60 points over 3 months, landing in grade 8 or worse."""
    return (behaviour_score_3m_ago - behaviour_score) >= threshold_applied \
        and behaviour_grade >= 8


def d02_grade_migration(behaviour_grade: int, behaviour_grade_3m_ago: int,
                        threshold_applied: float) -> bool:
    """Worsens by at least 3 notches into grades 9-12."""
    pass


def d03_emerging_arrears(cycles_past_due_current: int) -> bool:
    """One cycle past due in the current month. Immediate class."""
    return cycles_past_due_current >= 1


def d05_over_limit_persistence(over_limit_cycles_4: int, threshold_applied: float) -> bool:
    """Over limit at cycle end in 3 of the last 4 cycles. Immediate class."""
    pass


def d07_bureau_distress(new_default_listing: bool, new_judgment: bool,
                        new_administration_order: bool, new_debt_review: bool) -> bool:
    """A new adverse listing at any provider. Immediate class, reduce to balance."""
    pass


def d12_dormancy(current_balance_c: int, months_since_last_transaction: int,
                 threshold_applied: float) -> bool:
    """Zero balance and no transaction for 9 months. Notice class, and the ONLY
    trigger that does not suppress increases on the client's other accounts --
    dormancy is not a risk signal (s5.7)."""
    pass


def d14_income_disappearance(deposit_months_absent: int,
                             deposit_months_present_before: int) -> bool:
    """Verified salary deposit absent for 2 consecutive months where previously
    present for 6."""
    pass


# d04, d06, d08, d09, d10, d11, d13 follow the same one-module shape.

DecreaseTriggers = panel(
    "decrease_triggers",
    members=[module(d01_score_deterioration, name="d01"),
             module(d02_grade_migration, name="d02"),
             module(d03_emerging_arrears, name="d03"),
             module(d05_over_limit_persistence, name="d05"),
             module(d07_bureau_distress, name="d07"),
             module(d12_dormancy, name="d12"),
             module(d14_income_disappearance, name="d14")],   # + d04,06,08,09,10,11,13
    reduce="any",
    codes=DECREASE_TRIGGERS,
    writes={"value": "decrease_required",
            "fired": "trigger_codes",
            "primary": "primary_trigger_code",       # ranked by severity_rank
            "worst_attribute": ("notice_class", "notice_class_code")},
    evidence=["*", "observed_value", "threshold_applied"],
)

# Model Risk and Credit Committee both ask "what would we have done without the
# cut-off shift". Same mechanism as the matrix.
Triggers = (DecreaseThresholds.bind_to("threshold_applied")
            | ApplyCutoffOverlays
            | DecreaseTriggers) | shadow(
    DecreaseThresholds.bind_to("threshold_applied") | DecreaseTriggers,
    neutralise={"overlay_set": "off"},
    suffix="_unadjusted",
    keep=["decrease_required", "trigger_codes"],
)
