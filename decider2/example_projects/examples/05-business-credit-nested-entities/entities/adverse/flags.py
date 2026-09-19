"""Event-grain predicates that the Entity-grain roll-up counts and sums.

Every one of these is a plain boolean step over one event. They exist as named,
registered steps -- rather than as lambdas or as expressions inside the fold --
for four reasons, all of which the spec forces:

1. **They are what attribution names.** s5.6 requires a count-based rule to
   attribute to *all* the events that satisfied it. The fold that counts
   `is_minor_recent` carries the set of events where `is_minor_recent` was true
   (see gather.py). If the predicate were anonymous, so would the witness be.

2. **They are testable in isolation** -- `is_minor_recent.score(...)` -- which
   is the whole of doc 03 s11 and is how Policy signs the window off.

3. **They compile.** A predicate inside a fold would have to be either an
   expression string (forbidden, doc 08 s3.2) or a closure (not njit-able).

4. **They are the tuning surface.** Change scenario 4 -- "three-minor-in-12
   becomes four-in-18, for peripheral entities only" -- moves the window here
   and the count in verdict.py, and moves nothing else. See the note on
   `count_window_months` below, which is where that scenario stops being a
   param change and becomes a table change.
"""

from decider2 import module, param, step, table
from grains import Event, IMMATERIAL, MINOR, MATERIAL, DISQUALIFYING


# --------------------------------------------------------------------------
# s6.1 lists "Minor-event count window: 3 events in 12 months" as ONE parameter.
# It is two, and they live in two places: the window is a property of the event
# (here), the count is a property of the rule (verdict.py). Keeping them
# together in one param would make the window unobservable from the event
# record, which is what the committee pack needs to show.
#
# Change scenario 4 makes the window vary by criticality class. At that point
# `param(12.0)` becomes a table read keyed on the broadcast `criticality_class`
# -- which is a *signature change*, and therefore a recompile and an engineer.
# That is the sharpest ergonomic failure in this sketch and it is D19 in
# FRAMEWORK-DEMANDS: a param and a one-key table should be the same declaration
# at different arities, so promoting one to the other is a values change.
# --------------------------------------------------------------------------

def is_minor(event_severity_code: int) -> bool:
    """Classified minor."""
    pass


def is_material(event_severity_code: int) -> bool:
    """Classified material."""
    pass


def is_disqualifying(event_severity_code: int) -> bool:
    """Classified disqualifying. AE-R-01 counts these."""
    pass


def is_minor_recent(
    event_severity_code: int,
    event_age_months: float,
    count_window_months: float = param(
        12.0, ge=1.0, le=60.0,
        description="AE-R-02 window. Policy, quarterly and after loss events."),
) -> bool:
    """Minor, dated within the count window. AE-R-02 counts these."""
    pass


def is_material_within_24m(
    event_severity_code: int,
    event_age_months: float,
    material_window_months: float = param(24.0, ge=1.0, le=120.0),
) -> bool:
    """Material, dated within 24 months. AE-R-04 and AE-R-05 count these."""
    pass


def is_material_within_recency(
    event_severity_code: int,
    event_age_months: float,
    recency_window_months: float = param(
        6.0, ge=1.0, le=36.0,
        description="s6.1 recency window. AE-R-08 floors the verdict at material "
                    "and sets recent_adverse, which blocks entity grades 1-3."),
) -> bool:
    """Material, within the recency window. AE-R-08 counts these."""
    pass


def is_unsatisfied(is_satisfied: bool, status_code: int) -> bool:
    """Still outstanding. AE-R-06 and AE-R-07 sum the amounts of these."""
    pass


def is_dishonoured_within_12m(
    event_type_code: int,
    event_age_months: float,
    velocity_window_months: float = param(12.0, ge=1.0, le=36.0),
) -> bool:
    """Dishonoured payment within the velocity window. AE-R-09 counts these."""
    pass


def in_trailing_window(
    event_age_months: float,
    trend_window_months: float = param(12.0, ge=1.0, le=36.0),
) -> bool:
    """Dated in the trailing window. AE-R-10's first term."""
    pass


def in_preceding_window(
    event_age_months: float,
    trend_window_months: float = param(12.0, ge=1.0, le=36.0),
) -> bool:
    """Dated in the window before the trailing one. AE-R-10's second term."""
    pass


def counts_toward_clear(event_severity_code: int) -> bool:
    """Immaterial. AE-R-11's 'all events immaterial and count <= 2'."""
    pass


def is_write_off_or_fraud(event_type_code: int) -> bool:
    """AE-C-18 or AE-C-20 fired. AE-R-12's exception to the peripheral cap."""
    pass


def contributes_to_dispute_shadow(classification_provisional: bool) -> bool:
    """Masked out by `evidence/counterfactual.py`'s DisputeShadow."""
    pass


EventFlags = module(
    is_minor, is_material, is_disqualifying,
    is_minor_recent, is_material_within_24m, is_material_within_recency,
    is_unsatisfied, is_dishonoured_within_12m,
    in_trailing_window, in_preceding_window,
    counts_toward_clear, is_write_off_or_fraud,
    contributes_to_dispute_shadow,
    name="event_flags",
    grain=Event,
    contract="contracts/event_flags.json",
)
