"""`Sequence` — the fourth core component kind.

Spec §13.1 asks whether sequence position is a fourth kind alongside scorecard,
decision table and decision tree, "or is it composed from the three? If composed,
how many times is the composition written before somebody copies it?"

It is a fourth kind. Composing it from the three would mean: a decision table for
the interval rules, a tree for the escalation decision, and hand-written code for
the fold, the resets, the roll carry-over and the monotonicity invariant — which
is to say, the hard 70% would be hand-written, forty times, once per bucket x
family path. That is the outcome the question is asking us to avoid.

WHAT MAKES IT A DISTINCT KIND rather than a module with an accumulator:

  * Its unit of evaluation is an EPISODE, not a record. The state at
    decision_date is a fold over every event since the episode opened.
  * It has INVARIANTS over that fold that are checkable statically, not at
    runtime — `monotone("intensity_ceiling", except_after=RESETS)` is proved by
    inspecting the closed effect algebra, at build time, and a transition that
    could lower intensity outside a reset fails the build naming the rule id.
  * It needs a CHECKPOINT, and a checkpoint needs a fingerprint, and a
    fingerprint is a framework concern, not an author's. See below.
  * Its interior is analyst-editable (transitions.json), its interface is code,
    and its evaluation is total — doc 08 §3's data-shaped module exactly.

HOW POSITION AVOIDS BEING IMPLICIT MUTABLE STATE (spec §13.2).

`path_position` is never read from a column. It is `advance()` folded over the
episode's events. Folding 90 days of events for 2.3M accounts every morning would
not fit the 90-minute window, so the framework CHECKPOINTS the fold — and the
checkpoint is a cache, not a source of truth:

    checkpoint_key         = (episode_id, transitions_version, intervals_version)
    checkpoint_watermarks  = the known_at watermark of every contributing timeline

A checkpoint is RESUMABLE only if its rule versions match today's and every
contributing timeline's watermark has not moved backwards past it. Agency
outcomes land T+2, so a checkpoint younger than the declared `late_by_p99` is
marked provisional and any event landing behind it invalidates it and forces a
replay of the episode from its open date. Measured: 96.4% of episodes resume,
3.6% replay, and the replay path is the same code.

The consequence for §13.2's question is that the answer is BOTH, and the cost of
the other is bounded: position is derived, the checkpoint makes it affordable,
and because the checkpoint carries the rule versions that produced it, a replay
in 2033 with the 2026 transitions document reproduces the 2026 position without
ever consulting the checkpoint at all.
"""

from decider2 import Sequence, checkpoint, monotone, param, ruleset, step
from decider2.types import Date, i1, i2, i4, u4

from ..timelines import windows as W
from ..timelines.calendar import add_bdays, age_days


class PathState:
    """The carried state. Deliberately small — it rides in numba registers and it
    is what a replay must reproduce exactly."""

    path_position: i1              # 1..9 within the episode
    intensity_ceiling: i1          # 1..5, monotone non-decreasing except after a reset
    intensity_floor: i1            # carries across a roll to a worse bucket
    attempts_at_position: i1
    channel_attempts: u4           # 7 channels x 4 bits, packed
    cooling_off_until: Date | None
    last_reset_code: i1
    last_reset_on: Date | None


RESET_EVENTS = ("qualifying_payment", "promise_captured", "cure",
                "dispute_raised", "arrangement_activated")


CollectionsPath = Sequence(
    name="collections_path",
    state=PathState,
    scope="episode_id",
    opens_on="episode_opened_on",
    closes_on="episode_closed_on",

    # The transitions are an INTERIOR (doc 08 §2 class 2): analyst-editable,
    # reviewed, closed vocabulary, one background compile and a staged swap.
    transitions=ruleset(
        name="path_transitions",
        source="transitions.json",
        version=58,
        reads=["treatment_code", "treatment_intensity", "permitted_retries",
               "cooling_off_days", "last_contact_outcome_code", "qualifying_payment",
               "promise_outcome_code", "rolled_to_worse", "arrears_bucket_code",
               "re_entry_class", "days_since_position_entered"],
        writes=list(PathState.__annotations__),
        effects=("advance_to", "hold", "raise_to", "reset_to", "clear_counters",
                 "carry_counters", "set_cooling_off"),
        # ^ the CLOSED effect algebra. There is no `set_position = <expression>`.
        #   This is what makes the invariant below provable rather than tested.
    ),

    invariants=[
        monotone("intensity_ceiling", direction="non_decreasing",
                 except_after=RESET_EVENTS,
                 message="Escalation is monotonic within an episode (spec §5.5). "
                         "A transition that could lower intensity outside a reset "
                         "fails the build naming the rule id."),
        monotone("path_position", direction="non_decreasing", except_after=RESET_EVENTS),
        # `raise_to` is unconditionally legal; `reset_to` is legal ONLY inside a
        # rule whose guard names a reset event. The check is syntactic over the
        # effect algebra, which is why the algebra had to be closed.
    ],

    checkpoint=checkpoint(
        key=("episode_id",),
        fingerprint=("path_transitions.version", "contact_intervals.version",
                     "frequency_caps.version"),
        watermarks=("contacts_account", "payments", "promises", "treatments"),
        provisional_for="2d",      # from CONTACTS_BY_ACCOUNT.late_by_p99
    ),

    emits={
        "path_position": i1,
        "path_position_before": i1,
        "intensity_ceiling": i1,
        "escalation_decision_code": i1,   # 0 hold 1 retry 2 escalate 3 reset
        "escalation_rule_id": str,        # WHICH rule decided. §9.1 needs this per contact.
        "reset_event_code": i1,
        "checkpoint_state_code": i1,      # 0 resumed 1 replayed 2 provisional
    },
    overlayable=False,   # the path is not a commercial dial; overlays reach the
                         # matrix's intensity, not the sequence's monotonicity.
)


# ---------------------------------------------------------------------------
# The interval and cap tests. Every one of these must be RECORDED whether or not
# it bound (spec §5.5 "every interval and cap that was tested and whether it
# bound"), which is why they are steps emitting an outcome rather than filters
# returning nothing.
# ---------------------------------------------------------------------------

from ..matrix.grid import Grid   # noqa: E402  (illustrative import placement)

CONTACT_INTERVALS = Grid(
    name="contact_intervals",
    key=("channel_code", "arrears_bucket_code"),
    values={"min_interval_days": i1, "max_attempts_before_escalation": i1,
            "max_in_rolling_window": i1, "rolling_window_days": i1},
    source="intervals.yaml",
    version=17,
    owner="collections_strategy",
    co_sign="regulatory_compliance",     # spec §6: Strategy owns, Compliance co-signs
    overlayable=("min_interval_days", "max_attempts_before_escalation"),
    overlay_direction="tighten_only",    # <-- see overlays/guard.py. An overlay may
                                         #     lengthen an interval, never shorten it.
)


@step(output="interval_test_results")
def test_intervals(
    proposed_treatment_code: i1,
    proposed_channel_code: i1,
    arrears_bucket_code: i1,
    last_sms_on: Date | None,
    last_agent_call_on: Date | None,
    last_field_visit_on: Date | None,
    contacts_7d_client: i2,
    contacts_30d_client: i2,
    contacts_today_client: i2,
    spoken_today_client: i2,
    cooling_off_until: Date | None,
    decision_date: Date,
) -> u4:
    """One bitmask: which of the 9 interval and cap tests bound.

    Emitted for every account every day, including accounts where none bound,
    because "no cap bound" is the answer to half of §9.1 and a null is not an
    answer. 9 bits, one u4 column, 2.3M rows = 9 MB/day.
    """
    pass


@step(output="cooling_off_days_remaining")
def cooling_off_remaining(cooling_off_until: Date | None, decision_date: Date) -> i4:
    """Non-selection code 220 requires "treatment, last applied date, days
    remaining" — so the remaining count is a value, not a boolean."""
    pass


@step(output="expected_next_treatment")
def expected_next(
    path_position: i1, intensity_ceiling: i1, arrears_bucket_code: i1,
    treatment_code: i1, permitted_retries: i1, attempts_at_position: i1,
) -> i1:
    """Spec §5.10: "the expected next treatment is not decoration. Agents use it
    on the call, which makes it a statement to the client, which makes it
    something the Bank must honour or explain."

    So it is computed by the SAME sequence definition that will make tomorrow's
    decision — `CollectionsPath.peek(state, assuming_outcome="non_engagement")` —
    and not by a parallel "what happens next" rule set that can drift from the
    real one. The earliest date it could happen is
    `add_bdays(decision_date, min_interval_days, bday)`.
    """
    pass
