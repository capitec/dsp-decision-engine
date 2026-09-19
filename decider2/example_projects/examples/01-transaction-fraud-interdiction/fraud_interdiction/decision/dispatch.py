"""5.13 and 5.14 — dispatch, obligations, and client-facing wording.

Three things here are easy to get wrong in a way that only shows up in an
Ombud complaint eighteen months later, so all three are recorded artefacts
rather than derived-on-read:

  * the queue actually used, after capacity-aware demotion;
  * the challenge actually offered, after the consent and trust constraint,
    including the escalation when no permitted challenge exists;
  * the wording actually shown, alongside the internal reason set it came from.
"""

from __future__ import annotations

from decider2 import module, param
from decider2.tables import DecisionTable, Grid

CHALLENGE_MATRIX = Grid(
    name="challenge_matrix",
    dims=("challenge_type", "segment_id", "channel_code"),
    # Three cell states, not two. Spec §11.13: "the challenge matrix must
    # express 'not permitted' distinctly from 'not configured'". A boolean grid
    # would have silently turned a withdrawn SMS OTP into an unconfigured cell
    # and 60 rules would have escalated to hold_for_review overnight.
    cell_states=("permitted", "not_permitted", "not_configured"),
    document="tables/data/challenge_matrix.csv",
)

QUEUE_MATRIX = DecisionTable(
    name="queue_sla",
    conditions=["queue_code", "queue_depth_band"],
    outcomes=["effective_queue_code", "sla_tier", "overflow_applied"],
    document="tables/data/queue_sla.yaml",
)


def effective_queue(
    action_code: int,
    action_source_rule_id: str,
    queue_depth_band: int,
    queue_reroute_by_rule: tuple[int, ...],
) -> tuple[int, int, bool]:
    """(queue, sla_tier, overflow_applied).

    Queue choice comes from the winning rule, then a queue-reroute overlay,
    then capacity-aware demotion. "The overflow decision is part of the record"
    — because Fraud Ops is fixed capacity and a change that increases holds by
    30% produces a breached SLA, not 30% more reviews (spec §3).
    """
    pass


def challenge_selection(
    action_code: int,
    challenge_type_requested: int,
    client_segment_bits: int,
    channel_code: int,
    challenge_trust_bits: int,
) -> tuple[int, int]:
    """(challenge_type, escalation_code).

    "Where no permitted challenge exists the action escalates to
    hold_for_review, recorded as an escalation rather than silently." The
    escalation code distinguishes not-permitted from not-configured, so the
    weekly report separates "we withdrew SMS OTP" from "nobody filled in this
    cell".
    """
    pass


def obligations_raised(
    action_code: int,
    live_fire_bits: tuple[int, ...],
    hard_block_code: int,
) -> tuple[tuple[int, int, int], ...]:
    """(obligation_code, clock_start_us, addressee).

    Certain firings raise a suspicious-activity reporting obligation with a
    statutory clock. "The engine does not file the report; it starts the clock,
    names the trigger, and is the evidence of when awareness arose."

    The service runs 24/365 with no batch window, so no obligation may be
    deferred to "the next run". There is no next run.
    """
    pass


def downstream_instructions(action_code: int, channel_code: int) -> tuple[int, ...]:
    """Which systems must act on freeze_account or block_channel, and whether
    each instruction is idempotent on retry."""
    pass


def client_wording(
    action_code: int,
    primary_reason_code: int,
    client_language: int,
    disclosure_level: int = param(2, ge=1, le=4, unit="level"),
) -> tuple[int, str, int, int]:
    """(wording_key, rendered_text, language, disclosure_level).

    Fraud wording is deliberately less specific than credit decline wording:
    telling a client which velocity threshold they crossed tells the fraudster
    holding their phone the same thing.

    The record holds BOTH the internal reason set and the external wording, and
    the mapping between them, because "a complaint six months later is about
    what the client was told, a scheme representment about what was true".
    Wording changes; what the client was told on the day does not — so the
    rendered text is stored, not the key alone.
    """
    pass


Dispatch = module(
    effective_queue,
    challenge_selection,
    obligations_raised,
    downstream_instructions,
    name="dispatch",
    taps=["effective_queue", "challenge_selection", "obligations_raised"],
)

Wording = module(
    client_wording,
    name="wording",
    taps=["client_wording"],
)
