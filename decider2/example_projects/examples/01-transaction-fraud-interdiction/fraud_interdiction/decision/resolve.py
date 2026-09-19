"""5.12 — action resolution.

An argmax over a set, under a precedence the Head of Fraud owns and a regulator
is shown. Spec Q7 asks for it "as something an analyst can read and a regulator
can be shown, rather than as ordering logic", which rules out expressing it as
Python `if`s — a five-level tie-break with a critical override and an
allow-listing suppression written as control flow is exactly the artefact
doc 04 §6 says a reviewer cannot verify.

So the precedence is a **document** and this module is a generic kernel over it
(doc 08 §3.4's test: "can one compiled loop evaluate every instance of this
kind, with the instance supplied as arrays?" — yes, it is a lexicographic argmax
over fixed integer columns). Consequence: reordering the tie-break, or changing
which action outranks which, is a **value change with no recompile**, which is
what spec §6.3's "annual; changing it is a major event" deserves — major in
approval, trivial in machinery.

`reads` is the list below. It does not contain `shadow_fire_bits`, and that is
the whole of the shadow isolation guarantee.
"""

from __future__ import annotations

from decider2 import module, precedence

RESOLVER_READS = (
    "live_fire_bits",
    "hard_block_action_floor",
    "rule_attributes",             # action, severity, priority, family, critical, ...
    "severity_shift_by_rule",      # from the overlay scope class
    "action_escalation_by_rule",
    "family_precedence",
    "precedence_version",
)

PRECEDENCE = precedence(
    name="fraud_action_precedence",
    document="config/interdict/precedence.json",
    over="live_fire_bits",
    attributes=("action_rank", "severity", "priority", "family_precedence", "rule_id"),
)


def action_code(
    live_fire_bits: tuple[int, ...],
    hard_block_action_floor: int,
    rule_attributes,
    severity_shift_by_rule: tuple[int, ...],
    action_escalation_by_rule: tuple[int, ...],
    family_precedence: tuple[int, ...],
) -> int:
    """One action. Five resolution stages, in the document's declared order:

      1. floor        — hard_block_action_floor. Nothing may be milder.
      2. critical     — any fired critical rule is binding; a critical `allow`
                        suppresses non-critical firings in the families it names.
      3. collision    — two critical rules with different actions: more severe
                        wins, and a governance exception is emitted to both owners.
      4. max          — most severe action among the remaining firings.
      5. tie-break    — severity desc, priority asc, family precedence, rule_id asc.

    Nothing here is an early exit over the *evidence*: every applicable rule was
    already evaluated by the time this runs, hard block or not.
    """
    pass


def action_source_rule_id(live_fire_bits: tuple[int, ...], action_code: int) -> str:
    """The rule whose action was taken, after the same five stages."""
    pass


def resolution_trace(live_fire_bits: tuple[int, ...]) -> tuple[int, ...]:
    """Every resolution point that discriminated, plus the second and third most
    severe firings.

    "When an analyst asks why her rule did not take effect although it fired"
    this is the answer, and it has to be recorded rather than recomputed,
    because recomputing it needs the precedence version and the attribute table
    as they were. Eight int16s: winning stage, the discriminating field at each
    level, and two runner-up rule indices.
    """
    pass


def governance_exception(live_fire_bits: tuple[int, ...]) -> int:
    """Non-zero when two critical rules fired asking for different actions.

    "Critical rules are meant to be rare and non-overlapping; when they overlap,
    two owners have made incompatible assumptions." The tie-break resolves it
    deterministically; it does not resolve it correctly. This field is what gets
    routed to both owners so that a human does. Spec §11.12.
    """
    pass


def decline_reason_codes(live_fire_bits: tuple[int, ...]) -> tuple[int, ...]:
    """The ordered reason codes of ALL fired rules, ranked through
    core.reason_codes — not only the winner's (spec §5.12.6)."""
    pass


def primary_reason_code(action_source_rule_id: str) -> int:
    """The winner's reason code."""
    pass


Resolve = module(
    action_code,
    action_source_rule_id,
    resolution_trace,
    governance_exception,
    decline_reason_codes,
    primary_reason_code,
    name="resolve",
    reads=RESOLVER_READS,
    precedence=PRECEDENCE,
    taps=["action_code", "action_source_rule_id", "resolution_trace",
          "governance_exception"],
)

# ---------------------------------------------------------------------------
# The counterfactual — spec §5.12, §6.5.2
# ---------------------------------------------------------------------------
# "What would we have decided without the overlay stack?" is asked at every
# monthly fraud forum, and it is the first thing a regulator asks once overlays
# are known to exist. It is not a separate module, a separate mode or a
# separate code path — it is the same module, relabelled onto the base-threshold
# firing set. One line, and it cannot drift from the thing it is the
# counterfactual of.
#
# This needs `.at()` to relabel OUTPUTS as well as inputs. Doc 03 §5.2 only
# relabels inputs. FRAMEWORK-DEMANDS.md #16.
ResolveBase = Resolve.at(
    name="resolve_base",
    inputs={
        "live_fire_bits": "live_base_fire_bits",
        "severity_shift_by_rule": "zero_severity_shift",
        "action_escalation_by_rule": "zero_action_escalation",
    },
    outputs={
        "action_code": "counterfactual_action_code",
        "action_source_rule_id": "counterfactual_source_rule_id",
        "decline_reason_codes": "counterfactual_reason_codes",
        "resolution_trace": "counterfactual_resolution_trace",
        "governance_exception": "counterfactual_governance_exception",
        "primary_reason_code": "counterfactual_primary_reason_code",
    },
)


def fired_on_overlay_ids(
    live_fire_bits: tuple[int, ...],
    live_base_fire_bits: tuple[int, ...],
) -> tuple[int, ...]:
    """Rules that fired ONLY because an overlay moved a threshold.

    `live_fire_bits & ~live_base_fire_bits`. Ten `andnot` instructions. This is
    the field that makes "what did the overlay actually buy" a subtraction
    rather than a study.
    """
    pass


OverlayAttribution = module(fired_on_overlay_ids, name="overlay_attribution",
                           taps=["fired_on_overlay_ids"])
