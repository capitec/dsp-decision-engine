"""5.8 — segment and applicability resolution.

The awkward requirement, quoted exactly:

    "Recorded: rule_set_version; counts applicable by family; and — the awkward
     one — enough to reconstruct the applicable set exactly, without storing 635
     rule identifiers on every one of 140 million monthly records."

The answer is that applicability is a **pure function of six recorded scalars**,
so the record stores the arguments and not the result:

    applicable(rule) =
          event_type_code   in rule.event_type_mask
      and client_segment_bits & rule.segment_mask
      and rule.effective_from <= event_timestamp < rule.effective_to
      and not (degraded_mode suspends rule)
      and (rule.activated_in_modes is empty or degraded_mode in it)
      and not overlay_scope_restriction(rule)
      and not rule.circuit_open

Six scalars — event_type_code (2 B), client_segment_bits (8 B),
event_timestamp (8 B), degraded_mode_code (2 B), rule_set_version (4 B),
adjustment_stack_version (4 B) — is 28 bytes against 635 identifiers. Replay
recomputes the set from them, pinned to the generation.

Recomputation is trusted only because it is also **checked**: a 64-bit
`applicable_digest` over the resulting bitset is stored alongside. If a replay
reconstructs a different applicable set, the digest disagrees and the replay
fails loudly instead of quietly answering a different question. 8 bytes buys the
difference between a derivation and an assumption. FRAMEWORK-DEMANDS.md #8.
"""

from __future__ import annotations

from decider2 import module, param


def applicable_bits(
    event_type_code: int,
    client_segment_bits: int,
    event_timestamp: int,
    degraded_mode_code: int,
    mode_suspended_rule_bits: tuple[int, ...],
    mode_activated_rule_bits: tuple[int, ...],
    overlay_scope_restriction_bits: tuple[int, ...],
    circuit_open_bits: tuple[int, ...],
) -> tuple[int, ...]:
    """The applicable live population as a 10 x uint64 bitset over 635 slots.

    All eight inputs are already columns of the feature vector, so this step is
    pure integer work: ten iterations of mask-and. It is the only place the
    ordering of rules in the block layout is observable, and that ordering is
    the generation's, pinned.
    """
    pass


def applicable_digest(applicable_bits: tuple[int, ...]) -> int:
    """64-bit digest of the applicable set. Stored; checked on every replay."""
    pass


def applicable_counts_by_family(applicable_bits: tuple[int, ...]) -> tuple[int, ...]:
    """Five counts. Cheap, and it is what the operations dashboard plots when
    an analyst asks why the mule family's firing rate halved this afternoon."""
    pass


def circuit_open_bits(
    # Supplied as a params bundle field, swapped by the fire-rate monitor.
    # A rule exceeding its max fire rate in a 5-minute window self-demotes
    # (spec §6.1). That is outcome-affecting, so it is recorded on every event
    # assessed after it — which it is, because it is an input to applicability
    # and applicability's inputs are all recorded.
    #
    # Demotion masks a rule OUT of the live firing set. It does NOT move the
    # rule into the shadow population: shadow membership is structural and a
    # runtime demotion cannot cross that line. A demoted rule still evaluates,
    # in the live kernel, into `demoted_fire_bits` — so the owner can see what
    # it would have done while it was tripped.
    demoted: tuple[int, ...] = param((0,) * 10, unit="bitset"),
) -> tuple[int, ...]:
    """Rules currently tripped by their own fire-rate circuit breaker."""
    pass


def circuit_trip_attribution(
    circuit_open_bits: tuple[int, ...],
    applied_adjustment_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Per tripped rule: whether an overlay was in force on it when it tripped.

    "A rule tripping while an overlay is in force on it must attribute the trip
    correctly — the overlay is the likelier culprit and demoting the rule may be
    the wrong remedy." (spec §6.1) The page names the overlay, not the analyst.
    """
    pass


Applicability = module(
    circuit_open_bits,
    applicable_bits,
    applicable_digest,
    applicable_counts_by_family,
    circuit_trip_attribution,
    name="applicability",
    taps=["applicable_digest", "applicable_counts_by_family"],
)
