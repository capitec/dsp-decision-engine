"""5.9 — adjustment overlay resolution.

This is the module that doc 03 cannot express, and the reason is one sentence:

    An overlay's scope is declared over **segments, channels and event types**,
    so the effective value of a threshold varies **per record**.

Doc 02 §4 is explicit that params do not vary by record — that is the entire
basis of "a params bundle's type is fixed, so retuning never recompiles". A
per-record params bundle would either recompile per record or stop being params.
Neither is acceptable. See FRAMEWORK-DEMANDS.md #10 and #11.

--------------------------------------------------------------------------
The shape that works: a gain vector, not a parameter set
--------------------------------------------------------------------------
Overlays do not name individual thresholds. They name a **class** of threshold:
"every tunable in family MS with unit amount_zar_cents, x0.70". That is why
`unit:` is a required field on every tunable in a rule document — it is the
overlay's join key, and without it a sensitivity dial has nothing to grip.

So resolution produces a tiny dense array:

    gain[family_index][unit_index]     5 x 8 = 40 float64 = 320 bytes

built once per event by iterating the <=40 overlays in the register, not by
touching 1 900 thresholds. Cost is O(overlays), not O(overlays x params), which
is spec §8's "resolving an overlay stack of 40 entries must not be a per-rule
cost", met by construction rather than by optimisation.

A rule's emitted predicate reads:

    amount_zar_cents > tunables.min_amount * gain[MS][amount_zar_cents]

one multiply. `overlay_exempt` rules emit the same line with the gain term
absent — decided at codegen, so an exempt rule has no runtime path by which a
dial could reach it. That is stronger than checking a flag.

--------------------------------------------------------------------------
Rule-scoped overlays, and the one genuinely awkward part
--------------------------------------------------------------------------
Scope-restriction and severity-shift overlays can name individual rules (spec
§6.5: "suspend three named rules for the business-linked segment"). Those need a
per-rule vector, 635 wide, which cannot be rebuilt per event at 12 000/s.

It is rebuilt per **scope class** instead — the distinct tuple of
(event_type_code, channel_code, segment_bits & overlay_relevant_segment_mask).
Live overlays mention few segments, so realised scope classes are in the low
hundreds. `keyed_materialisation` memoises the built vector on that key: first
event of a class pays ~635 writes, every other event is a pointer read.

It is deterministic — a pure function of the key — so a cache hit and a cache
miss cannot produce different answers, which is what keeps spec §8's
"determinism under concurrency" intact. It is also the first thing I would put
under load test, because it is the one piece of this design whose correctness
argument is a cache-invalidation argument.
"""

from __future__ import annotations

from decider2 import keyed_materialisation, module, param
from decider2.governance import AdjustmentRegister

UNITS = ("amount_zar_cents", "hours", "count", "ratio", "band", "score",
         "minutes", "days")
KINDS = ("sensitivity_dial", "threshold_multiplier", "model_cutoff_shift",
         "severity_shift", "action_escalation", "scope_restriction", "queue_reroute")

# The register is a governed document. Its validators are not advisory:
#   * effective_to is REQUIRED and (effective_to - effective_from) <= 90 days
#   * review_date is REQUIRED and <= effective_to
#   * approval carries two distinct named people and a backtest_ref
#   * compose_at is an explicit integer; two overlays may not share one
#   * scope must name at least one of families / rule_ids / event_types
#   * an overlay whose scope resolves to the empty set at validation is an
#     ERROR, not a no-op (spec §5.9: "applying it outside that scope is an
#     error, not a silent no-op, and must fail loudly")
REGISTER = AdjustmentRegister(
    name="fraud_overlays",
    kinds=KINDS,
    units=UNITS,
    max_duration_days=90,
    require_review_date=True,
    require_approvers=2,
    require_backtest_ref=True,
    exempt_flag="overlay_exempt",
)


def adjustment_stack_version(event_timestamp: int) -> int:
    """The stack in force at `event_timestamp`, not at assessment time.

    NOTE, and this is a discrepancy in the spec worth naming: §5.9 says the
    overlay stack is resolved "at event_timestamp", while a rule's thresholds
    are whatever the deployed params generation holds at *assessment* time. For
    a SIM change that arrives three hours late those are different instants.
    Resolved here by recording both instants and both versions, and by making
    replay read the recorded generation rather than re-deriving one. See
    README "Two clocks".
    """
    pass


def applied_adjustment_ids(
    adjustment_stack_version: int,
    event_type_code: int,
    channel_code: int,
    client_segment_bits: int,
) -> tuple[int, ...]:
    """Overlays in force AND in scope, in composition order (`compose_at`)."""
    pass


def out_of_scope_adjustment_ids(
    adjustment_stack_version: int,
    applied_adjustment_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """Overlays in force but NOT in scope for this event.

    Recorded deliberately. "This is how an analyst confirms scoping does what
    she intended" (spec §5.9) — without it, a mis-scoped overlay is invisible
    because its symptom is an absence.
    """
    pass


def overlay_gain(
    applied_adjustment_ids: tuple[int, ...],
) -> tuple[float, ...]:
    """The 5 x 8 gain array, flattened. Composition is multiplicative in
    `compose_at` order, and the order is declared, never emergent from load
    order or insertion time (spec §6.5.3)."""
    pass


@keyed_materialisation(
    key=("event_type_code", "channel_code", "overlay_relevant_segment_bits"),
    capacity=4096,
)
def rule_scoped_overlay_vectors(
    event_type_code: int,
    channel_code: int,
    overlay_relevant_segment_bits: int,
    adjustment_stack_version: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """(scope_restriction_bits, severity_shift_by_rule, action_escalation_by_rule).

    Memoised on the scope class; invalidated wholesale when
    `adjustment_stack_version` changes, which happens on the order of once an
    hour and never mid-batch (the generation pointer is read once per
    invocation — doc 08 §4.1).
    """
    pass


def overlay_relevant_segment_bits(
    client_segment_bits: int,
    relevant_mask: int = param(0, unit="bitset"),
) -> int:
    """The client's segment bits, masked down to the segments any live overlay
    actually mentions. Keeps the scope-class cardinality bounded: 46 segments
    would be 2^46 classes, the ~6 segments overlays currently name are 64."""
    pass


def overlay_effect_record(
    applied_adjustment_ids: tuple[int, ...],
    overlay_gain: tuple[float, ...],
) -> tuple[tuple[int, float, float], ...]:
    """Per modified value: (tunable_id, base_value, effective_value).

    "Per modified value, base and effective side by side" (spec §5.9). This is
    the field the monthly fraud forum reads and the field a regulator asks for
    once overlays are known to exist. It is derived, not stored per event: the
    record holds the stack version and the gain array, and the pairs are
    reconstructed against the pinned rule set. 320 bytes on the record, a full
    table on the screen.
    """
    pass


OverlayStack = module(
    adjustment_stack_version,
    overlay_relevant_segment_bits,
    applied_adjustment_ids,
    out_of_scope_adjustment_ids,
    overlay_gain,
    rule_scoped_overlay_vectors,
    overlay_effect_record,
    name="overlay_stack",
    taps=["adjustment_stack_version", "applied_adjustment_ids",
          "out_of_scope_adjustment_ids", "overlay_gain"],
)
