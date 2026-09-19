"""The flat rule set — 521 live, 114 shadow, 1 360 retired.

This file is code and changes about twice a year. The 635 rules are documents
and change several times a day. That split is the governance model, and it is
the reason a `ruleset` declares its *interface* in Python and takes its *body*
from a validated document (doc 08 §3) rather than being a pile of `@step`
functions.

Three declarations, not one. The population is a **construction-time** property,
not a runtime flag, and that is what makes shadow isolation structural:

    LiveRules    -> live_fire_bits, live_base_fire_bits, unevaluable_bits
    ShadowRules  -> shadow_fire_bits            (different names, on purpose)
    RetiredRules -> retired_fire_bits           (replay generations only)

Nothing downstream of `Resolve` can be read by `Resolve`, and `Resolve`'s
`reads` is a Python list in `decision/resolve.py` that does not contain
`shadow_fire_bits`. A shadow rule influencing an outcome would require an
engineer to edit that list, in a code review, against a CI lineage assertion
that fails. See `tests/test_shadow_isolation.py` and FRAMEWORK-DEMANDS.md #9.

--------------------------------------------------------------------------
Why not 635 `@step` functions
--------------------------------------------------------------------------
Doc 03 §5.3 says a bare function is a pipeline element and doc 03 §1.1 says one
rule should cost one artefact. Applied literally here that is 635 Python
functions, each carrying 18 fields of governance metadata in a decorator, edited
by analysts, deployed by a commit. Four things break:

  * analysts are not engineers and the path is "under 10 minutes including
    approval" (spec §10.1);
  * a threshold change would be a code change, when 85% of changes are
    threshold-only (spec §6.2);
  * effective dating, four-eyes approval and retirement would live in git
    history rather than in the artefact, and "prove MS-0117 stopped affecting
    outcomes on 30 June" would be archaeology;
  * `@rule(...)` with 18 arguments is not an improvement on YAML. It is YAML
    with worse diffs.

So: doc 03's authoring surface is right for the ~30 hand-written enrichment and
decision steps in this project, and wrong for the 635 rules. Both live here.
"""

from __future__ import annotations

from decider2 import ruleset, stable_blocks
from decider2.governance import ApprovalBlock, EffectiveWindow

from fraud_interdiction.contracts import FEATURE_VECTOR
from fraud_interdiction.features import derived  # noqa: F401 — registers feat:* ids

# --- families: owners, precedence, severity floors -------------------------
# Family precedence is the fourth tie-break key (spec §5.12.5) and is owned by
# the Head of Fraud. It is a value: reordering it is a params swap, not a
# compile, which is what lets spec §11.12's cross-family collision be resolved
# "without either owner editing the other's rules".

FAMILIES = ("CF", "AT", "MS", "FP", "AA")

RULE_ATTRIBUTES = (
    # read by the resolver, never by a predicate. All values — a change to any
    # of them is a bundle swap, no compile.
    "action_code", "severity", "priority", "family_index", "reason_code",
    "critical", "critical_allow_families", "suppressible", "overlay_exempt",
    "queue_code", "sla_tier", "challenge_type", "max_fire_rate_pct",
    "effective_from", "effective_to", "event_type_mask", "segment_mask",
    "circuit_open",
)

_COMMON = dict(
    reads=FEATURE_VECTOR,
    attributes=RULE_ATTRIBUTES,
    features="feat:*",                      # registered derived features, by id
    # Compilation unit != authoring unit (doc 02 §1.1). One document of 521
    # rules compiles to 16 kernels, and a rule's block is a stable hash of its
    # rule_id — NOT its position. Adding the 522nd rule recompiles exactly one
    # block of ~33 rules (~2 s); the other 15 come from the numba cache
    # byte-identical. Position-based blocking would have shifted every rule
    # after the insertion point and recompiled all 16. Spec Q15.
    blocks=stable_blocks(by="rule_id", count=16),
    # Two evaluations per rule, one kernel: the effective-threshold pass and,
    # for rules in an overlay's scope only, the base-threshold pass. See
    # decision/overlays.py.
    emit_base_pass=True,
    # The compile key is this projection of the document, not the whole file.
    # A typo fix in a description must not cost a 40-second compile and an
    # approval cycle. FRAMEWORK-DEMANDS.md #3.
    shape_projection=("evaluate", "applies_to.event_types", "on_absent", "tunables.*.unit"),
)

LiveRules = ruleset(
    name="live_rules",
    population="live",
    writes=["live_fire_bits", "live_base_fire_bits", "unevaluable_bits",
            "demoted_fire_bits", "applicable_bits", "applicable_digest"],
    **_COMMON,
)

ShadowRules = ruleset(
    name="shadow_rules",
    population="shadow",
    writes=["shadow_fire_bits", "shadow_unevaluable_bits", "shadow_applicable_bits"],
    **_COMMON,
)

RetiredRules = ruleset(
    name="retired_rules",
    population="retired",
    # Compiled into replay generations only. A retired rule is date-gated out of
    # effect the instant its effective_to passes (free, a value change) and
    # physically dropped from the live kernel at the next housekeeping
    # generation. Replay of an event from 2025 pins the 2025 generation, whose
    # kernel still contains it. Spec §9.2, §11.11.
    writes=["retired_fire_bits"],
    **_COMMON,
)


def validate_documents(*documents) -> None:
    """Every static check that must pass before a rule set is compiled.

    Run in the analyst's browser-side validation, in CI, and again at
    `stage()`. The same function, so a rule that passes in the UI cannot fail
    at deployment — which is what makes "under 10 minutes including approval"
    a schedule rather than a hope.

      * every referenced feature is in the catalogue or registered as feat:*
      * every referenced feature exists for every event type in the rule's
        scope, over the whole of the rule's effective window (§4.1, §11.6)
      * every tunable referenced by a predicate is declared in `tunables`
      * `critical: true` carries two named approvers, not four-eyes
      * `overlay_exempt: true` on every rule whose family is AA or whose
        reason code is in the regulatory subset — asserted, not requested
      * `action: hold_for_review` carries a queue and an SLA tier
      * `action: step_up` carries a challenge type that the challenge matrix
        marks permitted-somewhere (spec §11.13: "not permitted" and "not
        configured" are different cells)
      * no two live rules share a rule_id, ever, including against the retired
        register — ids are never reused
      * the transitive feature closure of every rule fits inside `reads`
        (doc 08 §3.2's upper-bound rule)
    """
    pass
