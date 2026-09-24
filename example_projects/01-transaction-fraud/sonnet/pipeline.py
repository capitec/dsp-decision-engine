"""Transaction fraud interdiction: the real-time instant-payment flow (spec 01, SCOPE.md slice).

Event admission/context (§5.1-§5.2) and full enrichment tables (§5.3) are
stubbed per SCOPE.md -- the request already carries resolved feature
values, bands and states, as a real enrichment stage would emit them.
What runs at real depth: the degraded-mode verdict (§5.6), hard blocks
(§5.7), the ~635-rule flat rule set with live/shadow isolation (§5.8,
§5.10-§5.11), the overlay stack (§5.9), action resolution with the
counterfactual (§5.12), and reason/wording resolution (§5.14).

`build()` takes its three rule-set documents as `ConfigurableStep`s, in
the shape `decider template` produces (`def build(tree)` loads
`configs/<version>/tree.json`) -- here `live_rules`, `shadow_rules` and
`overlay_base_rules`, one file each.
"""
from __future__ import annotations

from datetime import date

from decider import dag, flow
from decider.steps.trees import TreeConfig  # side effect: registers the "tree" ConfigurableStep type
                                             # (decider build's `def build(x: ConfigurableStep)` loader
                                             # resolves the document's own `type` tag from the registry
                                             # of *imported* ConfigurableStep subclasses -- an untyped
                                             # `ConfigurableStep` parameter, or the same annotation
                                             # without this import anywhere in the pipeline module,
                                             # fails with `RegistryError: 'tree' is not a registered
                                             # ConfigurableStep type` because nothing ever imported
                                             # decider.steps.trees. See NOTES.md "Framework friction".

from credit_core import consent
from credit_core.dates import EffectiveDatedSet, EffectiveVersion

from fraud_interdiction import action_resolution, features, firing, hard_blocks, overlays, reasons
from fraud_interdiction.rules import build_rule_catalog

# §4.4 rule_set_version: an effective-dated artefact like any other in the library (`core.dates`),
# resolved against `decision_date`, never "today" (09 §5.15 item 4).
RULE_SET_VERSIONS = EffectiveDatedSet("fraud.rule_set", [
    EffectiveVersion("RS-2025.11", date(2025, 11, 1), date(2026, 1, 1)),
    EffectiveVersion("RS-2026.01", date(2026, 1, 1), None),
])


def build(live_rules: TreeConfig, shadow_rules: TreeConfig, overlay_base_rules: TreeConfig):
    catalog = build_rule_catalog()  # same fixed seed as generate_configs.py -- see its docstring

    rule_set_version_step = RULE_SET_VERSIONS.resolver_step(output="rule_set_version")

    return dag(
        rule_set_version_step,

        features.velocity_completeness_band_step,
        features.enrichment_degradation_code_step,
        features.degraded_mode_code_step,
        features.event_timestamp_flag_step,

        hard_blocks.hard_block_code_step,

        overlays.adjustment_stack_enabled_step,
        overlays.mule_scam_amount_multiplier_step,

        live_rules, shadow_rules, overlay_base_rules,

        firing.build_resolve_firing_step(catalog),
        # `resolve_action` writes raw `decline_reason_codes`; the registry's own step ranks it under
        # the same name (its documented contract) -- same same-name waterfall `flow` requires (see
        # 00 pipeline.py's `_reasons_unit`).
        flow(action_resolution.build_resolve_action_step(catalog), reasons.REASON_REGISTRY.resolve_step(),
             name="action_and_reasons"),
        consent.consent_verdict_step, consent.channel_permitted_step,
        reasons.client_wording_step,

        name="fraud_interdiction",
    ).emit(
        "decision_date", "channel_code",
        "rule_set_version", "adjustment_set_id", "applied_adjustment_ids",
        "mule_scam_amount_multiplier", "mule_scam_amount_multiplier_unadjusted",
        "velocity_completeness_band", "enrichment_degradation_code", "degraded_mode_code", "late_arrival_flag",
        "hard_block_code", "hard_block_forced_action",
        "fired_rule_ids", "unevaluable_rule_ids", "fired_on_overlay_ids", "counterfactual_fired_rule_ids",
        "shadow_fired_rule_ids",
        "action_code", "action_source_rule_id", "governance_exception", "counterfactual_action_code",
        "decline_reason_codes", "primary_reason_code", "reason_registry_version",
        "client_wording_key", "client_visible",
    )
