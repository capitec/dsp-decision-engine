"""The instant-payment fraud decision: `build` receives `rule_set.json` and `client_wording.json`."""
from __future__ import annotations

from datetime import date

from decider import DecisionTableConfig, dag, flow
from credit_core import consent
from credit_core.dates import EffectiveDatedSet, EffectiveVersion

from fraud_interdiction import action, enrichment, hard_blocks, overlays, wording
from fraud_interdiction.rules import RuleSet

RULE_SET_VERSIONS = EffectiveDatedSet("fraud.rule_set", [
    EffectiveVersion("RS-2025.11", date(2025, 11, 1), date(2026, 1, 1)),
    EffectiveVersion("RS-2026.01", date(2026, 1, 1)),
])


def build(rule_set: RuleSet, client_wording: DecisionTableConfig):
    return dag(
        RULE_SET_VERSIONS.resolver_step(output="rule_set_version"),
        enrichment.velocity_completeness_band, enrichment.enrichment_degradation_code,
        enrichment.degraded_mode_code, enrichment.late_arrival_flag,
        hard_blocks.hard_block_step,
        overlays.adjustment_stack_enabled, overlays.multiplier_step,
        rule_set,
        action.resolve_action_step(rule_set.rules),
        consent.consent_verdict_step, consent.channel_permitted_step,
        # The table writes the wording key and consent may replace it: a later write wins, so a flow.
        flow(client_wording, wording.client_wording_key, name="wording").emit("client_wording_key", "client_visible"),
        name="fraud_interdiction",
    ).emit(
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
