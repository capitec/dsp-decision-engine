"""The instant-payment fraud decision: `build` receives one document per argument, each a file in `configs/<version>/`."""
from __future__ import annotations

from decikit import DecisionTableConfig, dag, flow
from credit_core import consent
from credit_core.adjustments import AdjustmentStack
from credit_core.dates import EffectiveVersions
from credit_core.reason_codes import ReasonCodes

from fraud_interdiction import enrichment, hard_blocks, overlays, wording
from fraud_interdiction.action import ActionResolver
from fraud_interdiction.overlays import AmountOverlays
from fraud_interdiction.rules import RuleSet


def build(
    rule_set: RuleSet, client_wording: DecisionTableConfig, rule_set_versions: EffectiveVersions,
    amount_overlays: AmountOverlays, action_escalation: AdjustmentStack, reason_codes: ReasonCodes,
):
    return dag(
        rule_set_versions,
        enrichment.velocity_completeness_band, enrichment.enrichment_degradation_code,
        enrichment.degraded_mode_code, enrichment.late_arrival_flag,
        hard_blocks.hard_block_step,
        overlays.adjustment_stack_enabled, amount_overlays,
        rule_set,
        ActionResolver(rules=rule_set.rules, escalation=action_escalation, reason_codes=reason_codes),
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
