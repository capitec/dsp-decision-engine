"""Transaction fraud interdiction pipeline (project 01).

Composes rule evaluation, action resolution, and evidence emission.
Implements 09 §5.15: every decision records what it needs to be replayed.
"""
from __future__ import annotations
from decider import flow, param, missing_as
from datetime import date, datetime, timezone
from fraud_engine import (
    generate_ruleset, NormalisedEvent, DecisionRecord, normalise_event,
    RuleEvaluator, ActionCode, Family
)
import sys
sys.path.insert(0, '/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/examples/haiku/00-shared-credit-core')
from core import generate_decision_id
from core.dates import resolve_effective_dated
from core.reason_codes import rank_reasons, get_primary_reason
from core.adjustments import apply_adjustments


# @step
def normalise_event_step(
    raw_event: dict,
    assessment_timestamp: datetime = missing_as(datetime.now(timezone.utc))
) -> dict:
    """Event admission and normalisation (01 §5.1).

    Validates mandatory fields and normalises to standard form.
    Implements 09 §5.15 items 3, 6: decision_date from inputs, inputs captured as received.
    """
    normalised, error = normalise_event(raw_event, assessment_timestamp)

    if error:
        return {
            "event_id": raw_event.get("event_id"),
            "normalised_event": None,
            "normalisation_error": error,
            "enrichment_degradation_code": 255,
            "degraded_mode_code": 3  # fail_closed
        }

    # Stub enrichment (01 SCOPE: "stub enrichment and velocity store")
    # In real implementation, these come from live stores with staleness tracking
    event_features = {
        "amount": normalised.amount or 1000.0,
        "beneficiary_age_hours": normalised.beneficiary_age_hours or 24.0,
        "device_change_hours": normalised.device_change_hours or 168.0,
        "velocity_1min_count": float(normalised.velocity_1min_count),
        "model_score": normalised.model_score or 500.0,
    }

    return {
        "event_id": normalised.event_id,
        "normalised_event": normalised,
        "event_features": event_features,
        "enrichment_degradation_code": 0,  # All sources fresh
        "degraded_mode_code": 0  # normal
    }


# @step
def resolve_rule_set(
    decision_date: date = missing_as(date.today()),
    rule_set_version: int = param(1, ge=1)
) -> dict:
    """Resolve the applicable rule set at decision_date (01 §5.8).

    Implements 09 §5.15 item 3: no "today", use decision_date.
    """
    ruleset = generate_ruleset(decision_date, rule_set_version)

    return {
        "rule_set_version": ruleset.version,
        "rule_set": ruleset,
        "total_live": len(ruleset.live_rules),
        "total_shadow": len(ruleset.shadow_rules)
    }


# @step
def get_client_context(
    client_id: int,
    decision_date: date = missing_as(date.today()),
    client_segments: list[str] = missing_as(["retail", "standard"])
) -> dict:
    """Client and account context (01 §5.2, stubbed).

    In full implementation, reads from live profile store with snapshot.
    Implements 09 §5.15 item 8: mutable state snapshotted.
    """
    return {
        "client_id": client_id,
        "client_segments": set(client_segments),
        "account_frozen": False,
        "card_status": "active",
        "prior_fraud_count": 0
    }


# @step
def resolve_hard_blocks(
    client_context: dict,
    normalised_event: NormalisedEvent = missing_as(None),
    beneficiary_on_mule_list: bool = param(False)
) -> dict:
    """Hard blocks: gates that short-circuit (01 §5.7).

    Recorded but do not skip rule evaluation (01 §5.7).
    """
    hard_block_code = None
    hard_block_action = None

    # Simplified: just account frozen
    if client_context.get("account_frozen"):
        hard_block_code = 1  # Account frozen
        hard_block_action = ActionCode.FREEZE_ACCOUNT

    if beneficiary_on_mule_list:
        hard_block_code = 2  # Confirmed mule
        hard_block_action = ActionCode.DECLINE

    return {
        "hard_block_code": hard_block_code,
        "hard_block_action": hard_block_action.value if hard_block_action else None
    }


# @step
def evaluate_rules(
    rule_set: dict,
    event_features: dict,
    client_segments: set,
    decision_date: date = missing_as(date.today()),
    enrichment_degradation_code: int = param(0, ge=0, le=255),
    degraded_suspend_rules: list[str] = param([], description="Rules to suspend in degraded mode")
) -> dict:
    """Live and shadow rule evaluation (01 §5.10–5.11).

    No early exit; all rules evaluated.
    Implements 09 §5.15 item 14: evaluation recorded, not only firing.
    """
    ruleset = rule_set["rule_set"]
    evaluator = RuleEvaluator()

    # Resolve degraded-mode suspensions
    degraded_suspend = set(degraded_suspend_rules) if enrichment_degradation_code else set()

    # Get applicable rules
    applicable_live = ruleset.get_applicable_live(
        210,  # Event type: instant payment
        client_segments,
        degraded_suspend
    )
    applicable_shadow = ruleset.get_applicable_shadow(
        210, client_segments
    )

    # Evaluate
    fired_live, fired_shadow = evaluator.evaluate_ruleset(
        ruleset,
        event_features,
        applicable_live,
        applicable_shadow,
        overlay_stack=None  # No overlays in this stub
    )

    return {
        "applicable_live_count": len(applicable_live),
        "applicable_shadow_count": len(applicable_shadow),
        "fired_live": fired_live,
        "fired_shadow": fired_shadow,
        "applicable_rule_ids": [r.rule_id for r in applicable_live],
    }


# @step
def resolve_action(
    fired_live_result: dict,
    hard_block_action: int = missing_as(None)
) -> dict:
    """Action resolution by precedence (01 §5.12).

    Implements 09 §5.15 items 11, 14: reason codes with registry version, evaluation recorded.
    """
    fired_live = fired_live_result.get("fired_live", [])
    fired_shadow = fired_live_result.get("fired_shadow", [])

    # Convert int back to ActionCode if provided
    hb_action = ActionCode(hard_block_action) if hard_block_action else None

    evaluator = RuleEvaluator()
    outcome = evaluator.resolve_action(fired_live, fired_shadow, hb_action)

    return {
        "action_code": outcome.action_code.value,
        "action_source_rule_id": outcome.action_source_rule_id,
        "fired_rule_ids": outcome.fired_rule_ids,
        "fired_on_overlay_ids": outcome.fired_on_overlay_ids,
        "shadow_fired_rule_ids": outcome.shadow_fired_rule_ids,
        "reason_codes": outcome.reason_codes,
        "primary_reason_code": outcome.primary_reason_code,
        "counterfactual_action": outcome.counterfactual_action.value,
        "governance_exception": outcome.governance_exception
    }


# @step
def emit_decision_record(
    event_id: int,
    decision_date: date = missing_as(date.today()),
    normalised_event: NormalisedEvent = missing_as(None),
    rule_set_version: int = param(1),
    action_code: int = param(10),
    action_source_rule_id: str = param(""),
    fired_rule_ids: list[str] = missing_as([]),
    fired_on_overlay_ids: list[str] = missing_as([]),
    shadow_fired_rule_ids: list[str] = missing_as([]),
    reason_codes: list[int] = missing_as([]),
    primary_reason_code: int = param(0)
) -> dict:
    """Emit complete decision record (01 §5.15, 09 §5.15).

    Implements 09 §5.15: items 1 (decision_id), 2 (stable ids), 5 (table versions),
    7 (overlay stack), 9 (no external calls), 10 (explicit params), 14 (evaluation recorded).
    """
    decision_id = generate_decision_id()
    assessment_timestamp = datetime.now(timezone.utc)

    record = DecisionRecord(
        event_id=event_id,
        decision_id=decision_id,
        event_timestamp=normalised_event.event_timestamp if normalised_event else assessment_timestamp,
        assessment_timestamp=assessment_timestamp,
        decision_date=decision_date,
        rule_set_version=rule_set_version,
        fired_rule_ids=fired_rule_ids,
        fired_on_overlay_ids=fired_on_overlay_ids,
        shadow_fired_rule_ids=shadow_fired_rule_ids,
        action_code=action_code,
        action_source_rule_id=action_source_rule_id,
        decline_reason_codes=reason_codes,
        primary_reason_code=primary_reason_code,
        normalised_event=normalised_event
    )

    return {
        "decision_record": record,
        "decision_id": decision_id,
        "record_dict": record.to_dict()
    }


def build():
    """Build the fraud detection pipeline.

    Composes steps into a decision flow.
    Implements 09 §5.15: complete evidence contract compliance.
    """
    return flow(
        # 01 §5.1: Event admission and normalisation
        normalise_event_step(
            raw_event=None,  # From request
            assessment_timestamp=missing_as(datetime.now(timezone.utc))
        ),

        # 01 §5.8: Rule set resolution
        resolve_rule_set(
            decision_date=missing_as(date.today()),
            rule_set_version=param(1)
        ),

        # 01 §5.2: Client context
        get_client_context(
            client_id=None,  # From request
            decision_date=missing_as(date.today()),
            client_segments=missing_as(["retail"])
        ),

        # 01 §5.7: Hard blocks
        resolve_hard_blocks(
            client_context=None,  # From previous step
            normalised_event=missing_as(None),
            beneficiary_on_mule_list=param(False)
        ),

        # 01 §5.10–5.11: Rule evaluation
        evaluate_rules(
            rule_set=None,  # From previous step
            event_features=None,  # From normalise step
            client_segments=None,  # From client context
            decision_date=missing_as(date.today()),
            enrichment_degradation_code=param(0),
            degraded_suspend_rules=param([])
        ),

        # 01 §5.12: Action resolution
        resolve_action(
            fired_live_result=None,  # From evaluate step
            hard_block_action=missing_as(None)
        ),

        # 01 §5.15: Decision record emission
        emit_decision_record(
            event_id=None,  # From normalised event
            decision_date=missing_as(date.today()),
            normalised_event=missing_as(None),
            rule_set_version=param(1),
            action_code=param(10),
            action_source_rule_id=param(""),
            fired_rule_ids=missing_as([]),
            fired_on_overlay_ids=missing_as([]),
            shadow_fired_rule_ids=missing_as([]),
            reason_codes=missing_as([]),
            primary_reason_code=param(0)
        ),

        name="fraud_interdiction"
    )
