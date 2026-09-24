"""Event normalization and decision recording (01 §5.1–5.6)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, date
from typing import Optional, Any


@dataclass
class NormalisedEvent:
    """Normalised event after admission and validation (01 §5.1)."""
    event_id: int                          # Globally unique
    event_type_code: int                   # §4.1
    event_timestamp: datetime
    assessment_timestamp: datetime
    client_id: int
    channel_code: int
    product_code: int

    # Common fields (41 across all types)
    amount: Optional[float] = None
    currency_code: str = "ZAR"
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    beneficiary_id: Optional[int] = None

    # Enrichment (stubbed per SCOPE)
    beneficiary_age_hours: Optional[float] = None
    beneficiary_on_mule_list: bool = False
    device_reputation_band: Optional[int] = None
    device_change_hours: Optional[float] = None
    sim_change_hours: Optional[float] = None
    velocity_1min_count: int = 0
    velocity_1min_sum: float = 0.0
    velocity_10min_count: int = 0
    velocity_10min_sum: float = 0.0
    velocity_1h_count: int = 0
    velocity_24h_count: int = 0
    model_score: Optional[float] = None

    # Degradation tracking (01 §5.6)
    enrichment_degradation_code: int = 0  # Bitset
    degraded_mode_code: int = 0          # 0=normal, 1=reduced, 2=restricted, 3=fail_closed

    # Hard blocks
    hard_block_code: Optional[int] = None  # Reason for forced action
    hard_block_action: Optional[int] = None  # Action forced by block


@dataclass
class DecisionRecord:
    """Complete decision record (01 §5.15, 09 §5.15)."""
    # Event and timing
    event_id: int
    decision_id: str                       # Stable identifier (09 §5.15 item 1)
    event_timestamp: datetime
    assessment_timestamp: datetime
    decision_date: date

    # Rule set versioning
    rule_set_version: int
    applied_rule_ids: list[str] = field(default_factory=list)  # Which rules applied
    fired_rule_ids: list[str] = field(default_factory=list)
    fired_on_overlay_ids: list[str] = field(default_factory=list)
    shadow_fired_rule_ids: list[str] = field(default_factory=list)

    # Overlay stack (09 §5.15 item 7)
    adjustment_stack_version: int = 0
    applied_adjustment_ids: list[str] = field(default_factory=list)
    overlay_effects: dict[str, Any] = field(default_factory=dict)
    unadjusted_values: dict[str, Any] = field(default_factory=dict)

    # Action resolution (01 §5.12)
    action_code: int = 10  # ActionCode value
    action_source_rule_id: str = ""
    primary_reason_code: int = 0
    decline_reason_codes: list[int] = field(default_factory=list)
    counterfactual_action: int = 10
    governance_exception: bool = False

    # Evidence (09 §5.15)
    enrichment_degradation_code: int = 0
    degraded_mode_code: int = 0
    normalised_event: Optional[NormalisedEvent] = None
    feature_values: dict[str, float] = field(default_factory=dict)
    velocity_values: dict[str, Any] = field(default_factory=dict)
    model_score_used: Optional[float] = None
    table_versions: dict[str, int] = field(default_factory=dict)  # Table version attribution

    # Evaluation details (09 §5.15 item 14)
    rule_evaluations: list[dict[str, Any]] = field(default_factory=list)

    # Compliance fields
    _decision_id_pii: bool = False  # Marker for PII classification
    _decision_id_unadjusted: bool = False  # Marker for unadjusted values

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON emission."""
        d = asdict(self)
        # Convert enum-like values to their int values
        d["action_code"] = int(d["action_code"]) if isinstance(d["action_code"], int) else d["action_code"]
        # Handle normalised_event separately
        if self.normalised_event:
            d["normalised_event"] = asdict(self.normalised_event)
        return d


def normalise_event(
    raw_event: dict[str, Any],
    assessment_timestamp: datetime
) -> tuple[NormalisedEvent, Optional[str]]:
    """
    Normalise raw event to standard form (01 §5.1).
    Implements 09 §5.15 item 6: inputs captured as received.
    Returns (normalised_event, validation_error_if_any).
    """
    # Minimal validation
    event_id = raw_event.get("event_id")
    event_type_code = raw_event.get("event_type_code")
    event_timestamp_iso = raw_event.get("event_timestamp")
    client_id = raw_event.get("client_id")

    if not all([event_id, event_type_code, event_timestamp_iso, client_id]):
        return None, "Missing mandatory fields"

    # Parse timestamp
    try:
        event_timestamp = datetime.fromisoformat(event_timestamp_iso)
    except (ValueError, TypeError):
        return None, f"Invalid event_timestamp: {event_timestamp_iso}"

    # Create normalised event
    normalised = NormalisedEvent(
        event_id=event_id,
        event_type_code=event_type_code,
        event_timestamp=event_timestamp,
        assessment_timestamp=assessment_timestamp,
        client_id=client_id,
        channel_code=raw_event.get("channel_code", 0),
        product_code=raw_event.get("product_code", 0),
        amount=raw_event.get("amount"),
        currency_code=raw_event.get("currency_code", "ZAR"),
        device_id=raw_event.get("device_id"),
        session_id=raw_event.get("session_id"),
        beneficiary_id=raw_event.get("beneficiary_id"),
        beneficiary_age_hours=raw_event.get("beneficiary_age_hours"),
        beneficiary_on_mule_list=raw_event.get("beneficiary_on_mule_list", False),
        device_reputation_band=raw_event.get("device_reputation_band"),
        device_change_hours=raw_event.get("device_change_hours"),
        sim_change_hours=raw_event.get("sim_change_hours"),
        velocity_1min_count=raw_event.get("velocity_1min_count", 0),
        velocity_1min_sum=raw_event.get("velocity_1min_sum", 0.0),
        velocity_10min_count=raw_event.get("velocity_10min_count", 0),
        velocity_10min_sum=raw_event.get("velocity_10min_sum", 0.0),
        velocity_1h_count=raw_event.get("velocity_1h_count", 0),
        velocity_24h_count=raw_event.get("velocity_24h_count", 0),
        model_score=raw_event.get("model_score"),
    )

    return normalised, None
