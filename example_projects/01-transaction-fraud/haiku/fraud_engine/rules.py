"""Fraud rule definitions and evaluation (01 §5.10–5.12)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Callable, Optional


class Family(Enum):
    """Rule family classifications (01 §6.1)."""
    CF = "card_fraud"           # Card fraud (214 live, 31 shadow)
    AT = "account_takeover"     # Account takeover (112 live, 22 shadow)
    MS = "mule_scam"            # Mule and scam (96 live, 40 shadow)
    FP = "first_party_fraud"    # First-party fraud (41 live, 9 shadow)
    AA = "aml_adjacent"         # AML-adjacent (58 live, 12 shadow)


class RuleStatus(Enum):
    """Rule lifecycle status (01 §6.1)."""
    LIVE = "live"
    SHADOW = "shadow"
    RETIRED = "retired"


class ActionCode(Enum):
    """Action outcomes ordered by severity (01 §5.12)."""
    FREEZE_ACCOUNT = 70
    BLOCK_CHANNEL = 60
    DECLINE = 50
    HOLD_FOR_REVIEW = 40
    STEP_UP = 30
    MONITOR = 20
    ALLOW = 10


@dataclass
class Rule:
    """Complete rule definition (01 §6.1)."""
    rule_id: str               # Immutable identifier (e.g. "CF-0412")
    family: Family
    description: str
    severity: int              # 1–5 (higher = more severe)
    action: ActionCode
    priority: int              # 0–999 within action (for tie-breaking)
    status: RuleStatus
    effective_from: date
    effective_to: Optional[date]  # None if open-ended; retired when in past
    event_types: set[int]      # Applicable event type codes (e.g. {210})
    segments: Optional[set[str]]  # Applicable segment ids, or None for "all"
    predicates: list[str]      # Rule shape: predicate names this rule checks
    thresholds: dict[str, float]  # Tunable thresholds per predicate
    overlay_exempt: bool = False   # If True, no threshold overlay allowed
    critical: bool = False     # If fired, cannot be suppressed by lower-severity rules
    suppressible: bool = True  # If True, can be suspended by degraded mode
    reason_code: int = 0       # Into core.reason_codes registry
    max_fire_rate: float = 100.0  # Max % of applicable events per 5 min
    queue_code: Optional[int] = None  # For hold_for_review actions
    challenge_type: Optional[str] = None  # For step_up actions

    # Evidence tracking
    rule_version: int = 1      # Increments on any change
    author: Optional[str] = None
    owner: Optional[str] = None
    approver: Optional[str] = None
    approval_timestamp: Optional[date] = None

    def applies_at(self, decision_date: date) -> bool:
        """Check if rule is in effect at decision_date (09 §5.15 item 3)."""
        if self.status == RuleStatus.RETIRED:
            return False
        if decision_date < self.effective_from:
            return False
        if self.effective_to and decision_date > self.effective_to:
            return False
        return True

    def applicable_to(self, event_type: int, segments: Optional[set[str]]) -> bool:
        """Check if rule applies to event type and segments (01 §5.8)."""
        # Event type check
        if event_type not in self.event_types:
            return False
        # Segment check: None means "all"
        if self.segments is None:
            return True
        if segments is None:
            return False
        return bool(self.segments & segments)


@dataclass
class FiredRule:
    """A rule that evaluated true (01 §5.10)."""
    rule: Rule
    feature_values: dict[str, float]  # Values of tested features
    fired_on_base: bool               # True if fired on base thresholds
    fired_on_overlay: bool            # True if fired only because overlay moved threshold
    evaluation_order: int             # For deterministic ordering


@dataclass
class RuleSet:
    """A versioned set of rules (01 §5.8)."""
    version: int
    decision_date: date
    live_rules: list[Rule]
    shadow_rules: list[Rule]
    retired_rules: list[Rule] = field(default_factory=list)

    def get_applicable_live(
        self,
        event_type: int,
        segments: Optional[set[str]],
        degraded_suspend: Optional[set[str]] = None
    ) -> list[Rule]:
        """Get live rules applicable to this event (01 §5.8)."""
        applicable = [
            r for r in self.live_rules
            if r.applies_at(self.decision_date) and r.applicable_to(event_type, segments)
        ]
        # Apply degraded-mode suspensions
        if degraded_suspend:
            applicable = [r for r in applicable if r.rule_id not in degraded_suspend]
        return applicable

    def get_applicable_shadow(
        self,
        event_type: int,
        segments: Optional[set[str]],
        degraded_activate: Optional[set[str]] = None
    ) -> list[Rule]:
        """Get shadow rules applicable to this event (01 §5.11)."""
        applicable = [
            r for r in self.shadow_rules
            if r.applies_at(self.decision_date) and r.applicable_to(event_type, segments)
        ]
        # Apply degraded-mode activations
        if degraded_activate:
            applicable.extend([
                r for r in self.retired_rules
                if r.rule_id in degraded_activate and
                   r.applies_at(self.decision_date) and
                   r.applicable_to(event_type, segments)
            ])
        return applicable


def generate_ruleset(decision_date: date, rule_set_version: int = 1) -> RuleSet:
    """
    Generate a synthetic rule set of ~520 live and 100+ shadow rules.
    Keeps rule count at volume per SCOPE.
    """
    rules_live = []
    rules_shadow = []

    # Family distributions: CF 214, AT 112, MS 96, FP 41, AA 58
    families = [
        (Family.CF, 214, 31),   # (family, live_count, shadow_count)
        (Family.AT, 112, 22),
        (Family.MS, 96, 40),
        (Family.FP, 41, 9),
        (Family.AA, 58, 12)
    ]

    rule_counter = 1

    for family, live_count, shadow_count in families:
        # Generate live rules for this family
        for i in range(live_count):
            rule_id = f"{family.name}-{rule_counter:04d}"
            rule_counter += 1

            # Vary predicates and attributes
            severity = ((i % 5) + 1)  # 1-5
            action_idx = i % 7
            action_codes = [ActionCode.ALLOW, ActionCode.MONITOR, ActionCode.STEP_UP,
                            ActionCode.HOLD_FOR_REVIEW, ActionCode.DECLINE,
                            ActionCode.BLOCK_CHANNEL, ActionCode.FREEZE_ACCOUNT]
            action = action_codes[action_idx]

            rule = Rule(
                rule_id=rule_id,
                family=family,
                description=f"{family.name} rule {i}: {rule_id}",
                severity=severity,
                action=action,
                priority=i % 100,
                status=RuleStatus.LIVE,
                effective_from=decision_date,
                effective_to=None,
                event_types={210},  # Instant payment only (per SCOPE)
                segments=None,      # All segments
                predicates=["amount", "beneficiary_age_hours", "device_change_hours"],
                thresholds={
                    "amount": 5000.0 + (i * 100),
                    "beneficiary_age_hours": 2.0 + (i % 72),
                    "device_change_hours": 72.0 - (i % 48)
                },
                overlay_exempt=i % 50 == 0,  # ~2% overlay-exempt
                critical=i % 100 == 0,        # ~1% critical
                suppressible=(i % 10) != 0,   # ~90% suppressible
                reason_code=2001 + (i % 10),  # Map to reason codes
                max_fire_rate=5.0 + (i % 95),
                queue_code=i % 11 if action == ActionCode.HOLD_FOR_REVIEW else None,
                challenge_type="sms" if action == ActionCode.STEP_UP else None,
                author=f"analyst_{i % 5}",
                owner=f"owner_{family.name}",
                approver=f"approver_{i % 3}",
                approval_timestamp=decision_date
            )
            rules_live.append(rule)

        # Generate shadow rules for this family
        for i in range(shadow_count):
            rule_id = f"{family.name}S-{rule_counter:04d}"
            rule_counter += 1

            rule = Rule(
                rule_id=rule_id,
                family=family,
                description=f"{family.name} shadow rule {i}: {rule_id}",
                severity=((i % 5) + 1),
                action=ActionCode.DECLINE,  # Doesn't matter for shadow
                priority=i % 100,
                status=RuleStatus.SHADOW,
                effective_from=decision_date,
                effective_to=None,
                event_types={210},
                segments=None,
                predicates=["amount", "velocity_1min_count"],
                thresholds={
                    "amount": 2000.0 + (i * 50),
                    "velocity_1min_count": 3.0 + (i % 5)
                },
                suppressible=True,
                reason_code=2002,  # Refer for review
                author=f"analyst_{(i + 2) % 5}",
                owner=f"owner_{family.name}",
            )
            rules_shadow.append(rule)

    return RuleSet(
        version=rule_set_version,
        decision_date=decision_date,
        live_rules=rules_live,
        shadow_rules=rules_shadow
    )
