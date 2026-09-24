"""Rule evaluation and action resolution (01 §5.10–5.12)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from .rules import Rule, RuleSet, RuleStatus, ActionCode, FiredRule, Family


@dataclass
class OverlayApplication:
    """How an overlay is applied to a rule (01 §5.9)."""
    overlay_id: str
    overlay_name: str
    kind: str                    # "sensitivity_dial", "threshold_multiplier", "action_escalation"
    base_values: dict[str, float]  # Original threshold values
    overlay_values: dict[str, float]  # Adjusted threshold values
    multiplier: float = 1.0     # For dials and multipliers


@dataclass
class EvaluationResult:
    """Result of evaluating a rule (01 §5.10)."""
    rule: Rule
    evaluated: bool             # Was it in the applicable set?
    fired: bool                 # Did it fire?
    base_fired: bool            # Would it fire without overlay?
    feature_values: dict[str, float]
    effective_thresholds: dict[str, float]  # After overlay
    overlay_applied: Optional[str] = None   # Overlay id if threshold changed
    reason: str = ""            # Why didn't it fire (if applicable)


@dataclass
class DecisionOutcome:
    """Complete decision after rule evaluation (01 §5.12)."""
    action_code: ActionCode
    action_source_rule_id: str  # Which rule determined the action
    fired_rule_ids: list[str]
    fired_on_overlay_ids: list[str]  # Subset of fired_rule_ids
    shadow_fired_rule_ids: list[str]
    reason_codes: list[int]     # All firing rules' reasons, ranked
    primary_reason_code: int
    governance_exception: bool  # Two critical rules with conflicting actions
    counterfactual_action: ActionCode  # What base rules would have decided
    evaluation_details: list[EvaluationResult] = field(default_factory=list)


class RuleEvaluator:
    """Evaluates rules and resolves actions (01 §5.10–5.12)."""

    def evaluate_rule(
        self,
        rule: Rule,
        event_features: dict[str, float],
        effective_thresholds: Optional[dict[str, float]] = None,
        base_thresholds: Optional[dict[str, float]] = None
    ) -> tuple[bool, bool, dict[str, float]]:
        """
        Evaluate a single rule against features.
        Returns (fired_on_effective, fired_on_base, feature_values).
        Implements 09 §5.15 item 14: evaluation recorded, not only firing.
        """
        thresholds = effective_thresholds or rule.thresholds

        # Extract feature values for predicates
        feature_values = {}
        all_match = True

        for predicate in rule.predicates:
            if predicate not in event_features:
                # Feature missing: rule is unevaluable
                return False, False, {}

            feature_value = event_features[predicate]
            feature_values[predicate] = feature_value
            threshold = thresholds.get(predicate)

            # Simple threshold comparison (predicates assume feature > threshold means true)
            if threshold is not None and feature_value <= threshold:
                all_match = False

        fired_effective = all_match
        fired_base = fired_effective

        # If different thresholds requested, check base thresholds too
        if base_thresholds and base_thresholds != effective_thresholds:
            all_base_match = True
            for predicate in rule.predicates:
                feature_value = feature_values.get(predicate)
                if feature_value is not None:
                    base_threshold = base_thresholds.get(predicate)
                    if base_threshold is not None and feature_value <= base_threshold:
                        all_base_match = False
            fired_base = all_base_match

        return fired_effective, fired_base, feature_values

    def evaluate_ruleset(
        self,
        ruleset: RuleSet,
        event_features: dict[str, float],
        applicable_live_rules: list[Rule],
        applicable_shadow_rules: list[Rule],
        overlay_stack: Optional[dict[str, OverlayApplication]] = None
    ) -> tuple[list[FiredRule], list[FiredRule]]:
        """
        Evaluate all applicable rules.
        Implements 01 §5.10–5.11: no early exit, all rules evaluated.
        Returns (fired_live, fired_shadow).
        """
        overlay_stack = overlay_stack or {}
        fired_live = []
        fired_shadow = []

        # Evaluate live rules
        for eval_order, rule in enumerate(applicable_live_rules):
            effective_thresholds = rule.thresholds
            base_thresholds = rule.thresholds

            # Apply overlays if not exempt
            if not rule.overlay_exempt and rule.rule_id in overlay_stack:
                overlay = overlay_stack[rule.rule_id]
                effective_thresholds = overlay.overlay_values
                base_thresholds = overlay.base_values

            fired_eff, fired_base, features = self.evaluate_rule(
                rule, event_features, effective_thresholds, base_thresholds
            )

            if fired_eff:
                fired_live.append(FiredRule(
                    rule=rule,
                    feature_values=features,
                    fired_on_base=fired_base,
                    fired_on_overlay=fired_eff and not fired_base,
                    evaluation_order=eval_order
                ))

        # Evaluate shadow rules (separately, cannot affect outcome)
        for eval_order, rule in enumerate(applicable_shadow_rules):
            fired_eff, _, features = self.evaluate_rule(
                rule, event_features, rule.thresholds, rule.thresholds
            )
            if fired_eff:
                fired_shadow.append(FiredRule(
                    rule=rule,
                    feature_values=features,
                    fired_on_base=True,
                    fired_on_overlay=False,
                    evaluation_order=eval_order
                ))

        return fired_live, fired_shadow

    def resolve_action(
        self,
        fired_live: list[FiredRule],
        fired_shadow: list[FiredRule],
        hard_block_action: Optional[ActionCode] = None
    ) -> DecisionOutcome:
        """
        Resolve action from fired rules.
        Implements 01 §5.12: action precedence with severity and priority.
        """
        # Hard block is the floor
        min_severity = hard_block_action.value if hard_block_action else ActionCode.ALLOW.value

        # Separate critical from non-critical
        critical_firings = [f for f in fired_live if f.rule.critical]
        non_critical_firings = [f for f in fired_live if not f.rule.critical]

        # Resolve action
        action_code = hard_block_action or ActionCode.ALLOW
        action_rule = None
        governance_exception = False

        if critical_firings:
            # Check for conflicting critical rules
            critical_actions = set(f.rule.action for f in critical_firings)
            if len(critical_actions) > 1:
                governance_exception = True
            # Take most severe critical action
            critical_firings.sort(
                key=lambda f: (-f.rule.action.value, f.rule.priority)
            )
            action_code = critical_firings[0].rule.action
            action_rule = critical_firings[0].rule
        elif fired_live:
            # Sort by action severity, then priority
            fired_live.sort(
                key=lambda f: (-f.rule.action.value, f.rule.priority)
            )
            action_code = fired_live[0].rule.action
            action_rule = fired_live[0].rule

        # Ensure action meets minimum threshold from hard block
        if hard_block_action and action_code.value < hard_block_action.value:
            action_code = hard_block_action

        # Gather reason codes (from all fired rules, not just winner)
        from sys import path
        reason_codes = []
        for fired in fired_live:
            if fired.rule.reason_code:
                reason_codes.append(fired.rule.reason_code)

        # Rank reasons (will use core.reason_codes later)
        # For now, just keep them
        primary_reason = reason_codes[0] if reason_codes else 0

        # Identify overlay-induced firings
        overlay_induced = [f.rule.rule_id for f in fired_live if f.fired_on_overlay]

        # Calculate counterfactual (base rules only)
        base_only = [f for f in fired_live if f.fired_on_base]
        counterfactual_action = ActionCode.ALLOW
        if base_only:
            base_only.sort(key=lambda f: (-f.rule.action.value, f.rule.priority))
            counterfactual_action = base_only[0].rule.action

        return DecisionOutcome(
            action_code=action_code,
            action_source_rule_id=action_rule.rule_id if action_rule else "HARD_BLOCK",
            fired_rule_ids=[f.rule.rule_id for f in fired_live],
            fired_on_overlay_ids=overlay_induced,
            shadow_fired_rule_ids=[f.rule.rule_id for f in fired_shadow],
            reason_codes=reason_codes,
            primary_reason_code=primary_reason,
            governance_exception=governance_exception,
            counterfactual_action=counterfactual_action,
            evaluation_details=[]
        )
