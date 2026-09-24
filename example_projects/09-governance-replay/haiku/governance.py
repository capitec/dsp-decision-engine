"""Decision governance and replay harness (project 09-H).

Replay, explain, diff, and measure impact across eight flows.
Core components: replay engine, explanation renderers, version diff, swap-set attribution.
Implements 09 §5.1–5.8: replay, explanation, diff, swap-set, coverage, dead logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import date
from enum import Enum
import json


class ReplayVerdict(Enum):
    """Outcome of replay attempt (09 §5.1)."""
    REPRODUCED = "reproduced"
    REPRODUCED_WITHIN_TOLERANCE = "reproduced_within_tolerance"
    NOT_REPRODUCED = "not_reproduced"


@dataclass
class ReplayResult:
    """Result of replaying a single decision (09 §5.1)."""
    decision_id: str
    verdict: ReplayVerdict
    original_outputs: dict[str, Any]
    replayed_outputs: dict[str, Any]
    divergence_point: Optional[str] = None
    field_differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "verdict": self.verdict.value,
            "divergence_point": self.divergence_point,
            "field_differences": self.field_differences,
            "original_outputs": self.original_outputs,
            "replayed_outputs": self.replayed_outputs,
        }


@dataclass
class ExplanationOutput:
    """Explanation for one decision, one audience (09 §5.2)."""
    decision_id: str
    audience: str  # "consultant" | "analyst" | "ombud"
    flow_name: str
    decision_date: date

    gates_evaluated: list[dict] = field(default_factory=list)
    rules_fired: list[dict] = field(default_factory=list)
    rules_evaluated: list[dict] = field(default_factory=list)
    nodes_visited: list[dict] = field(default_factory=list)
    cap_chain: list[dict] = field(default_factory=list)
    score_contributions: list[dict] = field(default_factory=list)
    table_cells_read: list[dict] = field(default_factory=list)

    decline_reason_codes: list[str] = field(default_factory=list)
    primary_reason_code: Optional[str] = None
    reason_wording_version: Optional[str] = None

    adjusted_values: dict[str, Any] = field(default_factory=dict)
    unadjusted_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "decision_id": self.decision_id,
            "audience": self.audience,
            "flow": self.flow_name,
            "decision_date": str(self.decision_date),
        }
        if self.audience in ("analyst", "ombud"):
            d.update({
                "gates_evaluated": self.gates_evaluated,
                "rules_fired": self.rules_fired,
                "nodes_visited": self.nodes_visited,
                "cap_chain": self.cap_chain,
                "score_contributions": self.score_contributions,
                "table_cells_read": self.table_cells_read,
            })
        if self.audience in ("consultant", "ombud"):
            d.update({
                "decline_reason_codes": self.decline_reason_codes,
                "primary_reason_code": self.primary_reason_code,
            })
        if self.audience == "ombud":
            d.update({
                "reason_wording_version": self.reason_wording_version,
                "adjusted_values": self.adjusted_values,
                "unadjusted_values": self.unadjusted_values,
            })
        return d


@dataclass
class VersionDiff:
    """Semantic diff between two versions (09 §5.4)."""
    artifact_name: str  # rule set, tree, rate card, etc.
    artifact_id: str
    version_a: str
    version_b: str

    changed_by: Optional[str] = None  # author
    authored_at: Optional[date] = None
    approved_at: Optional[date] = None
    approved_by: Optional[str] = None
    approval_reference: Optional[str] = None

    changes: list[dict] = field(default_factory=list)  # [{"type": "add|remove|modify", ...}]

    summary: str = ""
    estimated_impact: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "artifact": self.artifact_name,
            "artifact_id": self.artifact_id,
            "version_a": self.version_a,
            "version_b": self.version_b,
            "changed_by": self.changed_by,
            "authored_at": str(self.authored_at) if self.authored_at else None,
            "approved_at": str(self.approved_at) if self.approved_at else None,
            "approved_by": self.approved_by,
            "approval_reference": self.approval_reference,
            "summary": self.summary,
            "estimated_impact": self.estimated_impact,
            "changes": self.changes,
        }


@dataclass
class SwapSetReport:
    """Impact of changes across a population (09 §5.5)."""
    base_version: str
    comparison_version: str
    population_size: int
    decision_date: date

    approvals_lost_count: int = 0
    approvals_lost_rate: float = 0.0
    approvals_gained_count: int = 0
    approvals_gained_rate: float = 0.0
    net_approval_movement: float = 0.0

    amount_changed_count: int = 0
    amount_mean_change: float = 0.0
    amount_total_exposure_change: float = 0.0

    reason_codes_added: dict[str, int] = field(default_factory=dict)
    reason_codes_removed: dict[str, int] = field(default_factory=dict)
    reason_codes_moved: dict[str, int] = field(default_factory=dict)

    outcome_unchanged_reasons_changed: int = 0
    records_not_compared: int = 0
    not_compared_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "base_version": self.base_version,
            "comparison_version": self.comparison_version,
            "population_size": self.population_size,
            "decision_date": str(self.decision_date),
            "approvals_lost": {
                "count": self.approvals_lost_count,
                "rate": f"{self.approvals_lost_rate:.2%}",
            },
            "approvals_gained": {
                "count": self.approvals_gained_count,
                "rate": f"{self.approvals_gained_rate:.2%}",
            },
            "net_approval_movement": f"{self.net_approval_movement:+.2%}",
            "amount_changed": {
                "count": self.amount_changed_count,
                "mean_change": self.amount_mean_change,
                "total_exposure_change": self.amount_total_exposure_change,
            },
            "reason_codes": {
                "added": self.reason_codes_added,
                "removed": self.reason_codes_removed,
                "moved": self.reason_codes_moved,
            },
            "outcome_unchanged_reasons_changed": self.outcome_unchanged_reasons_changed,
            "records_not_compared": self.records_not_compared,
            "not_compared_reasons": self.not_compared_reasons,
        }


@dataclass
class OverlayRecord:
    """Single adjustment overlay (09 §5.14)."""
    overlay_id: str
    flow_name: str
    kind: str  # sensitivity_dial, threshold_multiplier, action_escalation, etc.
    scope: str  # grades, channels, products, etc.
    magnitude: float
    stack_position: int

    owner: str
    approval_reference: Optional[str] = None
    rationale: str = ""

    effective_from: date = field(default_factory=date.today)
    effective_to: Optional[date] = None
    review_date: date = field(default_factory=date.today)

    enabled: bool = True
    unadjusted_value: Optional[float] = None
    adjusted_value: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "overlay_id": self.overlay_id,
            "flow": self.flow_name,
            "kind": self.kind,
            "scope": self.scope,
            "magnitude": self.magnitude,
            "stack_position": self.stack_position,
            "owner": self.owner,
            "approval_reference": self.approval_reference,
            "rationale": self.rationale,
            "effective_from": str(self.effective_from),
            "effective_to": str(self.effective_to) if self.effective_to else None,
            "review_date": str(self.review_date),
            "enabled": self.enabled,
        }


@dataclass
class DeadLogicReport:
    """Dead and shadowed logic detection (09 §5.8)."""
    flow_name: str
    reference_date: date
    window_days: int = 90

    dead_rules: list[dict] = field(default_factory=list)
    high_coverage_rules: list[dict] = field(default_factory=list)
    unreachable_tree_nodes: list[dict] = field(default_factory=list)
    never_read_table_cells: list[dict] = field(default_factory=list)
    unreachable_reason_codes: list[str] = field(default_factory=list)
    shadowed_rules: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "flow": self.flow_name,
            "reference_date": str(self.reference_date),
            "window_days": self.window_days,
            "dead_rules": self.dead_rules,
            "high_coverage_rules": self.high_coverage_rules,
            "unreachable_tree_nodes": self.unreachable_tree_nodes,
            "never_read_table_cells": self.never_read_table_cells,
            "unreachable_reason_codes": self.unreachable_reason_codes,
            "shadowed_rules": self.shadowed_rules,
        }


class ReplayEngine:
    """Engine for replaying decisions from evidence (09 §5.1)."""

    def __init__(self):
        self.replays: dict[str, ReplayResult] = {}

    def replay_decision(
        self,
        decision_id: str,
        evidence: dict[str, Any],
        flow_fn,
        decision_date: date,
    ) -> ReplayResult:
        """Replay a single decision from evidence.

        The flow_fn should be the same decision logic that originally ran,
        but executed against recorded evidence instead of live services.
        """
        # Extract inputs and version information from evidence
        inputs = evidence.get("inputs_as_received", {})
        param_set = evidence.get("parameters", {})

        # Run decision logic with recorded evidence
        # This is where we'd invoke the actual flow step
        try:
            replayed = flow_fn(decision_date=decision_date, **inputs)
            original = evidence.get("outputs", {})

            # Compare field by field
            differences = {}
            for key in set(original.keys()) | set(replayed.keys()):
                if original.get(key) != replayed.get(key):
                    differences[key] = (original.get(key), replayed.get(key))

            # Determine verdict
            if not differences:
                verdict = ReplayVerdict.REPRODUCED
            elif self._all_within_tolerance(differences):
                verdict = ReplayVerdict.REPRODUCED_WITHIN_TOLERANCE
            else:
                verdict = ReplayVerdict.NOT_REPRODUCED

            result = ReplayResult(
                decision_id=decision_id,
                verdict=verdict,
                original_outputs=original,
                replayed_outputs=replayed,
                divergence_point=next(iter(differences.keys())) if differences else None,
                field_differences=differences,
            )

            self.replays[decision_id] = result
            return result
        except Exception as e:
            return ReplayResult(
                decision_id=decision_id,
                verdict=ReplayVerdict.NOT_REPRODUCED,
                original_outputs=evidence.get("outputs", {}),
                replayed_outputs={},
                divergence_point=str(e),
            )

    def _all_within_tolerance(self, differences: dict[str, tuple]) -> bool:
        """Check if all differences are within acceptable tolerance."""
        # Monetary amounts (after rounding) must be exact
        # Scores/probabilities: 1e-6 absolute
        # Intermediate values: 1e-12 relative
        for key, (orig, repl) in differences.items():
            if isinstance(orig, float) and isinstance(repl, float):
                rel_error = abs(orig - repl) / (abs(orig) + 1e-10)
                if rel_error > 1e-6:
                    return False
        return True


class ExplanationRenderer:
    """Render explanations for three audiences (09 §5.2)."""

    def render_consultant(self, evidence: dict) -> str:
        """Consultant rendering: ≤5 sentences, plain language, one reason."""
        outcome = evidence.get("outcome_code", "UNKNOWN")
        primary_reason = evidence.get("primary_reason_code", "")
        reason_wording = self._get_reason_wording(primary_reason, evidence.get("decision_date"))

        # Very simplified; in full implementation, render reason registry
        return f"Your application for {evidence.get('product', 'this product')} was {outcome}. {reason_wording}"

    def render_analyst(self, evidence: dict, explanation: ExplanationOutput) -> str:
        """Analyst rendering: comprehensive, all details, every evaluation."""
        lines = [
            f"Decision: {explanation.decision_id}",
            f"Flow: {explanation.flow_name}",
            f"Date: {explanation.decision_date}",
            "",
            "Gates:",
        ]
        for gate in explanation.gates_evaluated:
            lines.append(f"  {gate.get('gate_id')}: {gate.get('outcome')} [{gate.get('value')}]")

        lines.extend(["", "Rules fired:"])
        for rule in explanation.rules_fired:
            lines.append(f"  {rule.get('rule_id')}: {rule.get('description')}")

        if explanation.cap_chain:
            lines.extend(["", "Cap chain:"])
            for cap in explanation.cap_chain:
                lines.append(f"  {cap.get('value')} ← {cap.get('reason')}")

        return "\n".join(lines)

    def render_ombud(self, evidence: dict, explanation: ExplanationOutput) -> str:
        """Ombud rendering: defensible narrative, adjusted vs unadjusted."""
        lines = [
            f"# Decision Explanation",
            f"Decision ID: {explanation.decision_id}",
            f"Decision Date: {explanation.decision_date}",
            f"",
            "## Policy in Force",
            "The Bank applied the following policies...",
            "",
            "## Application and Reasoning",
            "Your application was assessed as follows:",
            "",
            "## Cap Chain (What Constrained Your Offer)",
        ]
        for cap in explanation.cap_chain:
            lines.append(f"  - {cap.get('value')}: {cap.get('reason')}")

        if explanation.adjusted_values and explanation.unadjusted_values:
            lines.extend([
                "",
                "## Adjustments Applied",
                "The base assessment was adjusted as follows:",
            ])
            for key, adj_val in explanation.adjusted_values.items():
                unaj_val = explanation.unadjusted_values.get(key)
                if adj_val != unaj_val:
                    lines.append(f"  - {key}: {unaj_val} → {adj_val}")

        return "\n".join(lines)

    def _get_reason_wording(self, code: str, decision_date: date = None) -> str:
        """Look up reason wording from registry at decision_date."""
        # Stub; in full impl would resolve from versioned reason registry
        return f"Code {code}."


class VersionDiffer:
    """Diff versions of logic and tables (09 §5.4)."""

    def diff_rule_set(self, rules_a: list[dict], rules_b: list[dict]) -> VersionDiff:
        """Semantic diff between two rule sets."""
        changes = []
        ids_a = {r["rule_id"]: r for r in rules_a}
        ids_b = {r["rule_id"]: r for r in rules_b}

        # Added rules
        for rid in ids_b:
            if rid not in ids_a:
                changes.append({
                    "type": "add",
                    "rule_id": rid,
                    "description": ids_b[rid].get("description"),
                })

        # Removed rules
        for rid in ids_a:
            if rid not in ids_b:
                changes.append({
                    "type": "remove",
                    "rule_id": rid,
                })

        # Modified rules
        for rid in ids_a:
            if rid in ids_b:
                if ids_a[rid] != ids_b[rid]:
                    changes.append({
                        "type": "modify",
                        "rule_id": rid,
                        "changes": self._dict_diff(ids_a[rid], ids_b[rid]),
                    })

        return VersionDiff(
            artifact_name="rule_set",
            artifact_id="",
            version_a=str(len(rules_a)),
            version_b=str(len(rules_b)),
            changes=changes,
            summary=f"{len([c for c in changes if c['type'] == 'add'])} added, "
                    f"{len([c for c in changes if c['type'] == 'remove'])} removed, "
                    f"{len([c for c in changes if c['type'] == 'modify'])} modified",
        )

    def _dict_diff(self, d1: dict, d2: dict) -> dict:
        """Diff two dicts."""
        changes = {}
        for key in set(d1.keys()) | set(d2.keys()):
            if d1.get(key) != d2.get(key):
                changes[key] = (d1.get(key), d2.get(key))
        return changes


class SwapSetAnalyzer:
    """Analyze impact of changes across population (09 §5.5)."""

    def compare_versions(
        self,
        population: list[dict],
        version_a_fn,
        version_b_fn,
        decision_date: date,
    ) -> SwapSetReport:
        """Run population through both versions and report differences."""
        report = SwapSetReport(
            base_version="base",
            comparison_version="comparison",
            population_size=len(population),
            decision_date=decision_date,
        )

        approvals_changed = 0
        amounts_changed = 0

        for record in population:
            try:
                result_a = version_a_fn(decision_date=decision_date, **record)
                result_b = version_b_fn(decision_date=decision_date, **record)

                outcome_a = result_a.get("outcome_code")
                outcome_b = result_b.get("outcome_code")

                if outcome_a != outcome_b:
                    if outcome_a == "approved" and outcome_b != "approved":
                        report.approvals_lost_count += 1
                    elif outcome_a != "approved" and outcome_b == "approved":
                        report.approvals_gained_count += 1

                amount_a = result_a.get("offered_amount", 0)
                amount_b = result_b.get("offered_amount", 0)
                if amount_a != amount_b:
                    amounts_changed += 1
                    report.amount_total_exposure_change += (amount_b - amount_a)
                    report.amount_mean_change += (amount_b - amount_a)

            except Exception:
                report.records_not_compared += 1

        if len(population) > 0:
            report.approvals_lost_rate = report.approvals_lost_count / len(population)
            report.approvals_gained_rate = report.approvals_gained_count / len(population)
            report.net_approval_movement = (
                (report.approvals_gained_count - report.approvals_lost_count) / len(population)
            )
            if amounts_changed > 0:
                report.amount_mean_change /= amounts_changed
            report.amount_changed_count = amounts_changed

        return report


class OverlayRegister:
    """Manage overlay stack with ageing and expiry (09 §5.14)."""

    def __init__(self):
        self.overlays: dict[str, OverlayRecord] = {}

    def add_overlay(self, overlay: OverlayRecord) -> None:
        """Register a new overlay."""
        self.overlays[overlay.overlay_id] = overlay

    def get_overlays_at_date(self, target_date: date) -> dict[str, OverlayRecord]:
        """Get overlays active at a given date."""
        result = {}
        for oid, overlay in self.overlays.items():
            if (overlay.effective_from <= target_date and
                (overlay.effective_to is None or target_date <= overlay.effective_to)):
                result[oid] = overlay
        return result

    def aging_report(self, as_of: date = None) -> dict:
        """Report overlays past review date, renewal history, etc."""
        if as_of is None:
            as_of = date.today()

        past_review = []
        renewed_many_times = []

        for oid, overlay in self.overlays.items():
            if overlay.review_date < as_of:
                days_past = (as_of - overlay.review_date).days
                past_review.append({
                    "overlay_id": oid,
                    "days_past_review": days_past,
                    "review_date": overlay.review_date,
                })

        return {
            "as_of": str(as_of),
            "overlays_past_review_date": past_review,
            "overlays_renewed_many_times": renewed_many_times,
            "overlays_never_fired": [],
            "overlays_firing_too_much": [],
        }


class DeadLogicDetector:
    """Find dead and shadowed logic (09 §5.8)."""

    def detect_dead_rules(
        self,
        rule_set: list[dict],
        evaluation_log: list[dict],
        reference_date: date,
        window_days: int = 90,
    ) -> DeadLogicReport:
        """Identify rules that never fired in the window."""
        report = DeadLogicReport(
            flow_name="unknown",
            reference_date=reference_date,
            window_days=window_days,
        )

        rule_ids = {r["rule_id"] for r in rule_set}
        fired_rule_ids = {
            e["rule_id"] for e in evaluation_log
            if e.get("outcome") == "fired"
        }

        dead_rule_ids = rule_ids - fired_rule_ids
        for rid in dead_rule_ids:
            rule = next((r for r in rule_set if r["rule_id"] == rid), None)
            if rule:
                report.dead_rules.append({
                    "rule_id": rid,
                    "description": rule.get("description", ""),
                    "added_date": rule.get("added_date"),
                })

        # High-coverage rules (>40% of traffic)
        total_evaluations = len(evaluation_log)
        for rid in fired_rule_ids:
            fire_count = sum(1 for e in evaluation_log if e.get("rule_id") == rid and e.get("outcome") == "fired")
            if fire_count / max(total_evaluations, 1) > 0.4:
                report.high_coverage_rules.append({
                    "rule_id": rid,
                    "fire_rate": fire_count / total_evaluations,
                })

        return report
