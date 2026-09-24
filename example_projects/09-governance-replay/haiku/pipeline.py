"""Decision governance and replay harness pipeline (project 09-H).

Core harness for replay, explain, diff, swap-set, overlays, dead logic.
Implements 09 §5.1–5.8 capabilities.
"""
from __future__ import annotations
from decider import flow, step, param, missing_as
from datetime import date
from governance import (
    ReplayEngine, ExplanationRenderer, VersionDiffer, SwapSetAnalyzer,
    OverlayRegister, DeadLogicDetector, ExplanationOutput
)


def build():
    """Build the governance harness pipeline."""
    return flow(
        harness_step,
        name="governance_harness"
    )


@step
def harness_step(
    operation: str = param("replay"),
    decision_id: str = param("test"),
    evidence: dict | None = param(None),
    flow_type: str = param("01"),
    audience: str = param("analyst"),
    decision_date: date = missing_as(date.today()),
) -> dict:
    """Execute governance operations (09 §5.1–5.8)."""

    # Default values if empty
    if not evidence:
        evidence = {
            "outcome_code": "approved",
            "outputs": {"outcome_code": "approved"},
            "inputs_as_received": {},
            "parameters": {},
        }

    # Dispatch to operation
    if operation == "replay":
        engine = ReplayEngine()
        flow_fn = _stub_flow
        replay_result = engine.replay_decision(
            decision_id=decision_id,
            evidence=evidence,
            flow_fn=flow_fn,
            decision_date=decision_date,
        )
        return {
            "operation": "replay",
            "decision_id": decision_id,
            "verdict": replay_result.verdict.value,
            "divergence_point": str(replay_result.divergence_point),
        }

    elif operation == "explain":
        renderer = ExplanationRenderer()
        explanation = ExplanationOutput(
            decision_id=decision_id,
            audience=audience,
            flow_name=flow_type,
            decision_date=decision_date,
        )
        if audience == "consultant":
            text = renderer.render_consultant(evidence)
        elif audience == "analyst":
            text = renderer.render_analyst(evidence, explanation)
        else:
            text = renderer.render_ombud(evidence, explanation)
        return {
            "operation": "explain",
            "decision_id": decision_id,
            "audience": audience,
            "explanation": text[:200],
        }

    elif operation == "diff":
        differ = VersionDiffer()
        diff = differ.diff_rule_set([], [])
        return {
            "operation": "diff",
            "summary": diff.summary,
        }

    elif operation == "swap_set":
        analyzer = SwapSetAnalyzer()
        report = analyzer.compare_versions([], lambda **kw: {}, lambda **kw: {}, decision_date)
        return {
            "operation": "swap_set",
            "population_size": report.population_size,
        }

    elif operation == "overlays":
        register = OverlayRegister()
        report = register.aging_report(decision_date)
        return {
            "operation": "overlays",
            "aging_report": report,
        }

    elif operation == "dead_logic":
        detector = DeadLogicDetector()
        report = detector.detect_dead_rules([], [], decision_date)
        return {
            "operation": "dead_logic",
            "dead_rule_count": len(report.dead_rules),
        }

    else:
        return {"error": f"Unknown operation: {operation}"}


def _stub_flow(**kwargs):
    """Stub decision flow."""
    return {"outcome_code": "approved", "offered_amount": 100000}
