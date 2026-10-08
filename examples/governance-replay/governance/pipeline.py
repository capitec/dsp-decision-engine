"""The replay service: POST a flow and a decision id, get the replay verdict."""
from __future__ import annotations

from pathlib import Path

from decider import allow_fallback, param, step

from governance.evidence import EvidenceStore
from governance.replay import replay


@allow_fallback
def replay_decision(flow_code: str, decision_id: str, evidence_dir: str = param("evidence")) -> tuple[str, int, list[str], str]:
    verdict = replay(EvidenceStore(Path(__file__).parents[1] / evidence_dir).load(flow_code, decision_id))
    first = verdict.first
    return verdict.verdict, len(verdict.divergences), [d.field for d in verdict.divergences], first.field if first else ""


def build():
    return step(replay_decision, outputs=("replay_verdict", "divergence_count", "diverged_fields", "first_divergence"))
