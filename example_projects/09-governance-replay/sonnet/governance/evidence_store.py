"""Decision evidence: captured, persisted once, read many times.

09 §5.15 states the contract every flow must satisfy for this harness to work
at all: a stable decision id (item 1), the inputs as received (item 6), the
parameter set in force (item 10), the config version (items 4-5). None of
01/03/05 persist evidence externally today -- their own NOTES.md documents
that `.score()` returning every input plus every step output *is* the
decision record (00's own §5.15 items 1/6/10 are already satisfied by what
the pipeline emits; only writing it somewhere durable is missing, and that is
a store, not a flow, concern -- out of scope for each of their slices per
SCOPE.md).

This module is that store's minimal, honest stand-in: capturing a decision's
evidence is one call to the flow's own `.score()` (never a second,
harness-side re-derivation of anything), and the request is kept separate
from the resulting record so replay has a real "as received" input to
re-score rather than having to guess which fields of the recorded output were
inputs. Write-once (09 §9 item 3): `save` always writes a fresh file; nothing
in this module edits one in place.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from governance.flows import BuiltFlow, FlowAdapter

STORE_ROOT = Path(__file__).resolve().parents[1] / "evidence_store"


def json_safe(value: Any) -> Any:
    """`.score()`'s row loop can hand back numpy scalars for a numeric output (see
    01/00/05's own NOTES.md: a `list`-typed *input* arrives as a numpy array; the same
    row-loop machinery does the same for some numeric outputs) -- normalise those, and
    `datetime.date`, to plain JSON-able Python before anything is persisted or compared."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _params_digest(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class EvidenceRecord:
    decision_id: str
    flow_code: str
    decision_date: str
    config_version: str
    params_digest: str
    params: dict
    request: dict   # the inputs as received (09 §5.15 item 6) -- untyped JSON, exactly what arrived
    record: dict    # the full scored record: every input plus every step output

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, doc: dict) -> "EvidenceRecord":
        return cls(**doc)


def capture(adapter: FlowAdapter, built: BuiltFlow, request: dict) -> EvidenceRecord:
    """Scores `request` through `built` and captures the result as evidence.

    The one place this harness plays the part of a flow's own evidence-emission
    step -- never called from `replay.py`, which must only ever *read* a
    previously captured record, never produce one (a replay that could also
    write evidence would blur the line §5.14.5/§9 item 3 draws between the two).
    """
    typed_request = adapter.type_record(request)
    outcome = json_safe(built.executable.score(typed_request, built.params))
    decision_id = outcome.get("decision_id") or request.get("decision_id")
    if not decision_id:
        raise ValueError("request carries no decision_id (09 §5.15 item 1: assigned before any logic runs)")
    decision_date = outcome.get("decision_date") or request.get("decision_date")
    return EvidenceRecord(
        decision_id=str(decision_id), flow_code=adapter.code, decision_date=str(decision_date),
        config_version=built.config_version, params_digest=_params_digest(built.params),
        params=json_safe(built.params), request=request, record=outcome,
    )


def save(evidence: EvidenceRecord) -> Path:
    out_dir = STORE_ROOT / evidence.flow_code
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{evidence.decision_id}.json"
    path.write_text(json.dumps(evidence.to_json(), indent=2, sort_keys=True, default=str))
    return path


def load(flow_code: str, decision_id: str) -> EvidenceRecord:
    path = STORE_ROOT / flow_code / f"{decision_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no evidence for flow {flow_code!r} decision {decision_id!r} ({path})")
    return EvidenceRecord.from_json(json.loads(path.read_text()))


def list_ids(flow_code: str) -> list[str]:
    d = STORE_ROOT / flow_code
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
