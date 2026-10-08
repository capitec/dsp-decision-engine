"""Decision evidence: what a decision needs kept so it can be re-derived years later."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from governance.flows import Flow


def digest(document) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Evidence:
    """One decision as it happened.

    `request` is the JSON as received and `response` what the client got back. `documents` holds the digest of
    every config document of `config_version` (params included), because a version number can be edited in place
    and a digest cannot.
    """

    flow: str
    decision_id: str
    decision_date: str
    config_version: str
    mode: str
    documents: dict[str, str]
    params: dict
    request: dict
    response: dict


def capture(flow: Flow, request: dict, version: str | None = None, mode: str = "fused") -> Evidence:
    """Score `request` the way the flow is served and keep the evidence.

        capture(LOAN, json.load(open("sample_request.json")))
    """
    replayer = flow.replayer(version, mode)
    response = replayer.score(request)
    config = flow.store().read(replayer.active).config
    return Evidence(flow.code, response["decision_id"], response["decision_date"], str(replayer.active), mode,
                    {name: digest(doc) for name, doc in config.items()}, config.get("params", {}), request, response)


class EvidenceStore:
    """Evidence on disk, one file per decision, written once.

        store = EvidenceStore("evidence")
        store.save(evidence)
        store.load("loan", "d5f5f6b0-3f8a-4f2a-9c4b-2f6e7a1b0003")
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, evidence: Evidence) -> Path:
        path = self.root / evidence.flow / f"{evidence.decision_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x") as f:  # "x": evidence is never overwritten
            json.dump(asdict(evidence), f, indent=1, sort_keys=True)
        return path

    def load(self, flow: str, decision_id: str) -> Evidence:
        return Evidence(**json.loads((self.root / flow / f"{decision_id}.json").read_text()))

    def all(self, flow: str) -> list[Evidence]:
        return [Evidence(**json.loads(p.read_text())) for p in sorted((self.root / flow).glob("*.json"))]
