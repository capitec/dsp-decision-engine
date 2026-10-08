"""Release certification: does a candidate version change what it says it changes, and nothing else."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from governance.coverage import coverage
from governance.evidence import Evidence
from governance.flows import FLOWS
from governance.replay import grade
from governance.replayer import Replayer
from governance.versions import version_diff


@dataclass(frozen=True)
class Certificate:
    """The sign-off record. `unexpected` moved without being declared, `unmoved` were declared to move and did
    not; either blocks the release. A release that declares nothing expected to move is held to `moved == ()`."""

    flow: str
    before: str
    declared: tuple[str, ...]
    moved: tuple[str, ...]
    unexpected: tuple[str, ...]
    unmoved: tuple[str, ...]
    logic_proven_identical: bool | None
    changed_documents: tuple[str, ...]
    changed_params: tuple[str, ...]
    dead_rules: tuple[str, ...]
    approvals: tuple[str, ...]
    accountable: str

    @property
    def passed(self) -> bool:
        return not (self.unexpected or self.unmoved)

    def to_json(self) -> dict:
        return {**asdict(self), "passed": self.passed}


def _moved(e: Evidence, candidate: Replayer) -> bool:
    now = candidate.score(e.request)
    return any(grade(k, e.response.get(k), now.get(k)) == "diverged" for k in e.response.keys() | now.keys())


def certify(golden: list[Evidence], candidate: Replayer, expected_to_move: set[str] = frozenset(),
            approvals: tuple[str, ...] = (), accountable: str = "") -> Certificate:
    """Run the golden set (evidence of earlier decisions) through `candidate` and check the movement.

    A decision moved when any field is outside its declared tolerance. `expected_to_move` are the decision ids
    the release declares it will change.

        certify(store.all("loan"), candidate, {"gr-demo-003"}, accountable="J. Moyo").passed
    """
    flow = FLOWS[golden[0].flow]
    moved = tuple(sorted(e.decision_id for e in golden if _moved(e, candidate)))
    current = flow.replayer(golden[0].config_version)
    changes = version_diff(flow, current, candidate)
    dead = coverage(flow, [e.response for e in golden], golden[0].config_version).dead
    return Certificate(
        golden[0].flow, golden[0].config_version, tuple(sorted(expected_to_move)), moved,
        tuple(sorted(set(moved) - expected_to_move)), tuple(sorted(expected_to_move - set(moved))),
        changes.proof, tuple(changes.documents), tuple(p["path"] for p in changes.params), dead, approvals,
        accountable)
