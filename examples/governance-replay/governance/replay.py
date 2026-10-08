"""Exact replay: re-derive a decision from its evidence and say whether it comes out the same."""
from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl

from governance.evidence import Evidence, digest
from governance.flows import FLOWS

# Declared tolerances by field-name fragment: money to the cent, rates to 4 places, scores to 1e-6.
_MONEY = ("amount", "instalment", "fee", "cost", "premium", "balance", "income", "expense", "exposure", "limit", "cap")
_SCORE = ("score", "probability", "_pd")


def grade(field: str, recorded, replayed) -> str:
    """`"exact"`, `"tolerance"` (inside the declared band) or `"diverged"`; recurses into lists and dicts.

        grade("instalment", 4969.47, 4969.470000001)   # "tolerance"
    """
    if isinstance(recorded, dict) and isinstance(replayed, dict):
        grades = [grade(k, recorded.get(k), replayed.get(k)) for k in recorded.keys() | replayed.keys()]
    elif isinstance(recorded, list) and isinstance(replayed, list) and len(recorded) == len(replayed):
        grades = [grade(field, a, b) for a, b in zip(recorded, replayed)]
    else:
        return _scalar(field.lower(), recorded, replayed)
    return "diverged" if "diverged" in grades else "tolerance" if "tolerance" in grades else "exact"


def _scalar(field: str, a, b) -> str:
    if a == b:
        return "exact"
    numbers = [isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)]
    if not all(numbers):
        return "diverged"
    if math.isnan(a) and math.isnan(b):
        return "exact"
    if "rate" in field:
        close = round(a, 4) == round(b, 4)
    elif any(k in field for k in _SCORE):
        close = abs(a - b) <= 1e-6
    elif any(k in field for k in _MONEY):
        close = round(a, 2) == round(b, 2)
    else:
        close = abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1.0)
    return "tolerance" if close else "diverged"


@dataclass(frozen=True)
class Divergence:
    field: str
    recorded: object
    replayed: object
    grade: str
    step: str | None = None  # the step that first wrote the field in this run


@dataclass(frozen=True)
class Verdict:
    """`verdict` is `reproduced`, `reproduced_within_tolerance`, `not_reproduced`, or `not_reproducible` when
    a config document changed since capture (`changed_documents` names them) and the decision cannot be re-run.
    `first` is the earliest divergence in execution order."""

    decision_id: str
    verdict: str
    divergences: tuple[Divergence, ...] = ()
    changed_documents: tuple[str, ...] = ()

    @property
    def first(self) -> Divergence | None:
        return next((d for d in self.divergences if d.grade == "diverged"), None) or next(iter(self.divergences), None)


def replay(evidence: Evidence, mode: str | None = None, store=None) -> Verdict:
    """Re-score the recorded request against the recorded config version and compare every response field.

        replay(evidence).verdict   # "reproduced"
    """
    flow = FLOWS[evidence.flow]
    store = store or flow.store()
    config = store.read(evidence.config_version).config
    changed = tuple(sorted(n for n in config.keys() | evidence.documents.keys()
                           if digest(config.get(n)) != evidence.documents.get(n)))
    if changed:
        return Verdict(evidence.decision_id, "not_reproducible", changed_documents=changed)
    replayer = flow.replayer(evidence.config_version, mode or evidence.mode, store)
    replayed = replayer.score(evidence.request)
    found = {}
    for name in sorted(evidence.response.keys() | replayed.keys()):
        g = grade(name, evidence.response.get(name), replayed.get(name))
        if g != "exact":
            found[name] = Divergence(name, evidence.response.get(name), replayed.get(name), g)
    if not found:
        return Verdict(evidence.decision_id, "reproduced")
    ordered = _in_run_order(replayer, found)
    verdict = "not_reproduced" if any(d.grade == "diverged" for d in ordered) else "reproduced_within_tolerance"
    return Verdict(evidence.decision_id, verdict, tuple(ordered))


def _in_run_order(replayer, found: dict[str, Divergence]) -> list[Divergence]:
    """The divergences ordered by the step that first wrote each, from a debug session over the same record."""
    live = replayer.untapped()
    session = live.executable.session(pl.DataFrame([replayer.typed]), live.params)
    session.resume()
    written = {}
    for event in session.events:
        for name in getattr(event, "outputs", {}):
            written.setdefault(name, event.origin.path)
    steps = list(dict.fromkeys(written.values()))
    placed = [Divergence(d.field, d.recorded, d.replayed, d.grade, written.get(name)) for name, d in found.items()]
    # Fields no step wrote are request inputs echoed back; they come first.
    return sorted(placed, key=lambda d: steps.index(d.step) if d.step in steps else -1)
