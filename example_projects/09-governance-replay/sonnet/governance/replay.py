"""Exact replay (spec 09 §5.1).

Re-derive a decision from its captured evidence -- the recorded config
version, the recorded params, and the request as received -- with no live
call anywhere (09 §5.15 item 9: the replay environment enforces this simply
by construction, since `replay()` never reaches for anything but
`evidence.request`/`evidence.params`/the rebuilt pipeline). Compare every
field the flow returned, using the tolerance bands spec 03 §5.6/09 §5.1
declare: exact for codes and ids, to the cent for money, four decimal places
for rates, 1e-6 absolute for scores and probabilities, 1e-12 relative for
everything else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from governance import flows
from governance.evidence_store import EvidenceRecord, json_safe


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)

_MONEY = ("amount", "instalment", "fee", "cost", "premium", "balance", "obligation",
          "income", "expense", "exposure", "limit", "cap", "ceiling")
_RATE = ("rate",)
_SCORE = ("score", "probability", "_pd", "pd_")


def _compare_scalar(field: str, recorded, replayed) -> str:
    """Returns "exact", "tolerance" (within a declared band, but not bit-identical) or
    "diverged" -- the three-way split spec 09 §5.1's acceptance standard asks for."""
    if _is_nan(recorded) and _is_nan(replayed):
        return "exact"  # decider's own tables use NaN for "no value at this cell" (00 NOTES);
        # NaN != NaN in Python, but two recorded NaNs are not a divergence (spec 09 §5.1's
        # acceptance standard is bit-identical outputs, and NaN is a bit pattern like any other).
    if recorded == replayed:
        return "exact"
    if isinstance(recorded, bool) or isinstance(replayed, bool):
        return "diverged"  # bools compare exactly, never numerically
    if isinstance(recorded, (int, float)) and isinstance(replayed, (int, float)):
        a, b = float(recorded), float(replayed)
        lname = field.lower()
        if any(k in lname for k in _RATE):
            tol = round(a, 4) == round(b, 4)
        elif any(k in lname for k in _SCORE):
            tol = abs(a - b) <= 1e-6
        elif any(k in lname for k in _MONEY):
            tol = round(a, 2) == round(b, 2)
        else:
            tol = abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1.0)
        return "tolerance" if tol else "diverged"
    return "diverged"


def _compare_field(field: str, recorded, replayed) -> str:
    if isinstance(recorded, list) and isinstance(replayed, list):
        if len(recorded) != len(replayed):
            return "diverged"
        results = [_compare_field(field, a, b) for a, b in zip(recorded, replayed)]
        if all(r == "exact" for r in results):
            return "exact"
        return "diverged" if "diverged" in results else "tolerance"
    return _compare_scalar(field, recorded, replayed)


@dataclass(frozen=True)
class FieldDivergence:
    field: str
    recorded: object
    replayed: object
    grade: str  # "tolerance" | "diverged"


@dataclass(frozen=True)
class ReplayVerdict:
    decision_id: str
    flow_code: str
    config_version: str
    verdict: str  # "reproduced" | "reproduced_within_tolerance" | "not_reproduced"
    divergences: tuple[FieldDivergence, ...]
    first_divergence: FieldDivergence | None


def replay(evidence: EvidenceRecord, *, mode: str = "interpreted") -> ReplayVerdict:
    """Re-derives `evidence`'s decision and diffs it field by field against what was recorded."""
    adapter = flows.get(evidence.flow_code)
    built = adapter.build(evidence.config_version, mode=mode)
    typed_request = adapter.type_record(evidence.request)
    replayed = json_safe(built.executable.score(typed_request, evidence.params))

    non_exact: list[FieldDivergence] = []
    diverged: list[FieldDivergence] = []
    for field in sorted(set(evidence.record) | set(replayed)):
        recorded, got = evidence.record.get(field), replayed.get(field)
        grade = _compare_field(field, recorded, got)
        if grade == "exact":
            continue
        d = FieldDivergence(field, recorded, got, grade)
        non_exact.append(d)
        if grade == "diverged":
            diverged.append(d)

    if not non_exact:
        verdict = "reproduced"
    elif not diverged:
        verdict = "reproduced_within_tolerance"
    else:
        verdict = "not_reproduced"

    ordered = diverged or non_exact
    return ReplayVerdict(
        decision_id=evidence.decision_id, flow_code=evidence.flow_code, config_version=evidence.config_version,
        verdict=verdict, divergences=tuple(non_exact), first_divergence=ordered[0] if ordered else None,
    )


def replay_by_id(flow_code: str, decision_id: str, *, mode: str = "interpreted") -> ReplayVerdict:
    """Loads a decision's evidence by id and replays it -- the entry point the servable
    pipeline (`pipeline.py`) and an analyst's tool both call."""
    from governance import evidence_store

    evidence = evidence_store.load(flow_code, decision_id)
    return replay(evidence, mode=mode)
