"""Swap-set analysis and attribution (spec 09 §5.5, acceptance §10 item 20).

Runs two versions of a flow over the same population and reports who moved.
Attribution across population/data/logic/overlay is built the way §5.5 itself
resolves "which one of five changes cost us the movement": n changes -> n+1
cumulative runs, each increment's own swap set measured separately, in a
declared order -- never one number for a whole release.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from governance import flows
from governance.evidence_store import json_safe
from governance.flows import BuiltFlow


@dataclass(frozen=True)
class SwapSetReport:
    label: str
    population_size: int
    outcome_moved: tuple[dict, ...]     # ({"decision_id":..., "before":..., "after":...}, ...)
    reasons_appeared: dict              # {code: count}
    reasons_disappeared: dict
    unmatched: tuple[str, ...]
    outcome_moved_count: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "outcome_moved_count", len(self.outcome_moved))


def _score_population(adapter, built: BuiltFlow, records: list[dict]) -> dict[str, dict]:
    out = {}
    for record in records:
        typed = adapter.type_record(record)
        result = json_safe(built.executable.score(typed, built.params))
        decision_id = result.get("decision_id") or record.get("decision_id")
        out[str(decision_id)] = result
    return out


def compare(adapter, before_by_id: dict, after_by_id: dict, label: str) -> SwapSetReport:
    outcome_field, reason_field = adapter.outcome_field, adapter.reason_field
    moved, unmatched = [], []
    appeared: dict = {}
    disappeared: dict = {}
    for decision_id in before_by_id.keys() | after_by_id.keys():
        before, after = before_by_id.get(decision_id), after_by_id.get(decision_id)
        if before is None or after is None:
            unmatched.append(decision_id)
            continue
        b_out, a_out = before.get(outcome_field), after.get(outcome_field)
        if b_out != a_out:
            moved.append({"decision_id": decision_id, "before": b_out, "after": a_out})
        b_reasons, a_reasons = set(before.get(reason_field) or []), set(after.get(reason_field) or [])
        for code in a_reasons - b_reasons:
            appeared[code] = appeared.get(code, 0) + 1
        for code in b_reasons - a_reasons:
            disappeared[code] = disappeared.get(code, 0) + 1
    return SwapSetReport(label=label, population_size=len(before_by_id) or len(after_by_id),
                          outcome_moved=tuple(moved), reasons_appeared=appeared, reasons_disappeared=disappeared,
                          unmatched=tuple(unmatched))


def swap_set(flow_code: str, population: list[dict], version_a: str, version_b: str, *,
             mode: str = "interpreted", label: str = "A vs B") -> SwapSetReport:
    """§5.5's basic case: the same population, two config (logic) versions."""
    adapter = flows.get(flow_code)
    built_a = adapter.build(version_a, mode=mode)
    built_b = adapter.build(version_b, mode=mode)
    before = _score_population(adapter, built_a, population)
    after = _score_population(adapter, built_b, population)
    return compare(adapter, before, after, label)


def attributed_swap_set(
    flow_code: str, population: list[dict], increments: list[tuple[str, Callable[[dict], dict]]],
    *, config_version: str = "0.1.0", mode: str = "interpreted",
) -> list[SwapSetReport]:
    """§5.5's "a release of n changes supports n+1 runs... each increment's swap set
    measured separately". `increments` is `(label, transform)` pairs applied cumulatively
    to the built flow's own params document (a params change, an overlay-stack toggle, or
    any other params-shaped increment); each report compares against the run immediately
    before it, in the declared order, so a Credit Committee question -- "which one of
    these cost us the movement" -- has a per-increment answer, not one figure for the
    whole release."""
    adapter = flows.get(flow_code)
    built = adapter.build(config_version, mode=mode)
    baseline = _score_population(adapter, built, population)
    reports = [compare(adapter, baseline, baseline, "baseline (no change)")]
    current_params, previous_scores = built.params, baseline
    for label, transform in increments:
        current_params = transform(current_params)
        stepped = BuiltFlow(code=built.code, executable=built.executable, params=current_params,
                             config_version=built.config_version)
        new_scores = _score_population(adapter, stepped, population)
        reports.append(compare(adapter, previous_scores, new_scores, label))
        previous_scores = new_scores
    return reports
