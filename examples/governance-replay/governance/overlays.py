"""The estate-wide overlay register, its ageing, and the run with every overlay switched off."""
from __future__ import annotations

import copy
import pkgutil
from datetime import date

from governance.evidence import Evidence
from governance.flows import FLOWS, Flow

SWITCHES = ("adjustment_stack_enabled", "event_overlays_enabled")


def _adjustments(flow: Flow):
    for path in flow.registers:
        # workaround: AdjustmentRegister has no public iterator; `_all` is its list.
        yield from pkgutil.resolve_name(path)._all


def _rows(flows, keep=lambda a: True) -> list[dict]:
    return [{"flow": code, "adjustment_id": a.adjustment_id, "kind": a.kind, "target": a.target, "owner": a.owner,
             "approval_reference": a.approval_reference, "effective_from": a.effective_from,
             "effective_to": a.effective_to, "review_date": a.review_date, "tighten_only": a.tighten_only}
            for code in flows for a in _adjustments(FLOWS[code]) if keep(a)]


def register(flows=FLOWS) -> list[dict]:
    """One row per overlay across the flows: id, target, owner, approval, dates, tighten-only.

        [r["adjustment_id"] for r in register() if r["flow"] == "loan"]
    """
    return _rows(flows)


def ageing(on: date, flows=FLOWS) -> list[dict]:
    """The overlays past their review date and not lapsed on `on`, oldest review first, with their age in days.

        ageing(date(2026, 10, 1))[0]["age_days"]
    """
    due = _rows(flows, lambda a: a.due_for_review(on))
    return sorted(({**r, "age_days": (on - r["review_date"]).days} for r in due), key=lambda r: -r["age_days"])


def switched_off(params: dict) -> dict:
    """A copy of `params` with every overlay switch off."""
    out = copy.deepcopy(params)

    def walk(node):
        for key, value in node.items():
            if key in SWITCHES and isinstance(value, bool):
                node[key] = False
            elif isinstance(value, dict):
                walk(value)

    walk(out)
    return out


def unwind_estimate(evidences: list[Evidence]) -> dict:
    """Run each decision again through the same flow with the overlays off: how many outcomes would change.

        unwind_estimate(store.all("loan"))   # {"decisions": 9, "changed": 1, "moves": {"id": (1, 4)}}
    """
    moves = {}
    for e in evidences:
        flow = FLOWS[e.flow]
        off = flow.replayer(e.config_version, e.mode, params=switched_off(e.params)).score(e.request)
        if off[flow.outcome] != e.response[flow.outcome]:
            moves[e.decision_id] = (e.response[flow.outcome], off[flow.outcome])
    return {"decisions": len(evidences), "changed": len(moves), "moves": moves}
