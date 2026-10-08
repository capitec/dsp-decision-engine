"""Swap sets: who would have been decided differently under another version, and which change did it."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from governance.flows import Flow
from governance.replayer import Replayer


@dataclass(frozen=True)
class SwapSet:
    label: str
    population: int
    moved: tuple[tuple[str, object, object], ...]  # (decision_id, outcome before, outcome after)
    reasons_appeared: dict
    reasons_disappeared: dict
    reasons_only: tuple[str, ...]                  # outcome unchanged, reasons changed
    uncomparable: dict                             # decision_id -> why


def _score(replayer: Replayer, population: list[dict]) -> dict:
    out = {}
    for request in population:
        try:
            out[request["decision_id"]] = replayer.score(request)
        except Exception as e:  # one bad record must not hide the rest
            out[request["decision_id"]] = e
    return out


def _compare(flow: Flow, label: str, before: dict, after: dict) -> SwapSet:
    moved, only, appeared, gone, bad = [], [], Counter(), Counter(), {}
    for id_, b in before.items():
        a = after[id_]
        if isinstance(a, Exception) or isinstance(b, Exception):
            bad[id_] = str(a if isinstance(a, Exception) else b)
            continue
        if a[flow.outcome] != b[flow.outcome]:
            moved.append((id_, b[flow.outcome], a[flow.outcome]))
        elif a[flow.reasons] != b[flow.reasons]:
            only.append(id_)
        appeared.update(set(a[flow.reasons]) - set(b[flow.reasons]))
        gone.update(set(b[flow.reasons]) - set(a[flow.reasons]))
    return SwapSet(label, len(before), tuple(moved), dict(appeared), dict(gone), tuple(only), bad)


def swap_set(flow: Flow, population: list[dict], before: Replayer, after: Replayer, label: str = "") -> SwapSet:
    """Score `population` (raw requests) under two staged versions and report who moved.

        swap_set(LOAN, requests, current, candidate).moved   # (("id", 1, 4), ...)
    """
    return _compare(flow, label, _score(before, population), _score(after, population))


def attribute(flow: Flow, population: list[dict], runs: list[tuple[str, Replayer]]) -> list[SwapSet]:
    """n changes, n+1 runs: `runs` is the base then each change applied cumulatively, in the declared order.

    Each swap set compares one run with the run before it, so the answer to "which change cost us these
    approvals" is per change, not one figure for the release. A change may be a new rate card, new rules, or
    params (see `Flow.candidate`, `Flow.replayer(params=...)`).

        attribute(LOAN, requests, [("base", base), ("rate card refresh", step1), ("overlay off", step2)])
    """
    scored = [(label, _score(r, population)) for label, r in runs]
    return [_compare(flow, scored[i + 1][0], scored[i][1], scored[i + 1][1]) for i in range(len(scored) - 1)]
