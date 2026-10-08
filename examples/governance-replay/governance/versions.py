"""What changed between two versions of a flow: logic, params and each config document."""
from __future__ import annotations

from dataclasses import dataclass

from credit_core.rate_card import diff_cards
from decider.lab import diff

from governance.flows import Flow
from governance.replayer import Replayer


@dataclass(frozen=True)
class VersionDiff:
    """`structure` is decider's step-level diff and `proof` its verdict on whether the steps are provably the same
    (`True`, or `None` when it cannot tell, which is always so here: many steps read globals it can only key by
    identity, such as overlay registers, dates and dicts); `params` lists changed leaves; `documents` summarises each
    changed config document."""

    structure: diff.StructureDiff
    proof: bool | None
    params: tuple[dict, ...]
    documents: dict[str, dict]


def _leaves(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, f"{path}/{k}")
    else:
        yield path, node


def _rate_card(old: dict, new: dict) -> dict:
    changes = diff_cards(old["rows"], new["rows"])
    moves = [c["delta"] for c in changes if c["change"] == "repriced"]
    return {"cells": len(old["rows"]), "changed": len(changes), "mean_move": sum(moves) / len(moves) if moves else 0.0,
            "any_down": any(m < 0 for m in moves)}


def _rule_set(old: dict, new: dict) -> dict:
    a, b = ({r["rule_id"]: r for r in doc["rules"]} for doc in (old, new))
    return {"added": sorted(b.keys() - a.keys()), "removed": sorted(a.keys() - b.keys()),
            "changed": sorted(k for k in a.keys() & b.keys() if a[k] != b[k])}


def version_diff(flow: Flow, old: Replayer, new: Replayer) -> VersionDiff:
    """Compare two staged versions without running either.

    A reordered rule set or rate card gives an empty document diff: rows are matched by key, never by position.

        version_diff(LOAN, flow.replayer("0.1.0"), candidate).documents["rate_card_flex_loan"]["changed"]
    """
    configs = [r.store.read(r.active).config for r in (old, new)]
    steps = [r.pipeline_fn(c) for r, c in zip((old, new), configs)]
    params = [dict(_leaves(r.params if r.params is not None else c.get("params", {}))) for r, c in zip((old, new), configs)]
    changed = tuple({"path": p, "old": params[0].get(p), "new": params[1].get(p)}
                    for p in sorted(params[0].keys() | params[1].keys()) if params[0].get(p) != params[1].get(p))
    documents = {}
    for name in sorted(configs[0].keys() & configs[1].keys() - {"params"}):
        a, b = configs[0][name], configs[1][name]
        if a != b:
            kind = a.get("type")
            documents[name] = (_rule_set(a, b) if kind == "rule_set" else
                               _rate_card(a, b) if kind == "decision_table" and "rate" in a["outputs"] else
                               {"changed": True})
    return VersionDiff(diff.structure(*steps), diff.prove(*steps).identical, changed, documents)
