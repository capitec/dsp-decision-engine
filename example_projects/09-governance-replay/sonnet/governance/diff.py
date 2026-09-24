"""Version diff (spec 09 §5.4): semantic, not textual.

Three shapes recur across the eight flows, and each gets its own diff, keyed
by content rather than position (§5.4's own requirement -- "reordering the
rows of a rate card spreadsheet... must produce an empty or near-empty change
record"):

- a `DecisionTableConfig` document (a rate card, a risk-grade table, a cap
  table...): rows keyed by their match-key columns, diffed cell by cell.
  Reuses `credit_core.rate_card.diff_cards` directly where the value column
  is literally `rate` (a rate card); `diff_table_rows` below generalises the
  same idea to any value column, for every other table shape.
- a `TreeConfig` `prioritized_flat_rule` document (project 01's rule set):
  rules keyed by `meta.name` (the `rule_id`), diffed by whether the rule's
  condition tree changed.
- a params document: a plain nested dict, diffed key path by key path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CellChange:
    key: tuple
    change: str          # "added" | "removed" | "repriced"
    before: object | None
    after: object | None


def diff_table_rows(old_rows: list[dict], new_rows: list[dict], key_fields: tuple[str, ...],
                     value_field: str) -> list[CellChange]:
    """The generalisation of `credit_core.rate_card.diff_cards` (project 00) to any
    `DecisionTableConfig`'s value column, not only a rate card's `rate`."""

    def key(row: dict) -> tuple:
        return tuple(row[k] for k in key_fields)

    before_by_key = {key(r): r for r in old_rows}
    after_by_key = {key(r): r for r in new_rows}
    changes = []
    for k in before_by_key.keys() | after_by_key.keys():
        before, after = before_by_key.get(k), after_by_key.get(k)
        if before is None:
            changes.append(CellChange(k, "added", None, after[value_field]))
        elif after is None:
            changes.append(CellChange(k, "removed", before[value_field], None))
        elif before[value_field] != after[value_field]:
            changes.append(CellChange(k, "repriced", before[value_field], after[value_field]))
    return changes


def summarize_table_diff(changes: list[CellChange], total_cells: int, group_field_index: int | None = None) -> dict:
    """Spec 09 §5.4's own worked example: "1 840 of 63 360 cells changed... mean move
    +0.31... no cell moved down" -- an aggregation, not a row list, because a list of
    1 840 rows "trains reviewers to approve without reading" (§5.4)."""
    repriced = [c for c in changes if c.change == "repriced"]
    moves = [c.after - c.before for c in repriced if isinstance(c.before, (int, float))]
    summary = {
        "total_cells": total_cells,
        "changed_cells": len(changes),
        "added": sum(1 for c in changes if c.change == "added"),
        "removed": sum(1 for c in changes if c.change == "removed"),
        "repriced": len(repriced),
        "mean_move": (sum(moves) / len(moves)) if moves else 0.0,
        "max_move": max(moves, key=abs) if moves else 0.0,
        "any_moved_down": any(m < 0 for m in moves),
    }
    if group_field_index is not None:
        by_group: dict = {}
        for c in repriced:
            by_group.setdefault(c.key[group_field_index], []).append(c)
        summary["by_group"] = {g: len(cs) for g, cs in by_group.items()}
    return summary


@dataclass(frozen=True)
class RuleChange:
    rule_id: str
    change: str  # "added" | "removed" | "reworded"


def diff_rule_set(old_doc: dict, new_doc: dict) -> list[RuleChange]:
    """A `TreeConfig` `prioritized_flat_rule` document's rules, keyed by `meta.name`
    (the `rule_id`) -- inserting a rule mid-priority must not renumber every rule after
    it (09 §5.15 item 2), so this diffs by id, never by list position."""
    old_rules = {r["meta"]["name"]: r["rule"] for r in old_doc["tree"]["rules"]}
    new_rules = {r["meta"]["name"]: r["rule"] for r in new_doc["tree"]["rules"]}
    changes = []
    for rid in old_rules.keys() | new_rules.keys():
        before, after = old_rules.get(rid), new_rules.get(rid)
        if before is None:
            changes.append(RuleChange(rid, "added"))
        elif after is None:
            changes.append(RuleChange(rid, "removed"))
        elif before != after:
            changes.append(RuleChange(rid, "reworded"))
    return changes


def diff_params(before: dict, after: dict, path: str = "") -> list[dict]:
    """A params document's own nested-dict diff, key path by key path (e.g.
    `fraud_interdiction/_stack_enabled/adjustment_stack_enabled`)."""
    changes = []
    for key in sorted(set(before) | set(after)):
        p = f"{path}/{key}" if path else key
        b, a = before.get(key), after.get(key)
        if isinstance(b, dict) and isinstance(a, dict):
            changes.extend(diff_params(b, a, p))
        elif b != a:
            changes.append({"path": p, "before": b, "after": a})
    return changes


def find_param(params: dict, name: str) -> list[str]:
    """Every key path in a params document literally named `name` -- the same
    recursive-walk pattern 03's own `tests/test_pipeline.py` uses to flip
    `adjustment_stack_enabled` regardless of which of its four overlay points
    it belongs to, generalised here for any param name and any flow."""
    hits = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}/{k}" if path else k
                if k == name and not isinstance(v, dict):
                    # a leaf value named `name` -- a step happening to share its own name
                    # (e.g. `evaluation_ceiling_per_term`'s step *and* its one param) must
                    # not match at the step-dict level, only at the actual value.
                    hits.append(p)
                walk(v, p)

    walk(params, "")
    return hits


def load_config_document(path: Path) -> dict:
    return json.loads(Path(path).read_text())
