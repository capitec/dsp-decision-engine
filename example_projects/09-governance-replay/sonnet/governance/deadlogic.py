"""Coverage and dead-logic detection (spec 09 §5.8).

Over a batch of decisions -- a production month's full volume in the real
estate; here, whatever a demo population produces -- counts how often each
named piece of logic (a live rule, a cap-waterfall rule, a roll-up rule)
actually bound, and flags the two failure modes §5.8 names: **dead** (zero
firings) and **dominant** (fires on more than 40% of traffic -- "not a rule,
... a policy someone wrote as a rule"). Each flow's own evidence already
carries what fired (01's `fired_rule_ids`, 03's `waterfall_rule_ids` +
`waterfall_rule_status`, 05's `attr_rule_id`); this module only counts it
against each flow's declared rule universe, never re-derives firing itself.
"""
from __future__ import annotations

from dataclasses import dataclass

from governance import flows, paths

DOMINANT_THRESHOLD = 0.40


@dataclass(frozen=True)
class CoverageReport:
    flow_code: str
    component: str
    population_size: int
    universe_size: int
    counts: dict            # {rule_id: firing_count}, includes any hit outside the declared universe
    dead: tuple[str, ...]      # in the universe, zero firings in this window
    dominant: tuple[str, ...]  # fires on more than DOMINANT_THRESHOLD of the population


def _ensure_on_path(flow_code: str) -> None:
    adapter = flows.get(flow_code)
    for extra in adapter.extra_projects:
        paths.ensure_on_path(paths.sibling_project(extra))
    paths.ensure_on_path(adapter.project_dir())


def rule_universe(flow_code: str) -> set[str]:
    """Every rule id this flow governs, read from its own rule catalog/register --
    never regenerated or guessed at by this module."""
    _ensure_on_path(flow_code)
    if flow_code == "01":
        from fraud_interdiction.rules import build_rule_catalog
        return {r.rule_id for r in build_rule_catalog().rules if r.status == "live"}
    if flow_code == "03":
        from loan_granting.waterfall import REGISTER
        return {r["rule_id"] for r in REGISTER}
    if flow_code == "05":
        from business_nested import rollup
        return {getattr(rollup, name) for name in dir(rollup) if name.startswith("AE_R_")}
    raise KeyError(f"no rule universe known for flow {flow_code!r}")


def _hit_ids(flow_code: str, record: dict) -> set[str]:
    """Which rule ids this one decision counts as "fired" (03: bound, raised or forced
    the decline -- `evaluated_did_not_bind` is a real evaluation, but not a firing)."""
    if flow_code == "01":
        return set(record.get("fired_rule_ids") or [])
    if flow_code == "03":
        ids = record.get("waterfall_rule_ids") or []
        statuses = record.get("waterfall_rule_status") or []
        return {rid for rid, status in zip(ids, statuses) if status in ("bound", "raised", "declined")}
    if flow_code == "05":
        return set(record.get("attr_rule_id") or [])
    raise KeyError(f"no firing field known for flow {flow_code!r}")


def rule_coverage(flow_code: str, records: list[dict], component: str = "rules") -> CoverageReport:
    universe = rule_universe(flow_code)
    counts: dict[str, int] = {rid: 0 for rid in universe}
    for record in records:
        for rid in _hit_ids(flow_code, record):
            counts[rid] = counts.get(rid, 0) + 1
    n = len(records)
    dead = tuple(sorted(rid for rid in universe if counts.get(rid, 0) == 0))
    dominant = tuple(sorted(rid for rid, c in counts.items() if n and c / n > DOMINANT_THRESHOLD))
    return CoverageReport(flow_code=flow_code, component=component, population_size=n,
                           universe_size=len(universe), counts=counts, dead=dead, dominant=dominant)


def table_cell_coverage(records: list[dict], cell_field: str, total_cells: int) -> dict:
    """§5.8's "never-read table cells", measured but not thresholded (§5.6's own caveat: a
    large card cannot be meaningfully covered by a demo-scale population -- this reports
    the fraction, and makes the question askable, exactly as §5.8 asks of it)."""
    read = {r[cell_field] for r in records if r.get(cell_field)}
    return {
        "cell_field": cell_field, "cells_read": len(read), "total_cells": total_cells,
        "pct_read": (len(read) / total_cells) if total_cells else 0.0,
    }
