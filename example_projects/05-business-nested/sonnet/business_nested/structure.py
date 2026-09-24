"""Stage 1 -- structure resolution (spec 05 §5.1) and stage 2 -- entity
criticality (spec 05 §5.4's criticality table only; the 14-rule entity
disqualification matrix itself is out of this slice, per SCOPE.md).

**Framework note (spec 05 §13 Q1, Q12).** The request carries the outer
collection (entities) with the inner collection (each entity's adverse
events) nested inside it -- ordinary JSON, and ordinary arrow/polars
nested-struct types read it as an *input* without difficulty (verified: a
two-level `list[struct[list[struct]]]` top-level input scores and runs
fine). But no `decider` step -- `Step`, `FrameStep` or `ConfigurableStep`
-- can *emit* a nested (`list[struct]`, or `list[list[T]]`) output column;
both crash result materialisation (see NOTES.md "Framework friction" for
the two-line reproduction of each). So the outer-collection-of-inner-
collections shape can only exist on the way *in*. Every stage from here on
works in **long (relational) form**: this step unpacks the two nested
levels into three sets of flat, parallel, application-scoped lists --
entities, events, and (later, in `rollup.py`) attribution pairs -- joined
by `entity_id`/`event_id`, the same normalisation a database would apply
to a nested document. That answers spec 05 §13 Q1 (the unit that operates
on a nested collection is a `frame_step` at each level, not a "depth
parameter") and Q6 (a count-based attribution and a single-event
attribution are both just rows in the same flat pairs table -- the shape
does not vary by rule kind, only the row count does).

Ordering independence (acceptance §10 item 3) is handled here once: every
list this step emits is sorted by `entity_id` (entities) or by
`(entity_id, event_id)` (events), regardless of the order the request
supplied them in, so no later stage has to re-sort to be order-independent.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from decider import frame_step

from credit_core.adverse_events import event_age_months

from business_nested import vocab

MAX_RESOLVED_ENTITIES = 40
# spec 05 §5.1's depth bound (3) and expansion materiality floor (5.0%) are enforced by
# the caller, before this stage runs -- see the module docstring and NOTES.md "What I
# left out": the graph-to-bounded-structure traversal itself isn't built in this slice,
# so there is nothing here that would consume those two constants.
CRITICAL_OWNERSHIP_PCT = 25.0
SIGNIFICANT_OWNERSHIP_PCT = 10.0


def classify_criticality(
    effective_ownership_pct: float, is_controlling: bool, is_required_surety: bool, is_sole_principal: bool,
    relationship_type_code: int,
) -> int:
    """Spec 05 §5.4's criticality table (the corporate-guarantor->20%-cover clause is out
    of this slice's scope -- no security/cover model is built here; noted in NOTES.md)."""
    if is_controlling or effective_ownership_pct >= CRITICAL_OWNERSHIP_PCT or is_required_surety or is_sole_principal:
        return vocab.CRITICAL
    if SIGNIFICANT_OWNERSHIP_PCT <= effective_ownership_pct < CRITICAL_OWNERSHIP_PCT:
        return vocab.SIGNIFICANT
    if relationship_type_code in vocab.NON_OWNING_CONTROLLER_ROLES:
        return vocab.SIGNIFICANT
    return vocab.PERIPHERAL


def _dedup_key(raw: dict) -> str:
    return raw["entity_key"]


def _merge_duplicates(group: list[dict]) -> dict:
    """Spec 05 §5.1: same `entity_key` on two paths -> one entity. Ownership sums,
    role is the most senior, control is the disjunction, every path is retained."""
    first = group[0]
    total_ownership = sum(g.get("direct_ownership_pct") if g.get("direct_ownership_pct") is not None
                           else g.get("effective_ownership_pct", 0.0) or 0.0 for g in group)
    # Effective ownership is supplied by the caller per path (spec 05's own path-product
    # arithmetic is not re-derived here -- see NOTES.md "What I left out": full graph
    # traversal/cycle detection is a scope cut, the caller already resolves each disclosed
    # path to its own effective_ownership_pct before this stage runs).
    total_effective = sum(g.get("effective_ownership_pct", 0.0) or 0.0 for g in group)
    most_senior = min(group, key=lambda g: vocab.role_seniority_rank(g["relationship_type_code"]))
    merged = dict(first)
    merged["effective_ownership_pct"] = round(total_effective, 4)
    merged["relationship_type_code"] = most_senior["relationship_type_code"]
    merged["is_controlling"] = any(bool(g.get("is_controlling")) for g in group)
    merged["is_required_surety"] = any(bool(g.get("is_required_surety")) for g in group)
    merged["is_sole_principal"] = any(bool(g.get("is_sole_principal")) for g in group)
    merged["path_count"] = len(group)
    # An entity may reach the applicant through an owning path *and* a non-owning path
    # (a shareholder who is also a director) -- collapsing to the single most-senior
    # *displayed* role must not erase that it owns, or ownership reconciliation and the
    # people-blend's inclusion test (§5.8 PP-01/PP-02) silently lose the stake.
    merged["is_owner"] = any(g["relationship_type_code"] in vocab.OWNING_ROLES for g in group)
    return merged


def _dedup_events(raw_events: list[dict]) -> list[dict]:
    """An entity reached by two paths must not have its adverse events counted twice
    (spec 05 §5.1 "adverse events attached once, never once per path")."""
    seen: dict[int, dict] = {}
    for ev in raw_events:
        seen.setdefault(ev["event_id"], ev)
    return list(seen.values())


def resolve_entities(raw_entities: list[dict], decision_date) -> dict:
    """Pure function, unit-testable with no `decider` machinery (spec 00 §7.5's
    standalone-testability convention, followed here for a project-specific capability).

    Returns a dict of flat, parallel lists (entities, sorted by `entity_id`) plus the
    flattened, deduplicated event lists (sorted by `(entity_id, event_id)`).

    **Framework note.** `event_date` is converted to `ev_age_months` (an int) right here,
    inside the same `frame_step` invocation that still holds real `datetime.date` Python
    objects, and the raw date is never emitted as its own column. Reproduced: a
    `list[date]` column written by one `frame_step` and read by a *second* `frame_step`
    downstream silently degrades to `Array(Int64, ...)` (raw epoch-day integers) at the
    boundary between them -- no error, no warning, until something calls `.year` on an
    int. A third, distinct materialisation gap alongside the two 00/02 NOTES.md already
    document (`list[struct]` outputs, and empty/`None`-valued top-level fields) -- see
    NOTES.md "Framework friction" for the two-`frame_step` minimal reproduction.
    """
    by_key: dict[str, list[dict]] = {}
    for raw in raw_entities:
        by_key.setdefault(_dedup_key(raw), []).append(raw)

    resolved = []
    for key, group in by_key.items():
        merged = _merge_duplicates(group)
        merged["adverse_events"] = _dedup_events([ev for g in group for ev in (g.get("adverse_events") or [])])
        resolved.append(merged)

    # Stable order independent of request order: by entity_id (assigned by the caller,
    # unique within the application -- spec 05 §4.2).
    resolved.sort(key=lambda e: e["entity_id"])

    structure_unresolved = len(resolved) > MAX_RESOLVED_ENTITIES
    total_owner_ownership = sum(e["effective_ownership_pct"] for e in resolved if e["is_owner"])
    ownership_reconciled = 90.0 <= total_owner_ownership <= 110.0
    if not ownership_reconciled:
        structure_unresolved = True

    entity_ids, entity_keys, is_natural, rel_codes, eff_own, is_controlling = [], [], [], [], [], []
    is_surety, is_sole, path_counts, criticality, is_owner = [], [], [], [], []
    bureau_scores, months_on_record, worst_delinquency = [], [], []
    ev_entity_id, ev_event_id, ev_type, ev_amount, ev_age, ev_active, ev_disputed, ev_satisfied = (
        [], [], [], [], [], [], [], [])

    for e in resolved:
        cls = classify_criticality(
            e["effective_ownership_pct"], e["is_controlling"], e["is_required_surety"], e["is_sole_principal"],
            e["relationship_type_code"],
        )
        entity_ids.append(e["entity_id"])
        entity_keys.append(e["entity_key"])
        is_natural.append(bool(e.get("is_natural_person")))
        rel_codes.append(e["relationship_type_code"])
        eff_own.append(e["effective_ownership_pct"])
        is_controlling.append(e["is_controlling"])
        is_surety.append(e["is_required_surety"])
        is_sole.append(e["is_sole_principal"])
        path_counts.append(e["path_count"])
        criticality.append(cls)
        is_owner.append(e["is_owner"])
        bureau_scores.append(e.get("bureau_score"))
        months_on_record.append(e.get("months_on_record"))
        worst_delinquency.append(e.get("worst_delinquency_months"))

        for ev in sorted(e["adverse_events"], key=lambda x: x["event_id"]):
            ev_entity_id.append(e["entity_id"])
            ev_event_id.append(ev["event_id"])
            ev_type.append(ev["event_type_code"])
            ev_amount.append(ev.get("amount"))
            ev_age.append(event_age_months(ev["event_date"], decision_date))
            ev_active.append(bool(ev.get("status_is_active", True)))
            ev_disputed.append(bool(ev.get("is_disputed", False)))
            ev_satisfied.append(bool(ev.get("is_satisfied", False)))

    return {
        "resolved_entity_count": len(resolved),
        "structure_unresolved": structure_unresolved,
        "ownership_reconciliation_pct": round(total_owner_ownership, 4),
        "entity_id": entity_ids, "entity_key": entity_keys, "entity_is_natural_person": is_natural,
        "entity_relationship_type_code": rel_codes, "entity_effective_ownership_pct": eff_own,
        "entity_is_controlling": is_controlling, "entity_is_required_surety": is_surety,
        "entity_is_sole_principal": is_sole, "entity_path_count": path_counts,
        "entity_criticality_class": criticality, "entity_is_owner": is_owner,
        "entity_bureau_score": bureau_scores, "entity_months_on_record": months_on_record,
        "entity_worst_delinquency_months": worst_delinquency,
        "ev_entity_id": ev_entity_id, "ev_event_id": ev_event_id, "ev_event_type_code": ev_type,
        "ev_amount": ev_amount, "ev_age_months": ev_age, "ev_status_is_active": ev_active,
        "ev_is_disputed": ev_disputed, "ev_is_satisfied": ev_satisfied,
    }


_OUTPUT_COLUMNS = [
    "resolved_entity_count", "structure_unresolved", "ownership_reconciliation_pct",
    "entity_id", "entity_key", "entity_is_natural_person", "entity_relationship_type_code",
    "entity_effective_ownership_pct", "entity_is_controlling", "entity_is_required_surety",
    "entity_is_sole_principal", "entity_path_count", "entity_criticality_class", "entity_is_owner",
    "entity_bureau_score", "entity_months_on_record", "entity_worst_delinquency_months",
    "ev_entity_id", "ev_event_id", "ev_event_type_code", "ev_amount", "ev_age_months",
    "ev_status_is_active", "ev_is_disputed", "ev_is_satisfied",
]


@frame_step(reads=["entities", "decision_date"], writes=_OUTPUT_COLUMNS)
def resolve_structure(df: pl.DataFrame) -> pl.DataFrame:
    """Explodes the request's nested `entities` (each carrying nested `adverse_events`)
    into the flat, parallel, long-form columns every later stage reads (see module
    docstring). One row in, one row out -- the nesting is inside the row, not across rows."""
    results = [
        resolve_entities(row["entities"] or [], row["decision_date"])
        for row in df.select("entities", "decision_date").to_dicts()
    ]
    out = pl.DataFrame(results)
    return df.with_columns(out)
