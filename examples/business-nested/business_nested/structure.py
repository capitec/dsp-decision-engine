"""Structure: the disclosed paths to each person become one entity, ready to be assessed on its own.

A person reached by two paths (a director who also holds shares) arrives as two records. Everything
below the application level works on one entity at a time, so the merge happens first, and the context an
entity's own rules need (decision date, sector, requested amount) is copied onto it: a child flow sees only
the fields of its own item.
"""
from __future__ import annotations

from datetime import date
from typing import TypedDict

from decider import allow_fallback, param, step

from business_nested import vocab


class AdverseEvent(TypedDict, total=False):
    event_id: int
    event_type_code: int
    event_date: date
    amount: float  # null is not zero: a judgment with no recorded amount is of unknown size
    status_is_active: bool
    is_disputed: bool
    is_satisfied: bool


class Entity(TypedDict, total=False):
    entity_id: int
    entity_key: str  # the identity the paths of one person share
    is_natural_person: bool
    relationship_type_code: int
    effective_ownership_pct: float  # of this path; the product of the percentages along it
    is_controlling: bool
    is_required_surety: bool
    is_sole_principal: bool
    bureau_score: float
    months_on_record: float
    worst_delinquency_months: float
    adverse_events: list[AdverseEvent]


def criticality_class(
    effective_ownership_pct: float, is_controlling: bool, is_required_surety: bool, is_sole_principal: bool,
    relationship_type_code: int, critical_ownership_pct: float, significant_ownership_pct: float,
) -> int:
    """How much an entity matters to the business: control, a large stake or a required surety make it critical."""
    if is_controlling or is_required_surety or is_sole_principal or effective_ownership_pct >= critical_ownership_pct:
        return vocab.CRITICAL
    if effective_ownership_pct >= significant_ownership_pct or relationship_type_code in vocab.NON_OWNING_CONTROLLER_ROLES:
        return vocab.SIGNIFICANT
    return vocab.PERIPHERAL


def _merge(paths: list[dict]) -> dict:
    # Most senior path first, so the facts taken from it do not depend on the order of the request.
    paths = sorted(paths, key=lambda p: (vocab.role_rank(p["relationship_type_code"]), -(p.get("effective_ownership_pct") or 0.0)))
    return {
        **paths[0],
        "entity_id": min(p["entity_id"] for p in paths),
        "effective_ownership_pct": round(sum(p.get("effective_ownership_pct") or 0.0 for p in paths), 4),
        "is_controlling": any(p.get("is_controlling") for p in paths),
        "is_required_surety": any(p.get("is_required_surety") for p in paths),
        "is_sole_principal": any(p.get("is_sole_principal") for p in paths),
        # The displayed role is the most senior one, which must not hide that the person also owns shares.
        "is_owner": any(p["relationship_type_code"] in vocab.OWNING_ROLES for p in paths),
        "path_count": len(paths),
        "adverse_events": list({e["event_id"]: e for p in paths for e in p.get("adverse_events") or []}.values()),
    }


@allow_fallback
def resolved_entities(
    entities: list[Entity], decision_date: date, sector_code: int, requested_amount: float,
    critical_ownership_pct: float = param(25.0, ge=0.0, le=100.0),
    significant_ownership_pct: float = param(10.0, ge=0.0, le=100.0),
) -> list[dict]:
    """One record per person, sorted by `entity_id`, with its events sorted by `event_id`.

    Ownership adds up across paths, the role is the most senior one, control is any path's, and an event
    reported on two paths counts once. Each entity and event carries the context its rules read.
    """
    by_key: dict[str, list[dict]] = {}
    for raw in entities or []:
        by_key.setdefault(raw["entity_key"], []).append(raw)
    resolved = []
    for paths in by_key.values():
        e = _merge(paths)
        klass = criticality_class(
            e["effective_ownership_pct"], e["is_controlling"], e["is_required_surety"], e["is_sole_principal"],
            e["relationship_type_code"], critical_ownership_pct, significant_ownership_pct)
        events = [{
            "event_id": ev["event_id"], "event_type_code": ev["event_type_code"], "event_date": ev["event_date"],
            "amount": ev.get("amount"), "status_is_active": ev.get("status_is_active", True),
            "is_disputed": ev.get("is_disputed", False), "is_satisfied": ev.get("is_satisfied", False),
            "decision_date": decision_date, "sector_code": sector_code, "criticality_class": klass,
        } for ev in sorted(e["adverse_events"], key=lambda ev: ev["event_id"])]
        resolved.append({
            "entity_id": e["entity_id"], "entity_key": e["entity_key"],
            "is_natural_person": bool(e.get("is_natural_person", True)),
            "relationship_type_code": e["relationship_type_code"],
            "effective_ownership_pct": e["effective_ownership_pct"], "is_controlling": bool(e["is_controlling"]),
            "is_required_surety": bool(e["is_required_surety"]), "is_sole_principal": bool(e["is_sole_principal"]),
            "is_owner": e["is_owner"], "path_count": e["path_count"], "criticality_class": klass,
            "bureau_score": e.get("bureau_score"), "months_on_record": e.get("months_on_record"),
            "worst_delinquency_months": e.get("worst_delinquency_months"),
            "decision_date": decision_date, "requested_amount": requested_amount, "adverse_events": events,
        })
    return sorted(resolved, key=lambda e: e["entity_id"])


@allow_fallback
@step(outputs=("resolved_entity_count", "ownership_reconciliation_pct", "structure_unresolved"))
def structure_check(
    assessed_entities: list[dict],
    max_entities: int = param(40, ge=1), ownership_floor_pct: float = param(90.0, ge=0.0),
    ownership_ceiling_pct: float = param(110.0, ge=0.0),
) -> tuple[int, float, bool]:
    """How many entities there are, what the owners' ownership adds up to, and whether that can be trusted.

    A structure is unresolved with too many entities, or ownership that does not come to about 100: too
    little means a holder is missing, too much means a person was counted twice.
    """
    count = len(assessed_entities)
    owned = round(sum(e["effective_ownership_pct"] for e in assessed_entities if e["is_owner"]), 4)
    return count, owned, not (count <= max_entities and ownership_floor_pct <= owned <= ownership_ceiling_pct)
