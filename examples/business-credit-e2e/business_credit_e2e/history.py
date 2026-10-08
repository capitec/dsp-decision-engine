"""What a facility remembers: its decisions of record, and what the Bank knew about its entities and when."""
from __future__ import annotations

from datetime import date

from business_credit_e2e import vocab


class DecisionStore:
    """An append-only list of decisions of record per facility, in memory.

    Example: `store.append(decision_of_record(request, result, assessment_kind_code=1, comparison_basis_code=0))`
    returns the record with its `decision_of_record_id`; `store.latest(facility_id)` is the predecessor of the next one.
    """

    def __init__(self):
        self._records: dict[int, list[dict]] = {}
        self._count = 0

    def append(self, record: dict) -> dict:
        self._count += 1
        record = {**record, "decision_of_record_id": self._count}
        self._records.setdefault(record["facility_id"], []).append(record)
        return record

    def latest(self, facility_id: int) -> dict | None:
        return self._records.get(facility_id, [None])[-1]

    def history(self, facility_id: int) -> list[dict]:
        return list(self._records.get(facility_id, ()))


def decision_of_record(
    request: dict, result: dict, *, assessment_kind_code: int, comparison_basis_code: int,
    predecessor: dict | None = None, **extra,
) -> dict:
    """The decision of a scored `request` as a record: what a later review needs to compare itself with.

    It keeps the request as scored, so the previous decision can be run again; `predecessor` links the chain.
    """
    return {
        "decision_id": request.get("decision_id"), "facility_id": request["facility_id"],
        "assessment_kind_code": assessment_kind_code, "decision_date": request["decision_date"],
        "knowledge_date": request.get("knowledge_date", request["decision_date"]),
        "predecessor_decision_of_record_id": predecessor["decision_of_record_id"] if predecessor else None,
        "comparison_basis_code": comparison_basis_code, "request": request,
        "product_code": request["product_code"], "outcome_code": result["outcome_code"],
        "risk_grade": result["risk_grade"], "master_scale_version": result["master_scale_version"],
        "probability_of_default": result["probability_of_default"],
        "probability_of_default_unadjusted": result["probability_of_default_unadjusted"],
        "offered_amount": result["offered_amount"], "nominal_annual_rate": result["nominal_annual_rate"],
        "covenant_definition_version": result["covenant_definition_version"], **extra,
    }


def grade_migration(previous: dict, current: dict, previous_rescored: float, business_only: float, with_structure: float) -> dict:
    """Why the probability of default moved between two decisions, by cause. The causes add up to the movement.

    The other arguments are probabilities before overlays from three runs at the current decision date and policy:
    `previous_rescored` is the previous request as it was, `business_only` has the current financials, `with_structure`
    also the current entities' ownership and roles (events and bureau data still the previous). What is left to reach
    the current probability is the entities' own data. Example: a flat year with the policy changed gives a `model`
    cause and nothing else.
    """
    def overlay_effect(record: dict) -> float:
        return record["probability_of_default"] - record["probability_of_default_unadjusted"]

    causes = {
        "model": previous_rescored - previous["probability_of_default_unadjusted"],
        "business_data": business_only - previous_rescored,
        "entity_structure": with_structure - business_only,
        "entity_data": current["probability_of_default_unadjusted"] - with_structure,
        "overlay": overlay_effect(current) - overlay_effect(previous),
    }
    scale_changed = previous["master_scale_version"] != current["master_scale_version"]
    return {
        "grade_previous": previous["risk_grade"], "grade_current": current["risk_grade"],
        "observed_pd_movement": current["probability_of_default"] - previous["probability_of_default"],
        "causes": causes, "master_scale_changed": scale_changed,
        "comparison_basis_code": vocab.COMPARISON_NOT_COMPARABLE if scale_changed else vocab.COMPARISON_AS_GRADED,
    }


def entity_fact(entity_id: int, kind: str, value: dict, effective_from: date, effective_to: date | None, known_from: date) -> dict:
    """One thing the Bank holds about an entity: true from `effective_from` to `effective_to`, learned on `known_from`."""
    return {"entity_id": entity_id, "fact_kind": kind, "value": value, "effective_from": effective_from,
            "effective_to": effective_to, "known_from": known_from}


def query_effective(facts: list[dict], as_of: date) -> list[dict]:
    """What was true on `as_of`, whenever the Bank learned it (an ownership covenant's question)."""
    return [f for f in facts if f["effective_from"] <= as_of and (f["effective_to"] is None or as_of < f["effective_to"])]


def query_knowledge(facts: list[dict], as_of: date) -> list[dict]:
    """What the Bank held on `as_of`, corrections learned later left out (the question when reproducing a past decision).

    Per entity and kind, the latest fact learned by then wins. Example: a director who resigned in March but was
    recorded as resigned in November is still a director in a view as of June.
    """
    best: dict[tuple, dict] = {}
    for fact in sorted((f for f in facts if f["known_from"] <= as_of), key=lambda f: f["known_from"]):
        best[(fact["entity_id"], fact["fact_kind"])] = fact
    return list(best.values())
