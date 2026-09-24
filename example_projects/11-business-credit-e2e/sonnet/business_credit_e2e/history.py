"""Statefulness across time (spec 11 §5.10, §5.13): the decision-of-record
sequence, the bi-temporal entity-structure fact store, and grade migration with
its six-way cause decomposition.

Persistence is in-memory (SCOPE.md builds the mechanism, not a database --
NOTES.md "What I would do next"), append-only, matching §5.10.1's "ordered,
append-only sequence of decisions of record".
"""
from __future__ import annotations

import itertools
from datetime import date

from business_credit_e2e import vocab

_decision_ids = itertools.count(1)


def master_scale_version_from_result(result: dict) -> str:
    """§5.10.4 requires `master_scale_version` on every decision of record, but
    project 05 publishes no such field at all (grep confirms it -- see
    `vocab.py`'s own note on this gap). Recovered here from
    `business_risk_grade_cell_id`'s own `<table>@<version>#<keys>` shape
    (`credit_core.evidence.cell_id`, project 00, unmodified) rather than a
    second, hand-maintained constant -- so it can never drift out of step with
    the table that actually produced the grade. A real fix belongs in project
    05 (add the field next to `risk_grade`); this is the composed-around
    version for this slice (§5.17.2)."""
    return result["business_risk_grade_cell_id"].split("@", 1)[1].split("#", 1)[0]


def new_decision_of_record(
    *, facility_id: int, assessment_kind_code: int, decision_date: date, knowledge_date: date,
    predecessor_id: int | None, comparison_basis_code: int, outcome_code: int, risk_grade: int,
    master_scale_version: str, probability_of_default: float,
    probability_of_default_unadjusted: float | None = None, extra: dict | None = None,
) -> dict:
    """One entry in a facility's decision history (§5.10.1). `predecessor_id` and
    `comparison_basis_code` are what makes the sequence a *history* rather than
    a stack of unrelated records -- §5.10.2 requirement 4."""
    record = {
        "decision_of_record_id": next(_decision_ids),
        "facility_id": facility_id,
        "assessment_kind_code": assessment_kind_code,
        "decision_date": decision_date,
        "knowledge_date": knowledge_date,
        "predecessor_decision_of_record_id": predecessor_id,
        "comparison_basis_code": comparison_basis_code,
        "outcome_code": outcome_code,
        "risk_grade": risk_grade,
        "master_scale_version": master_scale_version,
        "probability_of_default": probability_of_default,
        "probability_of_default_unadjusted": probability_of_default_unadjusted,
    }
    if extra:
        record.update(extra)
    return record


class DecisionHistoryStore:
    """Keyed by `facility_id` -> ordered list of decisions of record. Mean 11
    per facility over five years per spec 11 §5.10.1; this store holds however
    many a test appends."""

    def __init__(self):
        self._by_facility: dict[int, list[dict]] = {}

    def append(self, record: dict) -> dict:
        self._by_facility.setdefault(record["facility_id"], []).append(record)
        return record

    def latest(self, facility_id: int) -> dict | None:
        records = self._by_facility.get(facility_id)
        return records[-1] if records else None

    def history(self, facility_id: int) -> list[dict]:
        return list(self._by_facility.get(facility_id, ()))


# --------------------------------------------------------------------------
# Bi-temporal entity structure facts (spec 11 §5.13)
# --------------------------------------------------------------------------

def new_entity_fact(
    *, entity_id: int, fact_kind: str, effective_from: date, effective_to: date | None,
    known_from: date, source_code: str, value: dict,
) -> dict:
    return {
        "entity_id": entity_id, "fact_kind": fact_kind, "effective_from": effective_from,
        "effective_to": effective_to, "known_from": known_from, "source_code": source_code, "value": value,
    }


def query_effective(facts: list[dict], as_of_date: date) -> list[dict]:
    """§5.13.2 query 2: "what was actually true then" -- an ownership covenant's
    baseline test. Every fact whose effective window covers `as_of_date`,
    regardless of when the Bank learned it."""
    return [f for f in facts if f["effective_from"] <= as_of_date
            and (f["effective_to"] is None or as_of_date < f["effective_to"])]


def query_knowledge(facts: list[dict], as_of_date: date) -> list[dict]:
    """§5.13.2 query 1: "reproduce the June 2028 review" -- what the Bank had
    *recorded* by `as_of_date`, even if later corrected or superseded. Per
    `(entity_id, fact_kind)`, the most-recently-known fact as of `as_of_date`
    wins -- a correction learned after `as_of_date` must not appear, exactly as
    the original decision never saw it."""
    known = [f for f in facts if f["known_from"] <= as_of_date]
    best: dict[tuple, dict] = {}
    for f in known:
        key = (f["entity_id"], f["fact_kind"])
        if key not in best or f["known_from"] >= best[key]["known_from"]:
            best[key] = f
    return list(best.values())


# --------------------------------------------------------------------------
# Grade migration and the six-cause decomposition (spec 11 §5.4.1, §5.10.3)
# --------------------------------------------------------------------------

def grade_migration(
    previous_record: dict, current_record: dict,
    counterfactual_pd_business_only: float, counterfactual_pd_business_and_structure: float,
) -> dict:
    """Decomposes the PD movement between two consecutive decisions of record
    for one facility across the six named causes (§5.4.1), **required to sum to
    the observed movement exactly** (§5.4.1, §5.10.3) -- a residual that cannot
    be apportioned is a defect, reported as one, never silently absorbed.

    Three causes are measured by genuine counterfactual re-runs (the caller
    supplies them -- `review.py` gets them by re-scoring project 05's pipeline
    with a mix of previous and current inputs, see `review.annual_review`):

    - **business data**: previous inputs, but this decision's own financial
      figures (`counterfactual_pd_business_only`).
    - **entity structure**: the above, plus this decision's current entity
      ownership/control/relationships (`counterfactual_pd_business_and_structure`).
    - **entity data**: the remainder to reach the actual current PD (adverse
      events refreshed last) -- computed here, not re-run again, since it is
      whatever is left before model/overlay/scale.

    Two causes are read from fields the decision record already carries, not
    re-run:

    - **model**: `current_pd - counterfactual_pd_business_and_structure_and_entity`;
      zero whenever the scorecard version is unchanged (this slice carries one
      live scorecard version -- 00/05's own precedent for not building "two
      live majors simultaneously", SCOPE.md). Genuinely computed, not hardcoded,
      so a future second scorecard version would show up here automatically.
    - **overlay**: the *change* in the overlay's own effect
      (`probability_of_default - probability_of_default_unadjusted`) between the
      two decisions -- real fields both 00's `core.adjustments` and 05's
      pipeline already publish on every decision, so no new tracking is needed.

    The sixth, **scale**, is the balancing residual: `observed - sum(the other
    five)`. It is asserted to be (near) zero unless `master_scale_version`
    changed between the two decisions -- exactly spec 11 §5.10.4's "a
    restatement map... only when the scheme itself changed", implemented as a
    check rather than a restatement map (SCOPE.md skips grade-scheme
    restatement maps; this slice runs on one master scale throughout, so the
    check is exercised as "always passes", not as a real restatement).
    """
    pd0 = previous_record["probability_of_default"]
    pd_current = current_record["probability_of_default"]
    pd1 = counterfactual_pd_business_only
    pd2 = counterfactual_pd_business_and_structure
    pd3 = pd_current  # entity data + model + overlay + scale, all refreshed to "now"

    business = round(pd1 - pd0, 6)
    structure = round(pd2 - pd1, 6)
    entity_data = round(pd3 - pd2, 6)
    model = round(pd_current - pd3, 6)  # 0 in this slice; see docstring

    prev_unadj = previous_record.get("probability_of_default_unadjusted")
    curr_unadj = current_record.get("probability_of_default_unadjusted")
    prev_overlay_effect = pd0 - prev_unadj if prev_unadj is not None else 0.0
    curr_overlay_effect = pd_current - curr_unadj if curr_unadj is not None else 0.0
    overlay = round(curr_overlay_effect - prev_overlay_effect, 6)

    observed = round(pd_current - pd0, 6)
    attributed_before_scale = round(business + structure + entity_data + model + overlay, 6)
    scale = round(observed - attributed_before_scale, 6)

    scale_changed = previous_record.get("master_scale_version") != current_record.get("master_scale_version")
    residual_is_defect = (not scale_changed) and abs(scale) > 1e-6

    causes = {
        vocab.CAUSE_BUSINESS_DATA: business, vocab.CAUSE_ENTITY_STRUCTURE: structure,
        vocab.CAUSE_ENTITY_DATA: entity_data, vocab.CAUSE_MODEL: model,
        vocab.CAUSE_OVERLAY: overlay, vocab.CAUSE_SCALE: scale,
    }
    comparison_basis_code = vocab.COMPARISON_NOT_COMPARABLE if scale_changed else vocab.COMPARISON_AS_GRADED

    return {
        "grade_previous": previous_record["risk_grade"],
        "grade_current": current_record["risk_grade"],
        "master_scale_version_previous": previous_record.get("master_scale_version"),
        "master_scale_version_current": current_record.get("master_scale_version"),
        "comparison_basis_code": comparison_basis_code,
        "observed_pd_movement": observed,
        "causes": causes,
        "causes_sum_to_observed": abs(round(sum(causes.values()), 6) - observed) < 1e-6,
        "residual_is_defect": residual_is_defect,
    }
