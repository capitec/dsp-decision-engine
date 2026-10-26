"""The people component: one probability of default and one grade for everybody standing behind the business.

Entities are weighted (an owner by their stake, a director or trustee by a notional share, a person in control
by at least 60%), and their probabilities of default are blended on the log-odds scale, never by averaging
grades. The worst entity then overrides what the blend says: a disqualifying entity declines the business
whether or not it is in the blend.
"""
from __future__ import annotations

import math

from decikit import allow_fallback, flow, param, step

from credit_core.vocab import SEGMENT_SME_PEOPLE

from business_nested import scoring, vocab

# Roles that join the blend whatever they own.
_INCLUDED_ROLES = frozenset({vocab.DIRECTOR, vocab.MEMBER, vocab.TRUSTEE, vocab.PARTNER})

PP_06_WEAK_LARGE_HOLDER = "PP-06-weak-large-holder"
PP_06_CRITICAL_MATERIAL = "PP-06-critical-material"


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def blend_weights(
    entities: list[dict], inclusion_floor_pct: float, control_ownership_pct: float, control_weight_floor: float,
    controller_points: float, controller_cap_points: float,
) -> list[float]:
    """One weight per entity, summing to 1; an entity outside the blend has weight 0.

    Example: a 55% controlling owner with two small holders gets 0.60 and the others share the other 0.40.
    """
    def included(e):
        if e["relationship_type_code"] in vocab.OUTSIDE_THE_BLEND_ROLES:
            return False
        return ((e["is_owner"] and e["effective_ownership_pct"] >= inclusion_floor_pct) or e["is_controlling"]
                or e["relationship_type_code"] in _INCLUDED_ROLES)

    weights, notional_used = [], 0.0
    for e in entities:
        if not included(e):
            weights.append(0.0)
        elif e["is_owner"]:
            weights.append(e["effective_ownership_pct"] / 100.0)
        else:
            points = min(controller_points, max(0.0, controller_cap_points - notional_used))
            notional_used += points
            weights.append(points / 100.0)
    # One person who holds control, or half the equity, carries at least 60% of the blend.
    controlling = [i for i, e in enumerate(entities) if weights[i] > 0
                   and (e["is_controlling"] or e["effective_ownership_pct"] >= control_ownership_pct)]
    if controlling:
        c = max(controlling, key=lambda i: entities[i]["effective_ownership_pct"])
        c_weight = max(entities[c]["effective_ownership_pct"] / 100.0, control_weight_floor)
        rest = sum(w for i, w in enumerate(weights) if i != c)
        weights = [c_weight if i == c else (w / rest * (1.0 - c_weight) if rest else 0.0) for i, w in enumerate(weights)]
    total = sum(weights)
    return [w / total for w in weights] if total else weights


def log_odds_blend(pds: list[float], weights: list[float]) -> float:
    """The weighted blend of probabilities of default on the log-odds scale; the worst one when all weights are 0."""
    if not any(weights):
        return max(pds, default=1.0)  # nobody in the blend: assume the worst entity
    return 1.0 / (1.0 + math.exp(-sum(w * _logit(p) for w, p in zip(weights, pds))))


@allow_fallback
@step(outputs=("people_pd", "people_pd_unadjusted", "people_weights", "insufficient_people_coverage",
               "no_scoreable_natural_person"))
def people_blend(
    assessed_entities: list[dict],
    inclusion_floor_pct: float = param(5.0, ge=0.0, le=100.0), control_ownership_pct: float = param(50.0, ge=0.0, le=100.0),
    control_weight_floor: float = param(0.60, ge=0.0, le=1.0), controller_points: float = param(10.0, ge=0.0),
    controller_cap_points: float = param(30.0, ge=0.0), coverage_requirement: float = param(0.75, ge=0.0, le=1.0),
) -> tuple[float, float, list[float], bool, bool]:
    """The blended probability of default, with and without the overlays, the weight of every entity, and two
    reasons to refer: too little of the ownership is in the blend, or no natural person is.

    `people_weights` is index-aligned with `assessed_entities`.
    """
    weights = blend_weights(assessed_entities, inclusion_floor_pct, control_ownership_pct, control_weight_floor,
                            controller_points, controller_cap_points)
    owners = sum(e["effective_ownership_pct"] for e in assessed_entities if e["is_owner"])
    covered = sum(e["effective_ownership_pct"] for e, w in zip(assessed_entities, weights) if e["is_owner"] and w > 0)
    return (log_odds_blend([e["entity_pd"] for e in assessed_entities], weights),
            log_odds_blend([e["entity_pd_unadjusted"] for e in assessed_entities], weights), weights,
            owners > 0 and covered / owners < coverage_requirement,
            not any(e["is_natural_person"] and w > 0 for e, w in zip(assessed_entities, weights)))


def people_segment_code() -> float:
    return float(SEGMENT_SME_PEOPLE)


def people_pd_overlay_contribution(people_pd: float, people_pd_unadjusted: float) -> float:
    return people_pd - people_pd_unadjusted


@allow_fallback
@step(outputs=("people_grade", "people_cap_rule"))
def people_grade_caps(
    people_grade_blended: int, assessed_entities: list[dict], people_weights: list[float],
    weak_grade: int = param(11, ge=1, le=12), weak_ownership_pct: float = param(20.0, ge=0.0, le=100.0),
    weak_cap_grade: int = param(10, ge=1, le=12), critical_floor_grade: int = param(5, ge=1, le=12),
) -> tuple[int, str]:
    """The blended grade, made no better than the worst entity allows.

    A blended entity of grade `weak_grade` or worse holding `weak_ownership_pct` or more, or a critical entity
    whose adverse verdict is material or worse, keeps the people grade from being better than `weak_cap_grade`
    and `critical_floor_grade` respectively. A material verdict stands in for a recent adverse event.
    """
    grade, rule = people_grade_blended, ""
    if any(w > 0 and e["entity_grade"] >= weak_grade and e["effective_ownership_pct"] >= weak_ownership_pct
           for e, w in zip(assessed_entities, people_weights)) and grade < weak_cap_grade:
        grade, rule = weak_cap_grade, PP_06_WEAK_LARGE_HOLDER
    if any(e["criticality_class"] == vocab.CRITICAL and e["adverse_verdict_code"] >= vocab.MATERIAL
           for e in assessed_entities) and grade < critical_floor_grade:
        grade, rule = critical_floor_grade, PP_06_CRITICAL_MATERIAL
    return grade, rule


@allow_fallback
@step(outputs=("business_decline_from_entity", "business_decline_entity_id", "business_decline_from_disputed_event"))
def entity_disqualification(assessed_entities: list[dict]) -> tuple[bool, int | None, bool]:
    """Whether any entity, in the blend or not, is disqualifying, and the first such entity by `entity_id`.

    The last value is true when the business would not be declined if every disputed event were left out.
    """
    ids = [e["entity_id"] for e in assessed_entities if e["adverse_verdict_code"] == vocab.DISQUALIFYING]
    undisputed = any(e["adverse_verdict_code_undisputed"] == vocab.DISQUALIFYING for e in assessed_entities)
    return bool(ids), ids[0] if ids else None, bool(ids) and not undisputed


def people_unit():
    return flow(
        people_blend, step(people_segment_code), step(people_pd_overlay_contribution),
        scoring.grade_lookup("people_pd", "people_grade_blended", "people_segment_code"),
        scoring.grade_lookup("people_pd_unadjusted", "people_grade_unadjusted", "people_segment_code"),
        people_grade_caps, entity_disqualification,
        name="people")
