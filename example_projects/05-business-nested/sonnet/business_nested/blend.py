"""Stage 6 -- the people component (spec 05 §5.8), a reduced rule set: PP-01 (inclusion),
PP-02 (coverage), PP-03 (base weights), PP-04 (control weighting), PP-05 (log-odds blend),
a three-item subset of PP-06 (worst-of overrides), and PP-11 (overlay decomposition, as an
aggregate delta rather than the full per-entity-per-overlay table -- see NOTES.md).
PP-08/PP-09 (surety/guarantor cover) are out of this slice: no security/cover model is
built here (SCOPE.md keeps structuring at "a single product 50 lookup").

Ordinary flat, parallel entity-level lists in, ordinary scalars out -- no ragged
Python loop is needed here (unlike `structure.py`/`events.py`/`rollup.py`), so this
is a plain `step()`, not a `frame_step`: everything it reads is already one row's
worth of application-level lists.
"""
from __future__ import annotations

import math

from decider import missing_as, param, step

from business_nested import vocab
from business_nested.scoring import grade_from_pd

BLEND_INCLUSION_FLOOR_PCT = 5.0
COVERAGE_REQUIREMENT = 0.75
CONTROL_WEIGHT_FLOOR = 0.60
NON_OWNING_CONTROLLER_POINTS = 10.0
NON_OWNING_CONTROLLER_CAP = 30.0
CAP_GRADE_11_12_OWNERSHIP_PCT = 20.0

PP_06_DISQUALIFYING_ENTITY = "PP-06-disqualifying-entity"
PP_06_WEAK_CONTROLLER_CAP = "PP-06-weak-controller-cap"
PP_06_CRITICAL_MATERIAL_FLOOR = "PP-06-critical-material-floor"


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _blend_weights(
    entity_id: list[int], is_owner: list[bool], is_controlling: list[bool], rel_code: list[int],
    eff_own: list[float], grade: list[int],
) -> tuple[list[bool], list[float], float]:
    """PP-01/PP-03/PP-04. Returns (included flags, final weight per entity [0 outside
    the blend], scoreable-ownership coverage ratio)."""
    included = [
        is_owner[i] and eff_own[i] >= BLEND_INCLUSION_FLOOR_PCT
        or is_controlling[i]
        or rel_code[i] in vocab.BLEND_ROLE_INCLUDE
        for i in range(len(entity_id))
    ]
    # PP-07: sureties/guarantors/group companies/signatories never enter the blend,
    # ownership or control notwithstanding -- they are surety cover (PP-08, out of
    # scope) or exposure aggregation (out of scope), not people risk.
    for i, code in enumerate(rel_code):
        if code in vocab.SURETY_OR_GUARANTOR_ROLES or code == vocab.GROUP_COMPANY or code == vocab.SIGNATORY:
            included[i] = False

    owner_total = sum(eff_own[i] for i in range(len(entity_id)) if is_owner[i])
    scoreable_owner_total = sum(
        eff_own[i] for i in range(len(entity_id)) if is_owner[i] and included[i] and grade[i] is not None
    )
    coverage = 1.0 if owner_total <= 0 else scoreable_owner_total / owner_total

    weights = [0.0] * len(entity_id)
    non_owning_points_used = 0.0
    for i in range(len(entity_id)):
        if not included[i]:
            continue
        if is_owner[i]:
            weights[i] = eff_own[i] / 100.0
        else:
            points = min(NON_OWNING_CONTROLLER_POINTS, max(0.0, NON_OWNING_CONTROLLER_CAP - non_owning_points_used))
            non_owning_points_used += points
            weights[i] = points / 100.0

    # PP-04: one controlling/majority entity takes >= 0.60 of the blend; the rest renormalise.
    control_idx = None
    for i in range(len(entity_id)):
        if included[i] and (is_controlling[i] or eff_own[i] >= 50.0):
            if control_idx is None or eff_own[i] > eff_own[control_idx]:
                control_idx = i
    if control_idx is not None:
        control_weight = max(eff_own[control_idx] / 100.0, CONTROL_WEIGHT_FLOOR)
        rest_total = sum(w for i, w in enumerate(weights) if i != control_idx)
        remaining = 1.0 - control_weight
        for i in range(len(entity_id)):
            if i == control_idx:
                weights[i] = control_weight
            elif rest_total > 0:
                weights[i] = weights[i] / rest_total * remaining
    else:
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

    return included, weights, coverage


def people_component(
    entity_id: list[int], entity_is_owner: list[bool], entity_is_controlling: list[bool],
    entity_relationship_type_code: list[int], entity_effective_ownership_pct: list[float],
    entity_criticality_class: list[int], entity_verdict_code: list[int],
    entity_verdict_code_unadjusted: list[int], entity_pd: list[float], entity_pd_unadjusted: list[float],
    entity_grade: list[int],
) -> dict:
    n = len(entity_id)
    if n == 0:
        return {
            "people_pd": 1.0, "people_pd_unadjusted": 1.0, "people_grade": 12, "people_grade_unadjusted": 12,
            "people_weight_vector": [], "insufficient_people_coverage": True, "no_scoreable_natural_person": True,
            "business_decline_from_entity": False, "business_decline_entity_id": None,
            "business_decline_binding_rule": "", "people_pd_overlay_contribution": 0.0,
            "people_cap_rule": "",
        }

    included, weights, coverage = _blend_weights(
        entity_id, entity_is_owner, entity_is_controlling, entity_relationship_type_code,
        entity_effective_ownership_pct, entity_grade,
    )

    def blend(pds: list[float]) -> float:
        total_w = sum(w for w in weights if w > 0)
        if total_w <= 0:
            return max(pds) if pds else 1.0
        logit_sum = sum(w * _logit(pds[i]) for i, w in enumerate(weights) if w > 0)
        return _sigmoid(logit_sum / total_w)

    people_pd = blend(entity_pd)
    people_pd_unadjusted = blend(entity_pd_unadjusted)
    people_grade = grade_from_pd(people_pd, vocab.SEGMENT_SME_PEOPLE)
    people_grade_unadjusted = grade_from_pd(people_pd_unadjusted, vocab.SEGMENT_SME_PEOPLE)

    cap_rule = ""
    # PP-06 (subset): weak-grade cap for a large included stake.
    for i in range(n):
        if included[i] and entity_grade[i] is not None and entity_grade[i] >= 11 \
                and entity_effective_ownership_pct[i] >= CAP_GRADE_11_12_OWNERSHIP_PCT:
            if people_grade < 10:
                people_grade = 10
                cap_rule = PP_06_WEAK_CONTROLLER_CAP

    # PP-06 (subset): a critical entity at material-or-worse floors the people grade at 5
    # (standing in for AE-R-08's `recent_adverse`, out of this slice's 6 roll-up rules --
    # see NOTES.md).
    for i in range(n):
        if entity_criticality_class[i] == vocab.CRITICAL and entity_verdict_code[i] >= vocab.MATERIAL:
            if people_grade < 5:
                people_grade = 5
                cap_rule = PP_06_CRITICAL_MATERIAL_FLOOR

    # PP-06 / PP-07: any entity disqualifying (included or not -- exclusion from the
    # average is not exclusion from the rules) declines the business outright.
    business_decline = False
    decline_entity_id = None
    decline_rule = ""
    for i in range(n):
        if entity_verdict_code[i] == vocab.DISQUALIFYING:
            business_decline = True
            decline_entity_id = entity_id[i]
            decline_rule = PP_06_DISQUALIFYING_ENTITY
            break

    no_scoreable_natural_person = not any(
        included[i] and entity_grade[i] is not None
        for i in range(n)
        # `entity_is_owner` alone doesn't tell us natural-person-ness; callers pass
        # `entity_is_natural_person` separately at the pipeline level for PP-10's
        # check (see `grade.py`) -- this flag only covers "nobody scoreable at all".
    )

    return {
        "people_pd": people_pd, "people_pd_unadjusted": people_pd_unadjusted,
        "people_grade": people_grade, "people_grade_unadjusted": people_grade_unadjusted,
        "people_weight_vector": weights, "insufficient_people_coverage": coverage < COVERAGE_REQUIREMENT,
        "no_scoreable_natural_person": no_scoreable_natural_person,
        "business_decline_from_entity": business_decline, "business_decline_entity_id": decline_entity_id,
        "business_decline_binding_rule": decline_rule,
        "people_pd_overlay_contribution": people_pd - people_pd_unadjusted,
        "people_cap_rule": cap_rule,
    }


_PEOPLE_OUTPUTS = (
    "people_pd", "people_pd_unadjusted", "people_grade", "people_grade_unadjusted", "people_weight_vector",
    "insufficient_people_coverage", "no_scoreable_natural_person", "business_decline_from_entity",
    "business_decline_entity_id", "business_decline_binding_rule", "people_pd_overlay_contribution",
    "people_cap_rule",
)


def _people_component_tuple(
    entity_id: list[int], entity_is_owner: list[bool], entity_is_controlling: list[bool],
    entity_relationship_type_code: list[int], entity_effective_ownership_pct: list[float],
    entity_criticality_class: list[int], entity_verdict_code: list[int],
    entity_verdict_code_unadjusted: list[int], entity_pd: list[float], entity_pd_unadjusted: list[float],
    entity_grade: list[int],
) -> tuple[float, float, int, int, list[float], bool, bool, bool, int | None, str, float, str]:
    """`decider.step()` requires several outputs as a plain tuple, in declared order
    (its own docstring: "the return annotation must be a `tuple[...]`") -- this thin
    wrapper is the adapter; `people_component` above stays dict-returning and directly
    unit-testable (readable field names, no positional tuple to miscount)."""
    result = people_component(
        entity_id, entity_is_owner, entity_is_controlling, entity_relationship_type_code,
        entity_effective_ownership_pct, entity_criticality_class, entity_verdict_code,
        entity_verdict_code_unadjusted, entity_pd, entity_pd_unadjusted, entity_grade,
    )
    return tuple(result[name] for name in _PEOPLE_OUTPUTS)


people_component_step = step(_people_component_tuple, name="people_component", outputs=_PEOPLE_OUTPUTS)
