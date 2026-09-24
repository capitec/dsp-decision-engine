"""P10 -- Affordability (spec 10 §5.11): this project's own calibration over 00's unit.

The chain (10 §5.11):

    discretionary_income = net_monthly_income - living_expenses - existing_obligations
    capacity             = discretionary_income - residual_floor(dependants)
    ratio_ceiling        = instalment_to_income_ratio(grade, product) x net_monthly_income
    pre_buffer           = min(capacity, ratio_ceiling)
    max_affordable_instalment = pre_buffer x (1 - buffer(grade, product, channel))

`core.affordability.discretionary_income` computes the first line exactly
(reused unchanged). The `capacity`/`ratio_ceiling`/`pre_buffer` composition
is this project's own -- 00 deliberately does not own it (00-ADDENDUM
"ownership" note: 00 owns the units' arithmetic, 02 -- and here, 10 -- own
the household framing). `core.affordability.max_affordable_instalment`
(`pre_buffer x (1 - buffer)`) is then reused for the *last* line by
`.relabel()`-ing its `discretionary_income` read to `pre_buffer`: the
"same capability, different settings" pattern project 00's own NOTES.md
recommends (§10 item 4), not a second implementation of one multiply.

The verdict (`affordability_verdict_code`) is this project's **own**
policy calibration -- R185 minimum viable instalment, an 8% marginal band,
evidence-tier gating -- not `core.affordability.affordability_verdict_code`,
whose 5%-band default is a different product's policy (00's demo). This is
the exact ownership boundary project 00's NOTES.md "Spec problems" flags as
easy to get wrong: 00 publishes a verdict *shape* (PASS/MARGINAL/FAIL/
INDETERMINATE) as a unit output; the policy that decides *where* those
boundaries sit is this project's, reproduced here as its own function
rather than assumed to be 00's default.
"""
from __future__ import annotations

from decider import missing_as, param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.affordability import FAIL, INDETERMINATE, MARGINAL, PASS
from credit_core.affordability import max_affordable_instalment_step as _core_max_affordable_instalment_step
from credit_core.evidence import cell_id as _cell_id
from retail_credit.overlays import ADJUSTMENT_SET_ID, OVERLAY_REGISTER

BUFFER_VERSION = "p10-buffer-2026.09"
RATIO_VERSION = "p10-ratio-2026.09"

MINIMUM_VIABLE_INSTALMENT = 185.0
MARGINAL_BAND = 0.08
MINIMUM_EVIDENCE_TIER = 6  # tiers 6-7 are below minimum (10 §5.7(b)): verdict caps at MARGINAL/INDETERMINATE

_RESIDUAL_FLOOR = [1450.0, 1950.0, 2400.0, 2850.0, 3250.0, 3600.0, 3900.0]  # 0..6+ dependants


def residual_floor(dependants_count: int) -> float:
    idx = min(max(int(dependants_count), 0), len(_RESIDUAL_FLOOR) - 1)
    return _RESIDUAL_FLOOR[idx]


residual_floor_step = step(residual_floor)


def build_ratio_ceiling_table() -> DecisionTableConfig:
    """Instalment-to-income ratio by grade, product 10 only: 32% at grade 1 to 14% at
    grade 11 (10 §5.11), linear between.
    """
    rows = []
    for grade in range(1, 13):
        ratio = round(0.32 - (grade - 1) * (0.32 - 0.14) / 10.0, 4) if grade <= 11 else 0.10
        rows.append({"grade": grade, "ratio": ratio,
                      "cell_id": _cell_id("p10.ratio_ceiling", RATIO_VERSION, grade), "ratio_version": RATIO_VERSION})
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "p10_ratio_ceiling",
        "columns": {"grade": "Int64", "ratio": "Float64", "cell_id": "String", "ratio_version": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
        "outputs": ["ratio", "cell_id", "ratio_version"],
        "default": [None, None, None],
    })


def ratio_ceiling(ratio: float, net_monthly_income: float) -> float:
    return round(ratio * net_monthly_income, 2)


ratio_ceiling_step = step(ratio_ceiling, output="ratio_ceiling")


def capacity(discretionary_income: float, residual_floor: float) -> float:
    return round(discretionary_income - residual_floor, 2)


capacity_step = step(capacity, output="capacity")


def pre_buffer(capacity: float, ratio_ceiling: float) -> tuple[float, str]:
    """(pre_buffer, binding_constraint) -- min(capacity, ratio_ceiling), attributed (10 §5.14
    item 6 vocabulary reused here: BIND-AFF-CAP vs. BIND-AFF-RATIO)."""
    if capacity <= ratio_ceiling:
        return capacity, "capacity"
    return ratio_ceiling, "ratio_ceiling"


pre_buffer_step = step(pre_buffer, outputs=("pre_buffer", "affordability_binding_constraint"))


def build_buffer_table() -> DecisionTableConfig:
    """Buffer by grade x channel class, product 10 only (10 §5.11: 12 grades x 6 products x
    4 channel classes = 288 in full; this project's own slice is 12 x 4 = 48, one product).
    """
    channel_classes = {1: 0.06, 2: 0.10, 3: 0.14, 4: 0.18}  # branch, app, web, broker/partner
    rows = []
    for grade in range(1, 13):
        for channel_class, base in channel_classes.items():
            buffer = round(min(0.24, base + (grade - 1) * 0.01), 4)
            rows.append({
                "grade": grade, "channel_class": channel_class, "buffer": buffer,
                "cell_id": _cell_id("p10.buffer", BUFFER_VERSION, grade, channel_class),
                "buffer_version": BUFFER_VERSION,
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "p10_buffer",
        "columns": {"grade": "Int64", "channel_class": "Int64", "buffer": "Float64", "cell_id": "String",
                    "buffer_version": "String"},
        "rows": rows,
        "expression": {
            "type": "and",
            "expressions": [
                {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
                {"type": "eq", "variable": "channel_code", "value_column": "channel_class"},
            ],
        },
        "outputs": ["buffer", "cell_id", "buffer_version"],
        "default": [None, None, None],
    })


def buffer_adjustment_step():
    """The buffer overlay (10 §5.11's worked example, ADJ-0114: +4pp on channel 3). Same
    register, same resolution O-05/O-21 requires -- read from `retail_credit.overlays`,
    never re-resolved here.
    """
    return OVERLAY_REGISTER.apply_stack_step(
        "affordability_buffer", ADJUSTMENT_SET_ID, base_field="buffer",
        adjusted_output="affordability_buffer_applied", unadjusted_output="affordability_buffer_unadjusted",
    )


# `pre_buffer x (1 - buffer)` is exactly `core.affordability.max_affordable_instalment`'s
# formula (`discretionary_income x (1 - affordability_buffer_applied)`) -- reused by
# `.relabel()`-ing its `discretionary_income` read to this project's `pre_buffer`, the
# "same capability, different settings" pattern (00 NOTES.md §10 item 4), not a second
# one-line implementation of the same multiply.
max_affordable_instalment_step = _core_max_affordable_instalment_step.relabel(reads={"discretionary_income": "pre_buffer"})


def affordability_verdict_code(
    max_affordable_instalment: float, income_evidence_tier: int = missing_as(1),
    minimum_viable_instalment: float = param(MINIMUM_VIABLE_INSTALMENT, ge=0.0),
    marginal_band: float = param(MARGINAL_BAND, ge=0.0, le=0.5),
) -> int:
    """This project's own verdict policy (10 §5.11's table): 1 pass, 2 marginal, 3 fail,
    4 indeterminate. Evidence tiers 6-7 are below the minimum tier for product 10 amounts
    above R80 000 (10 §5.7(b)); at this project's slice depth that is simplified to "any
    tier 6-7 request is at best marginal", which is the conservative direction of the
    spec's actual rule and declared here rather than silently narrowed.
    """
    if income_evidence_tier >= MINIMUM_EVIDENCE_TIER:
        return INDETERMINATE if income_evidence_tier == 7 else MARGINAL
    if max_affordable_instalment >= minimum_viable_instalment:
        return PASS
    if max_affordable_instalment >= minimum_viable_instalment * (1.0 - marginal_band):
        return MARGINAL
    return FAIL


affordability_verdict_code_step = step(affordability_verdict_code, output="affordability_verdict_code")
