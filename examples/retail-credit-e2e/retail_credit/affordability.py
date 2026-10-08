"""Affordability: what the client can commit each month, from the shared discretionary income to a verdict.

    discretionary_income      = net income - living expenses - existing obligations      (credit_core)
    capacity                  = discretionary_income - residual floor for the dependants
    pre_buffer                = the lower of capacity and the grade's share of net income
    max_affordable_instalment = pre_buffer x (1 - buffer for grade and channel)           (credit_core)
    instalment_cap            = max_affordable_instalment, reduced for self-employed and joint households

`capacity_unit` is everything that depends on the obligations, so the consolidation search can run it again on a
hypothetical figure; `limits_unit` is everything that depends only on the grade and channel.
"""
from __future__ import annotations

from decider import DecisionTableConfig, dag, missing_as, param, step

from credit_core import affordability
from credit_core.affordability import FAIL, INDETERMINATE, MARGINAL, PASS
from credit_core.evidence import cell_id
from retail_credit.overlays import overlay_step

RATIO_VERSION = "ratio-2026.09"
BUFFER_VERSION = "buffer-2026.09"

# affordability_binding_constraint_code
CAPACITY_BOUND = 1
RATIO_CEILING_BOUND = 2

_CHANNEL_BUFFER = {1: 0.06, 2: 0.10, 3: 0.14, 4: 0.18}  # branch, app, web, broker or partner


def build_residual_floor_table() -> DecisionTableConfig:
    """Writes `residual_floor`: R1 450 for no dependants rising to R3 900 for six or more."""
    floors = [1450.0, 1950.0, 2400.0, 2850.0, 3250.0, 3600.0, 3900.0]
    edges = [None, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, None]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "residual_floor_grid",
        "columns": {"lo": "Float64", "hi": "Float64", "floor": "Float64"},
        "rows": [{"lo": edges[i], "hi": edges[i + 1], "floor": floor} for i, floor in enumerate(floors)],
        "expression": {"type": "between", "variable": "dependants_count", "lower_bound_column": "lo",
                       "upper_bound_column": "hi"},
        "outputs": ["floor"], "default": [3900.0],
    }).relabel(writes={"floor": "residual_floor"})


def build_ratio_ceiling_table() -> DecisionTableConfig:
    """Writes `ratio` (the share of net income an instalment may take: 32% at grade 1 to 14% at grade 11, 10% at 12)
    and `ratio_ceiling_cell_id`, for `risk_grade`."""
    rows = [{"grade": grade, "ratio": round(0.32 - (grade - 1) * 0.18 / 10, 4) if grade <= 11 else 0.10,
             "cell_id": cell_id("ratio_ceiling", RATIO_VERSION, grade)} for grade in range(1, 13)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "ratio_ceiling_grid",
        "columns": {"grade": "Int64", "ratio": "Float64", "cell_id": "String"},
        "rows": rows, "expression": {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
        "outputs": ["ratio", "cell_id"], "default": [None, None],
    }).relabel(writes={"cell_id": "ratio_ceiling_cell_id"})


def build_buffer_table() -> DecisionTableConfig:
    """Writes `buffer` and `buffer_cell_id` for `risk_grade` and `channel_code`: from 6% to 24%, rising with the grade."""
    rows = [{"grade": grade, "channel": channel, "buffer": round(min(0.24, base + (grade - 1) * 0.01), 4),
             "cell_id": cell_id("buffer", BUFFER_VERSION, grade, channel)}
            for grade in range(1, 13) for channel, base in _CHANNEL_BUFFER.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "buffer_grid",
        "columns": {"grade": "Int64", "channel": "Int64", "buffer": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "risk_grade", "value_column": "grade"},
            {"type": "eq", "variable": "channel_code", "value_column": "channel"}]},
        "outputs": ["buffer", "cell_id"], "default": [None, None],
    }).relabel(reads={"channel_code": "channel_class"}, writes={"cell_id": "buffer_cell_id"})


def channel_class(channel_code: int) -> float:
    # workaround: a table reads its `eq` variable as float and an input column has one type, while the shared overlay
    # step reads `channel_code` as int; this converts for the table.
    return float(channel_code)


def ratio_ceiling(ratio: float, net_monthly_income: float) -> float:
    return round(ratio * net_monthly_income, 2)


def capacity(discretionary_income: float, residual_floor: float) -> float:
    return round(discretionary_income - residual_floor, 2)


def pre_buffer(capacity: float, ratio_ceiling: float) -> tuple[float, int]:
    """(pre_buffer, affordability_binding_constraint_code): the lower of the two limits, and which one it was."""
    return (capacity, CAPACITY_BOUND) if capacity <= ratio_ceiling else (ratio_ceiling, RATIO_CEILING_BOUND)


def instalment_cap(
    max_affordable_instalment: float, self_employed_haircut: bool = missing_as(False),
    is_joint_application: bool = missing_as(False), self_employed_pct: float = param(0.90, gt=0.0, le=1.0),
    joint_household_pct: float = param(0.95, gt=0.0, le=1.0),
) -> float:
    """The instalment the solve may not exceed: the affordable one, reduced for a self-employed earner and for a joint
    household. Only ever lower than `max_affordable_instalment`."""
    cap = max_affordable_instalment * (self_employed_pct if self_employed_haircut else 1.0)
    return round(cap * (joint_household_pct if is_joint_application else 1.0), 2)


def affordability_verdict_code(
    max_affordable_instalment: float, income_evidence_tier: int, income_source_code: int,
    minimum_viable_instalment: float = param(185.0, ge=0.0), marginal_band: float = param(0.08, ge=0.0, le=0.5),
    weak_evidence_tier: int = param(6, ge=1, le=7),
) -> int:
    """PASS from the minimum viable instalment up, MARGINAL within `marginal_band` below it, else FAIL.

    Income evidence that is only a bureau estimate or a declaration (tier `weak_evidence_tier` or worse) turns a pass
    into a marginal, never a fail into anything better. No income established at all is indeterminate: the remedy is
    evidence, not a decline.
    """
    if income_source_code == 0:
        return INDETERMINATE
    if max_affordable_instalment >= minimum_viable_instalment:
        verdict = PASS
    elif max_affordable_instalment >= minimum_viable_instalment * (1.0 - marginal_band):
        verdict = MARGINAL
    else:
        return FAIL
    return MARGINAL if income_evidence_tier >= weak_evidence_tier else verdict


def limits_unit():
    """Grade- and channel-based limits: residual floor, ratio ceiling, buffer with its overlay.

    Writes `residual_floor`, `ratio_ceiling`, `affordability_buffer_applied`.
    """
    return dag(
        build_residual_floor_table(), build_ratio_ceiling_table(), step(ratio_ceiling), step(channel_class),
        build_buffer_table(),
        overlay_step("affordability_buffer", "buffer", base_field="buffer", adjusted_output="affordability_buffer_applied",
                     unadjusted_output="affordability_buffer_unadjusted"),
        name="limits")


# the shared step multiplies discretionary income by (1 - buffer); here it multiplies the pre-buffer capacity
max_affordable_instalment_step = affordability.max_affordable_instalment_step.relabel(
    reads={"discretionary_income": "pre_buffer"})


def capacity_unit(name: str = "capacity"):
    """Everything that depends on `existing_obligations`; run again on a hypothetical figure by the consolidation search.

    Writes `discretionary_income`, `capacity`, `pre_buffer`, `max_affordable_instalment`, `instalment_cap` and
    `affordability_verdict_code`.
    """
    return dag(
        affordability.discretionary_income_step, step(capacity), step(pre_buffer, outputs=("pre_buffer", "affordability_binding_constraint_code")),
        max_affordable_instalment_step, step(instalment_cap), step(affordability_verdict_code),
        name=name)
