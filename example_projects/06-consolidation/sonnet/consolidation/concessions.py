"""The concession catalogue (spec 06 §5.9, §6.1: "8 concessions x 4 levels x 5
attributes"). SCOPE.md keeps only this table from the restructure variant, because
project 11 depends on it (11 §5.17.4: "components from four other projects", 00,
02, 05 and 06) even though this slice does not build the restructure search
itself.

A `DecisionTableConfig` keyed on `concession_code`, so it carries the same
cell-level attribution and reviewable-diff properties as every other table in
this library (00 §8) -- table lookup, not a plain dict, even at eight rows,
because the *evidence* requirement is what a restructure consumer needs, not
the row count.
"""
from __future__ import annotations

from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

CONCESSION_CATALOGUE_VERSION = "concessions-2026.09"

CNC_PAYMENT_HOLIDAY = 1
CNC_TERM_EXTENSION = 2
CNC_RATE_CONCESSION = 3
CNC_ARREARS_CAPITALISATION = 4
CNC_TEMP_REDUCTION_STEP_UP = 5
CNC_FEE_INTEREST_WAIVER = 6
CNC_PARTIAL_FORGIVENESS = 7
CNC_CONSOLIDATION_INTO_11 = 8

_ROWS = [
    # code, name, min_bound, max_bound, unit, min_authority_level
    (CNC_PAYMENT_HOLIDAY, "payment_holiday", 1, 3, "months", 1),
    (CNC_TERM_EXTENSION, "term_extension", 0, 36, "months", 1),
    (CNC_RATE_CONCESSION, "rate_concession", 0, 400, "bps", 2),
    (CNC_ARREARS_CAPITALISATION, "arrears_capitalisation", 0, 6, "months_of_arrears", 1),
    (CNC_TEMP_REDUCTION_STEP_UP, "temp_reduction_step_up", 50, 80, "pct_of_contractual", 2),
    (CNC_FEE_INTEREST_WAIVER, "fee_interest_waiver", 0, 15_000, "rand", 1),
    (CNC_PARTIAL_FORGIVENESS, "partial_capital_forgiveness", 0, None, "rand", 4),
    (CNC_CONSOLIDATION_INTO_11, "consolidation_into_product_11", 0, None, "rand", 2),
]

# authority_level_code -> (npv_cost_floor, npv_cost_ceiling_exclusive), §5.9's table.
AUTHORITY_BANDS = [
    (1, 0.0, 5_000.0),
    (2, 5_000.0, 25_000.0),
    (3, 25_000.0, 150_000.0),
    (4, 150_000.0, float("inf")),
]


def build_concession_catalogue() -> DecisionTableConfig:
    rows = [
        {"concession_code": code, "name": name, "min_bound": min_b, "max_bound": max_b, "unit": unit,
         "min_authority_level": level, "cell_id": _cell_id("concessions", CONCESSION_CATALOGUE_VERSION, code)}
        for code, name, min_b, max_b, unit, level in _ROWS
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "concession_catalogue",
        "columns": {"concession_code": "Int64", "name": "String", "min_bound": "Float64", "max_bound": "Float64",
                    "unit": "String", "min_authority_level": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "concession_code", "value_column": "concession_code"},
        "outputs": ["name", "min_bound", "max_bound", "unit", "min_authority_level", "cell_id"],
        "default": [None, None, None, None, None, None],
    })


def authority_level_for_npv_cost(npv_cost: float, repeat_concession: bool = False) -> int:
    """§5.9: "A second concession to the same client within 12 months raises the
    required authority by one level." """
    for level, lo, hi in AUTHORITY_BANDS:
        if lo <= npv_cost < hi:
            return min(level + 1, 4) if repeat_concession else level
    return 4
