"""The limit assignment matrix: 12 grades x 8 utilisation bands x 6 tenure bands x 2 products = 1 152 cells.

Each cell carries a multiplier, a maximum increase and a minimum increment. The matrix is a config document
(`configs/<version>/matrix.json`, written by `scripts/generate_configs.py`) so Credit Risk Policy can replace it
without a deployment; `pipeline.build(matrix)` receives it loaded.
"""
from __future__ import annotations

from decider import DecisionTableConfig, allow_fallback, step

from credit_core.evidence import cell_id
from limit_mgmt.vocab import PRODUCT_ACCESS_FACILITY, PRODUCT_EVERYDAY_CARD

MATRIX_VERSION = "limit-matrix-2026.09"

N_GRADES, N_UTIL_BANDS, N_MOB_BANDS = 12, 8, 6

_UTIL_FACTOR = {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.55, 5: 0.70, 6: 0.85, 7: 0.95, 8: 0.40}  # a fully drawn account is a worse bet
_MOB_FACTOR = {1: 0.10, 2: 0.30, 3: 0.55, 4: 0.75, 5: 0.90, 6: 0.60}
_MOB_CEILING = {1: 3_000, 2: 35_000, 3: 15_000}  # R3 000 under a year of tenure; tenures beyond these get R60 000


def _multiplier(grade: int, util_band: int, mob_band: int) -> float:
    if grade >= 9:
        return 1.00
    raw = 1.0 + 0.75 * ((13 - grade) / 12.0) * _UTIL_FACTOR[util_band] * _MOB_FACTOR[mob_band]
    return round(min(1.75, raw), 4)


def _max_increase(grade: int, mob_band: int, product: int) -> float:
    if grade >= 9:
        return 0.0
    grade_scale = max(0.15, (13 - grade) / 12.0)
    product_scale = 1.0 if product == PRODUCT_EVERYDAY_CARD else 0.65
    return round(min(60_000, _MOB_CEILING.get(mob_band, 60_000) * grade_scale * product_scale) / 500) * 500


def generate_matrix_rows() -> list[dict]:
    rows = []
    for product in (PRODUCT_EVERYDAY_CARD, PRODUCT_ACCESS_FACILITY):
        for grade in range(1, N_GRADES + 1):
            for util_band in range(1, N_UTIL_BANDS + 1):
                for mob_band in range(1, N_MOB_BANDS + 1):
                    rows.append({
                        "product": product, "grade": grade, "util_band": util_band, "mob_band": mob_band,
                        "multiplier": _multiplier(grade, util_band, mob_band),
                        "max_increase": _max_increase(grade, mob_band, product),
                        "min_increment": float(min(2_500, 500 + 150 * (grade - 1))),
                        "cell_id": cell_id("limit_matrix", MATRIX_VERSION, product, grade, util_band, mob_band)})
    return rows


def matrix_document(rows: list[dict] | None = None) -> dict:
    return {
        "type": "decision_table", "name": "limit_assignment_matrix",
        "columns": {"product": "Int64", "grade": "Int64", "util_band": "Int64", "mob_band": "Int64",
                    "multiplier": "Float64", "max_increase": "Float64", "min_increment": "Float64",
                    "cell_id": "String"},
        "rows": generate_matrix_rows() if rows is None else rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "behaviour_grade", "value_column": "grade"},
            {"type": "eq", "variable": "utilisation_band", "value_column": "util_band"},
            {"type": "eq", "variable": "mob_band", "value_column": "mob_band"}]},
        "outputs": ["multiplier", "max_increase", "min_increment", "cell_id"],
        "default": [1.00, 0.0, 2_500.0, None],
    }


def build_matrix_table(rows: list[dict] | None = None) -> DecisionTableConfig:
    """The matrix as `pipeline.build` receives it; pass `rows` to simulate a candidate matrix."""
    return DecisionTableConfig.load(matrix_document(rows))


def lookup(matrix: DecisionTableConfig):
    """The matrix step writing the cell as authored: `matrix_multiplier_unadjusted`, `matrix_max_increase_unadjusted`,
    `matrix_min_increment_unadjusted` and `matrix_cell_id`."""
    return matrix.relabel(writes={
        "multiplier": "matrix_multiplier_unadjusted", "max_increase": "matrix_max_increase_unadjusted",
        "min_increment": "matrix_min_increment_unadjusted", "cell_id": "matrix_cell_id"})


@allow_fallback
def matrix_version() -> str:
    return MATRIX_VERSION


def uncapped_target_limit(current_limit: float, matrix_multiplier: float) -> float:
    return round(current_limit * matrix_multiplier, 2)


def uncapped_target_limit_unadjusted(current_limit: float, matrix_multiplier_unadjusted: float) -> float:
    return round(current_limit * matrix_multiplier_unadjusted, 2)


matrix_version_step = step(matrix_version)
uncapped_target_limit_step = step(uncapped_target_limit)
uncapped_target_limit_unadjusted_step = step(uncapped_target_limit_unadjusted)
