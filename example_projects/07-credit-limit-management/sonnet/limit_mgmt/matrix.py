"""§5.4 -- the limit assignment matrix. **Kept at full declared size**
(SCOPE.md rule 1: "keep the dominant difficulty from README §4 at real
size wherever the size *is* the difficulty" -- README §4 names this
project's Q2 stress explicitly as "the limit assignment matrix is
12 x 8 x 6 x 2... 1 152 cells, 3 456 values"). Generated synthetically,
not hand-authored, exactly as 00 generated the 63 360-cell Flex Loan rate
card for the same reason: generating it is cheap and shrinking it would
throw away the one thing this project is in the set to prove.

Loaded as a `configs/<version>/matrix.json` document (`DecisionTableConfig`,
via `generate_configs.py`), so it is the one artefact a Credit Risk Policy
analyst can replace without a code deployment (§6.3 item 1: "authored in a
spreadsheet... loaded without a deployment") and so it is diffable cell by
cell against a candidate (§6.3 item 3) with plain JSON tooling, not code.

The cycle dial (a matrix multiplier over the excess above 1.00) and the
cycle cap are `core.adjustments` overlays over this table's *output*, not
edits to the table -- exactly the "overlay, not edit" property the
library's mechanism exists to hold (00 `adjustments.py` property 1).
"""
from __future__ import annotations

from decider import step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id
from limit_mgmt.vocab import PRODUCT_ACCESS_FACILITY, PRODUCT_EVERYDAY_CARD

MATRIX_VERSION = "limit-matrix-2026.09"

N_GRADES = 12
N_UTIL_BANDS = 8
N_MOB_BANDS = 6


def _multiplier(grade: int, util_band: int, mob_band: int, product: int) -> float:
    """Synthetic but policy-shaped (§5.4: "not monotonic in the obvious direction"):
    rewards higher utilisation and longer tenure, penalises worse grades, floors at
    1.00 (never a decrease from this stage) and caps at 1.75."""
    grade_factor = max(0.0, (13 - grade) / 12.0)          # 1.0 at grade 1, ~0.08 at grade 12
    util_factor = {1: 0.05, 2: 0.15, 3: 0.35, 4: 0.55, 5: 0.70, 6: 0.85, 7: 0.95, 8: 0.40}[util_band]
    mob_factor = {1: 0.10, 2: 0.30, 3: 0.55, 4: 0.75, 5: 0.90, 6: 0.60}[mob_band]
    if grade >= 9:  # §5.4: "a grade-9 account gets 1.00 everywhere"
        return 1.00
    raw = 1.0 + 0.75 * grade_factor * util_factor * mob_factor
    return round(min(1.75, max(1.00, raw)), 4)


def _max_increase(grade: int, util_band: int, mob_band: int, product: int) -> float:
    """§5.4: "Grade-4 accounts in the 6-11 month band are capped at R3 000... nine
    months of good behaviour is nine months of evidence." Ceiling rises with tenure and
    improves (falls) with worse grade; product loading for the facility's lower ceiling."""
    if grade >= 9:
        return 0.0
    base_ceiling = {1: 20_000, 2: 35_000, 3: 15_000}.get(mob_band, 60_000) if mob_band != 1 else 3_000
    grade_scale = max(0.15, (13 - grade) / 12.0)
    product_scale = 1.0 if product == PRODUCT_EVERYDAY_CARD else 0.65
    return round(min(60_000, base_ceiling * grade_scale * product_scale) / 500) * 500


def _min_increment(grade: int, util_band: int, mob_band: int, product: int) -> float:
    """§5.4: R500 - R2 500. Higher for worse grades (not worth a token increase)."""
    return float(min(2_500, 500 + 150 * max(0, grade - 1)))


def generate_matrix_rows() -> list[dict]:
    rows = []
    for product in (PRODUCT_EVERYDAY_CARD, PRODUCT_ACCESS_FACILITY):
        for grade in range(1, N_GRADES + 1):
            for util_band in range(1, N_UTIL_BANDS + 1):
                for mob in range(1, N_MOB_BANDS + 1):
                    rows.append({
                        "product": product, "grade": grade, "util_band": util_band, "mob_band": mob,
                        "multiplier": _multiplier(grade, util_band, mob, product),
                        "max_increase": _max_increase(grade, util_band, mob, product),
                        "min_increment": _min_increment(grade, util_band, mob, product),
                        "cell_id": _cell_id("limit_matrix", MATRIX_VERSION, product, grade, util_band, mob),
                    })
    return rows


def build_matrix_table(rows: list[dict] | None = None) -> DecisionTableConfig:
    """Built directly (not from `configs/`) for tests and for `simulation.py`'s
    candidate-matrix runs; `pipeline.build()` instead loads the externalised document
    (see this module's docstring) so it can be refreshed without a redeploy."""
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "limit_assignment_matrix",
        "columns": {"product": "Int64", "grade": "Int64", "util_band": "Int64", "mob_band": "Int64",
                    "multiplier": "Float64", "max_increase": "Float64", "min_increment": "Float64",
                    "cell_id": "String"},
        "rows": rows if rows is not None else generate_matrix_rows(),
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "product_code", "value_column": "product"},
            {"type": "eq", "variable": "behaviour_grade", "value_column": "grade"},
            {"type": "eq", "variable": "utilisation_band", "value_column": "util_band"},
            {"type": "eq", "variable": "mob_band", "value_column": "mob_band"},
        ]},
        "outputs": ["multiplier", "max_increase", "min_increment", "cell_id"],
        "default": [1.00, 0.0, 2_500.0, None],
    }).relabel(writes={
        "multiplier": "matrix_multiplier_unadjusted", "max_increase": "matrix_max_increase_unadjusted",
        "min_increment": "matrix_min_increment_unadjusted", "cell_id": "matrix_cell_id",
    })


def matrix_version_step_output() -> str:
    return MATRIX_VERSION


matrix_version_step = step(matrix_version_step_output, output="matrix_version")


def uncapped_target_limit(current_limit: float, matrix_multiplier: float) -> float:
    """§5.4 "Emits": the uncapped target = `current_limit` x the (overlay-adjusted)
    multiplier."""
    return round(current_limit * matrix_multiplier, 2)


def uncapped_target_limit_unadjusted(current_limit: float, matrix_multiplier_unadjusted: float) -> float:
    """The same arithmetic with the overlay stack disabled -- required to survive beside
    the adjusted figure everywhere (§5.4 "Emits")."""
    return round(current_limit * matrix_multiplier_unadjusted, 2)


uncapped_target_limit_step = step(uncapped_target_limit)
uncapped_target_limit_unadjusted_step = step(uncapped_target_limit_unadjusted)
