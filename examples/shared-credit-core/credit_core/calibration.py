"""Score to probability of default and back, with a logistic curve per segment."""
from __future__ import annotations

import math

from decider import DecisionTableConfig, step

from credit_core.evidence import cell_id

CALIBRATION_VERSION = "cal-2026.09"

# segment_code -> (anchor score, scale): PD is 50% at the anchor and falls as the score rises.
_SEGMENTS = {1: (600.0, 60.0), 2: (620.0, 55.0), 3: (560.0, 70.0), 4: (600.0, 60.0), 5: (600.0, 60.0)}


def build_calibration_table() -> DecisionTableConfig:
    """Writes `anchor`, `scale`, `calibration_cell_id` and `calibration_version` for the record's `segment_code`."""
    rows = [{"segment": segment, "anchor": anchor, "scale": scale,
             "cell_id": cell_id("calibration", CALIBRATION_VERSION, segment), "calibration_version": CALIBRATION_VERSION}
            for segment, (anchor, scale) in _SEGMENTS.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "calibration_segments",
        "columns": {"segment": "Int64", "anchor": "Float64", "scale": "Float64", "cell_id": "String",
                    "calibration_version": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "segment_code", "value_column": "segment"},
        "outputs": ["anchor", "scale", "cell_id", "calibration_version"],
        "default": [600.0, 60.0, None, CALIBRATION_VERSION],
    }).relabel(writes={"cell_id": "calibration_cell_id"})


def probability_of_default(score: float, anchor: float, scale: float) -> float:
    return 1.0 / (1.0 + math.exp((score - anchor) / scale))


def score_for_probability(probability_of_default: float, anchor: float, scale: float) -> float:
    """The inverse: the score that calibrates to a target PD (kept inside 1e-9 .. 1 - 1e-9)."""
    p = min(max(probability_of_default, 1e-9), 1.0 - 1e-9)
    return anchor + scale * math.log((1.0 - p) / p)


probability_of_default_step = step(probability_of_default)
score_for_probability_step = step(score_for_probability)
