"""`core.calibration` -- score to probability, and back (spec 00 §6.11; addendum A2).

A logistic calibration curve, keyed by `segment_code` (addendum A2: 00 keys
calibration on "segment" but never named it -- this is the fix). Both
directions are closed-form and exact inverses of each other by
construction, which is what "must agree to tolerance" (§6.11 "Hard part")
needs: `score_for_pd(pd_for_score(s, seg), seg) == s` to floating-point
precision, not merely within a band.

The anchor/scale pair is a per-segment table lookup (`DecisionTableConfig`),
so "the same capability twice with different settings" (§10 item 4) is one
call with a different `segment_code`, not a fork.
"""
from __future__ import annotations

import math

from decider import step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

CALIBRATION_VERSION = "cal-2026.09"

# segment_code -> (anchor score, scale). Illustrative (front matter).
_SEGMENTS = {1: (600.0, 60.0), 2: (620.0, 55.0), 3: (560.0, 70.0), 4: (600.0, 60.0), 5: (600.0, 60.0)}


def build_calibration_table() -> DecisionTableConfig:
    rows = [
        {"segment": seg, "anchor": anchor, "scale": scale,
         "cell_id": _cell_id("calibration", CALIBRATION_VERSION, seg), "calibration_version": CALIBRATION_VERSION}
        for seg, (anchor, scale) in _SEGMENTS.items()
    ]
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
    """A higher score is lower risk: PD falls as score rises past `anchor`."""
    return 1.0 / (1.0 + math.exp((score - anchor) / scale))


def score_for_probability(probability_of_default: float, anchor: float, scale: float) -> float:
    """The inverse: the score that would calibrate to a target PD (appetite's "what score would we need")."""
    p = min(max(probability_of_default, 1e-9), 1.0 - 1e-9)
    return anchor + scale * math.log((1.0 - p) / p)


probability_of_default_step = step(probability_of_default)
score_for_probability_step = step(score_for_probability)
