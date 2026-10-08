"""The collections score, the roll probability and the bands, with four kinds of overlay.

The scorecard gives points per characteristic. Overlays then move the score, the calibration scale, the roll
probability and one band edge, and the unadjusted score, probability and band stay beside the adjusted ones.
"""
from __future__ import annotations

from datetime import date

from decider import DecisionTableConfig, ParamTable, ScorecardConfig, allow_fallback, missing_as, param, param_table, step

from credit_core import calibration
from credit_core.adjustments import Adjustment, AdjustmentEffect
from credit_core.evidence import cell_id

from collections_treatment import overlays

BAND_VERSION = "collections-band-2026.09"
CALIBRATION_VERSION = "collections-cal-2026.09"
CONTACT_BAND_VERSION = "contact-band-2026.09"
RECOVERY_CURVE_VERSION = "recovery-2026.09"
ADJUSTMENT_SET_ID = "AS-COLL-SCORE-2026.09"

ADJUSTMENTS = overlays.register([
    Adjustment(
        adjustment_id="ADJ-COLL-SCORE-001", kind="score_shift", target="collections_score",
        effect=AdjustmentEffect("add", -22.0), scope={"agency_placed_12m": True}, stack_position=1,
        owner="Collections Strategy", approval_reference="MRV-2026-014", tighten_only=True,
        rationale="Accounts placed with an agency in the last 12 months score optimistic on the base curve",
        effective_from=date(2026, 1, 1), effective_to=date(2026, 12, 31), review_date=date(2026, 10, 1)),
    Adjustment(
        adjustment_id="ADJ-COLL-SCORE-002", kind="odds_multiplier", target="roll_probability",
        effect=AdjustmentEffect("multiply", 1.28), scope={"product_family_code": 1}, stack_position=2,
        owner="Model Risk and Validation", approval_reference="MRV-2026-019", tighten_only=True,
        rationale="Deteriorating quarter on the unsecured book",
        effective_from=date(2026, 6, 1), effective_to=date(2027, 3, 1), review_date=date(2026, 12, 1)),
    Adjustment(
        adjustment_id="ADJ-COLL-SCORE-003", kind="scaling_change", target="calibration_scale",
        effect=AdjustmentEffect("multiply", 23.0 / 20.0), scope={"product_family_code": 2}, stack_position=1,
        owner="Model team", approval_reference="MRV-2026-021",
        rationale="Points-to-double-the-odds moved from 20 to 23 for the revolving segment",
        effective_from=date(2026, 3, 1), effective_to=date(2027, 1, 1), review_date=date(2026, 12, 1)),
    Adjustment(
        adjustment_id="ADJ-COLL-SCORE-004", kind="band_boundary_shift", target="collections_band_boundary",
        effect=AdjustmentEffect("add", 0.03), scope={}, stack_position=1,
        owner="Collections Strategy", approval_reference="CCF-2026-028",
        rationale="Move the band 3/4 edge up by 0.03 of roll probability",
        effective_from=date(2026, 5, 1), effective_to=date(2026, 11, 1), review_date=date(2026, 10, 1)),
])

_BUCKET_POINTS = [40, 20, 0, -20, -35, -50, -60, -65]


def _binned(name: str, default: int, bins: list[tuple]) -> dict:
    """A scored characteristic from (points, lower, upper) bins; either bound may be None."""
    return {"type": "scored", "variable_name": name, "strict": False, "default": {"value": default, "name": "unknown"},
            "bins": [{"value": points, **({} if lo is None else {"lower_bound": lo}),
                      **({} if hi is None else {"upper_bound": hi})} for points, lo, hi in bins]}


def _counted(name: str, default: int, points: list[int]) -> dict:
    return {"type": "scored", "variable_name": name, "strict": False, "default": {"value": default, "name": "unknown"},
            "bins": [{"value": p, "items": [i]} for i, p in enumerate(points)]}


def build_scorecard() -> ScorecardConfig:
    """Base 600 plus points for the bucket, cure history, payment behaviour, contact and promise record.

    Writes `collections_score_unadjusted`. A missing characteristic scores its default bin, never zero by accident.
    """
    return ScorecardConfig.load({
        "type": "scorecard", "name": "collections_roll_cure", "output_name": "collections_score_unadjusted",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "base_score"},
            {"type": "scored", "variable_name": "arrears_bucket_code", "strict": False,
             "default": {"value": -10, "name": "unknown"},
             "bins": [{"value": p, "items": [i + 1]} for i, p in enumerate(_BUCKET_POINTS)]},
            _counted("times_cured_12m", 0, [15, -5, -20]),
            _binned("payment_ratio_last", -15, [(-25, None, 0.3), (0, 0.3, 0.9), (30, 0.9, None)]),
            _binned("days_since_last_payment", -30, [(20, None, 30.0), (-5, 30.0, 90.0), (-30, 90.0, None)]),
            _binned("arrears_to_balance_ratio", 0, [(15, None, 0.1), (-10, 0.1, 0.4), (-30, 0.4, None)]),
            _binned("right_party_contact_rate_90d", -15, [(-10, None, 0.2), (10, 0.2, 0.6), (25, 0.6, None)]),
            _binned("promise_kept_rate", 0, [(-20, None, 0.3), (5, 0.3, 0.75), (25, 0.75, None)]),
            _counted("broken_promises_90d", 0, [10, -15, -35]),
            _binned("account_age_months", 0, [(-5, None, 6.0), (5, 6.0, 60.0), (10, 60.0, None)]),
            _counted("other_accounts_in_arrears", -10, [10, -10, -25]),
        ],
    })


def build_calibration_table() -> DecisionTableConfig:
    """Anchor and points-to-double-the-odds `scale` for the account's `product_family_code`."""
    rows = [{"family": family, "anchor": 600.0, "scale": scale,
             "cell_id": cell_id("collections_calibration", CALIBRATION_VERSION, family)}
            for family, scale in {1: 55.0, 2: 60.0, 3: 65.0, 4: 50.0}.items()]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "collections_calibration",
        "columns": {"family": "Int64", "anchor": "Float64", "scale": "Float64", "cell_id": "String"}, "rows": rows,
        "expression": {"type": "eq", "variable": "product_family_code", "value_column": "family"},
        "outputs": ["anchor", "scale", "cell_id"], "default": [600.0, 60.0, None],
    }).relabel(writes={"cell_id": "calibration_cell_id"})


def build_contact_band_table() -> DecisionTableConfig:
    """`contact_band_code`: 1 responsive, 2 intermittent, 3 silent, 4 unreachable, from the 90-day contact rate."""
    inf = float("inf")
    rows = [{"lo": lo, "hi": hi, "band": band, "cell_id": cell_id("contact_band", CONTACT_BAND_VERSION, band)}
            for lo, hi, band in [(-inf, 0.05, 4), (0.05, 0.2, 3), (0.2, 0.5, 2), (0.5, inf, 1)]]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "contact_band",
        "columns": {"lo": "Float64", "hi": "Float64", "band": "Int64", "cell_id": "String"}, "rows": rows,
        "expression": {"type": "between", "variable": "right_party_contact_rate_90d",
                       "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band", "cell_id"], "default": [4, None],
    }).relabel(writes={"band": "contact_band_code", "cell_id": "contact_band_cell_id"})


def build_recovery_curve_table() -> DecisionTableConfig:
    """`recovery_rate` by arrears bucket and collections band."""
    rows = [{"bucket": bucket, "band": band, "recovery_rate": round(max(0.02, 1.0 - (bucket - 1) * 0.10 - (band - 1) * 0.08), 4),
             "cell_id": cell_id("recovery_curve", RECOVERY_CURVE_VERSION, bucket, band)}
            for bucket in range(1, 9) for band in range(1, 7)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "recovery_curve",
        "columns": {"bucket": "Int64", "band": "Int64", "recovery_rate": "Float64", "cell_id": "String"}, "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "arrears_bucket_code", "value_column": "bucket"},
            {"type": "eq", "variable": "collections_band_code", "value_column": "band"}]},
        "outputs": ["recovery_rate", "cell_id"], "default": [0.05, None],
    }).relabel(writes={"cell_id": "recovery_curve_cell_id"})


def _stack(target: str, base: float, provenance: dict, decision_date: date, enabled: bool) -> tuple[float, list[str]]:
    result = ADJUSTMENTS.apply_stack(target, base, provenance, decision_date, ADJUSTMENT_SET_ID, stack_enabled=enabled)
    return result.adjusted_value, list(result.adjustments_applied)


@allow_fallback
def score_shift(
    collections_score_unadjusted: float, decision_date: date, agency_placed_12m: bool = missing_as(False),
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, list[str]]:
    return _stack("collections_score", collections_score_unadjusted, {"agency_placed_12m": bool(agency_placed_12m)},
                  decision_date, adjustment_stack_enabled)


@allow_fallback
def scaling_change(
    scale: float, decision_date: date, product_family_code: float = missing_as(-1.0),
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, list[str]]:
    return _stack("calibration_scale", scale, {"product_family_code": product_family_code}, decision_date,
                  adjustment_stack_enabled)


@allow_fallback
def odds_multiplier(
    roll_probability_before_overlay: float, decision_date: date, product_family_code: float = missing_as(-1.0),
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, list[str]]:
    value, applied = _stack("roll_probability", roll_probability_before_overlay,
                            {"product_family_code": product_family_code}, decision_date, adjustment_stack_enabled)
    return min(max(value, 0.0), 1.0), applied


@allow_fallback
def band_boundary_shift(decision_date: date, adjustment_stack_enabled: bool = param(True)) -> tuple[float, list[str]]:
    """How far the band 3/4 edge has moved, as a probability."""
    return _stack("collections_band_boundary", 0.0, {}, decision_date, adjustment_stack_enabled)


BAND_EDGES = param_table({"upper": float}, default=[{"upper": e} for e in (0.10, 0.25, 0.45, 0.65, 0.85)],
                         shared_key="collections_band_edges")
SHIFTED_EDGE = 2  # the edge between bands 3 and 4


def collections_bands(
    roll_probability: float, roll_probability_unadjusted: float, band_edge_shift: float,
    band_edges: ParamTable = BAND_EDGES,
) -> tuple[int, int]:
    """`collections_band_code` and `collections_band_code_unadjusted`: 1 most likely to cure, 6 most likely to roll.

    Only the 3/4 edge moves with `band_edge_shift`; the unadjusted band uses the edges as authored.
    """
    band = unadjusted = 1
    for i in range(len(band_edges.upper)):
        band += roll_probability >= band_edges.upper[i] + (band_edge_shift if i == SHIFTED_EDGE else 0.0)
        unadjusted += roll_probability_unadjusted >= band_edges.upper[i]
    return band, unadjusted


@allow_fallback
def collections_band_cell_id(collections_band_code: int) -> str:
    return cell_id("collections_band", BAND_VERSION, collections_band_code)


def recovery_and_cost(
    outstanding_balance: float, recovery_rate: float, cost_per_treatment: float = param(15.0, ge=0.0),
    expected_treatments: float = param(3.0, ge=0.0),
) -> tuple[float, float]:
    """`recovery_estimate` and `cost_to_collect`."""
    return round(outstanding_balance * recovery_rate, 2), round(cost_per_treatment * expected_treatments, 2)


score_shift_step = step(score_shift, outputs=("collections_score", "score_shift_adjustments_applied"))
scaling_change_step = step(scaling_change, outputs=("adjusted_scale", "scaling_change_adjustments_applied"))
odds_multiplier_step = step(odds_multiplier, outputs=("roll_probability", "odds_multiplier_adjustments_applied"))
band_boundary_shift_step = step(band_boundary_shift, outputs=("band_edge_shift", "band_boundary_shift_adjustments_applied"))
collections_bands_step = step(collections_bands, outputs=("collections_band_code", "collections_band_code_unadjusted"))
recovery_and_cost_step = step(recovery_and_cost, outputs=("recovery_estimate", "cost_to_collect"))

roll_probability_unadjusted = calibration.probability_of_default_step.relabel(
    reads={"score": "collections_score_unadjusted"}, writes={"probability_of_default": "roll_probability_unadjusted"},
).named("roll_probability_unadjusted")
roll_probability_before_overlay = calibration.probability_of_default_step.relabel(
    reads={"score": "collections_score", "scale": "adjusted_scale"},
    writes={"probability_of_default": "roll_probability_before_overlay"},
).named("roll_probability_before_overlay")
