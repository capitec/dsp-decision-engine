"""Risk assessment: the collections score and its bands (spec 08 §5.3).

Built entirely from decider's and 00's own mechanisms, nothing bespoke:
`ScorecardConfig` (00's own `credit_core.scorecard` pattern) for the score
and its per-characteristic contributions; `credit_core.calibration`'s
sigmoid, reused unmodified, for score -> roll probability (the same
"anchor/scale, closed-form, invertible" mechanism 00 built for PD, applied
here to roll probability -- 00 §6.11's own point that calibration is a
generic score-to-probability capability, not a credit-granting-specific
one); `credit_core.adjustments.AdjustmentRegister` for the four overlay
kinds (§5.3 table), called through `apply_stack` directly (its own
docstring: "a consumer needing a wider scope... builds its own step from
`apply_stack` directly") rather than through `apply_stack_step`'s
fixed four-key provenance, since this project's overlay scopes are bucket,
product family and placement history, not product/segment/channel/
scorecard.

The four kinds are each a different `target` on one register, not four
mechanisms:

| Kind (§5.3)      | `target`                    | effect      |
|---|---|---|
| Score shift        | `collections_score`           | add       |
| Scaling change      | `calibration_scale`           | multiply  |
| Odds multiplier     | `roll_probability`            | multiply  |
| Band boundary shift  | `collections_band_boundary`   | add (subtracted from the probability before the band table reads it -- moving the probability the table sees is exactly equivalent to moving the edge the table is built with, and it keeps the band table itself untouched between releases) |
"""
from __future__ import annotations

from datetime import date

from decider import missing_as, param, step
from decider.steps.scorecard import ScorecardConfig
from decider.steps.tables import DecisionTableConfig

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.calibration import probability_of_default as _pd_curve
from credit_core.evidence import cell_id as _cell_id

from collections_treatment import vocab

SCORECARD_ID = 2
SCORECARD_VERSION = "collections-sc-2026.09"
CALIBRATION_VERSION = "collections-cal-2026.09"
BANDING_VERSION = "collections-band-2026.09"
RECOVERY_CURVE_VERSION = "recovery-2026.09"
CONTACT_BAND_VERSION = "contact-band-2026.09"


def assert_no_statutory_target(register: AdjustmentRegister) -> None:
    """§5.4: "an overlay definition that names one of [the statutory parameters] is
    rejected at authoring time, not at run time." Checked once, at register
    construction, against every adjustment this project ever defines."""
    for a in register._all:  # noqa: SLF001 -- the register has no public enumerator; see NOTES.md
        if a.target in vocab.STATUTORY_TARGETS:
            raise ValueError(
                f"{a.adjustment_id}: target {a.target!r} is a Compliance-owned statutory "
                f"parameter and may never be named by an overlay (spec 08 §5.4)"
            )


# --- The scorecard (§5.3 item 1: 28 characteristics in the spec; working depth here) -------
# ponytail: 10 characteristics, not 28. The mechanism (bin -> points, per-characteristic
# contribution, null-is-a-bin) is what this slice proves; scorecard.py in project 00 sets
# the precedent for this exact trim. Add characteristics by appending `variables` below.

_REASON_BY_VARIABLE = {
    "current_bucket": 4201, "times_cured_12m": 4202, "payment_ratio_last": 4203,
    "days_since_last_payment": 4204, "arrears_to_balance_ratio": 4205,
    "right_party_contact_rate_90d": 4206, "promise_kept_rate": 4207,
    "broken_promises_90d": 4208, "account_age_months": 4209, "other_accounts_in_arrears": 4210,
}
VARIABLES = tuple(_REASON_BY_VARIABLE)


def build_scorecard() -> ScorecardConfig:
    return ScorecardConfig.load({
        "type": "scorecard", "name": "collections_roll_cure",
        "output_name": "collections_score_raw",
        "variables": [
            {"type": "constant", "score": 600, "output_name": "base_score"},
            {"type": "scored", "variable_name": "current_bucket", "strict": False,
             "default": {"value": -10, "name": "unknown"}, "bins": [
                 {"value": 40, "items": [1]}, {"value": 20, "items": [2]}, {"value": 0, "items": [3]},
                 {"value": -20, "items": [4]}, {"value": -35, "items": [5]}, {"value": -50, "items": [6]},
                 {"value": -60, "items": [7]}, {"value": -65, "items": [8]},
             ]},
            {"type": "scored", "variable_name": "times_cured_12m", "strict": False,
             "default": {"value": 0, "name": "unknown"}, "bins": [
                 {"value": 15, "items": [0]}, {"value": -5, "items": [1]}, {"value": -20, "items": [2]},
             ]},
            {"type": "scored", "variable_name": "payment_ratio_last", "strict": False,
             "default": {"value": -15, "name": "no_payment"}, "bins": [
                 {"value": -25, "upper_bound": 0.3, "name": "low"},
                 {"value": 0, "lower_bound": 0.3, "upper_bound": 0.9, "name": "partial"},
                 {"value": 30, "lower_bound": 0.9, "name": "full_or_more"},
             ]},
            {"type": "scored", "variable_name": "days_since_last_payment", "strict": False,
             "default": {"value": -30, "name": "never"}, "bins": [
                 {"value": 20, "upper_bound": 30.0, "name": "recent"},
                 {"value": -5, "lower_bound": 30.0, "upper_bound": 90.0, "name": "aging"},
                 {"value": -30, "lower_bound": 90.0, "name": "stale"},
             ]},
            {"type": "scored", "variable_name": "arrears_to_balance_ratio", "strict": False,
             "default": {"value": 0, "name": "unknown"}, "bins": [
                 {"value": 15, "upper_bound": 0.1, "name": "low"},
                 {"value": -10, "lower_bound": 0.1, "upper_bound": 0.4, "name": "mid"},
                 {"value": -30, "lower_bound": 0.4, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "right_party_contact_rate_90d", "strict": False,
             "default": {"value": -15, "name": "no_contact"}, "bins": [
                 {"value": -10, "upper_bound": 0.2, "name": "low"},
                 {"value": 10, "lower_bound": 0.2, "upper_bound": 0.6, "name": "mid"},
                 {"value": 25, "lower_bound": 0.6, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "promise_kept_rate", "strict": False,
             "default": {"value": 0, "name": "no_promises"}, "bins": [
                 {"value": -20, "upper_bound": 0.3, "name": "low"},
                 {"value": 5, "lower_bound": 0.3, "upper_bound": 0.75, "name": "mid"},
                 {"value": 25, "lower_bound": 0.75, "name": "high"},
             ]},
            {"type": "scored", "variable_name": "broken_promises_90d", "strict": False,
             "default": {"value": 0, "name": "unknown"}, "bins": [
                 {"value": 10, "items": [0]}, {"value": -15, "items": [1]}, {"value": -35, "items": [2]},
             ]},
            {"type": "scored", "variable_name": "account_age_months", "strict": False,
             "default": {"value": 0, "name": "unknown"}, "bins": [
                 {"value": -5, "upper_bound": 6.0, "name": "new"},
                 {"value": 5, "lower_bound": 6.0, "upper_bound": 60.0, "name": "established"},
                 {"value": 10, "lower_bound": 60.0, "name": "mature"},
             ]},
            {"type": "scored", "variable_name": "other_accounts_in_arrears", "strict": False,
             "default": {"value": -10, "name": "unknown"}, "bins": [
                 {"value": 10, "items": [0]}, {"value": -10, "items": [1]}, {"value": -25, "items": [2]},
             ]},
        ],
    })


def adverse_action_codes(scores: dict[str, float], n: int = 4) -> list[int]:
    negative = sorted(((v, pts) for v, pts in scores.items() if pts < 0), key=lambda item: item[1])
    return [_REASON_BY_VARIABLE[v] for v, _ in negative[:n]]


# --- Score-to-probability (reuses `credit_core.calibration`'s sigmoid unmodified) ----------

def build_calibration_table() -> DecisionTableConfig:
    """One segment per product family (working depth -- 00's own `calibration.py` keys on a
    handful of segments too). `scale` is what the "scaling change" overlay multiplies."""
    rows = [
        {"segment": pf, "anchor": 600.0, "scale": scale,
         "cell_id": _cell_id("collections_calibration", CALIBRATION_VERSION, pf)}
        for pf, scale in {1: 55.0, 2: 60.0, 3: 65.0, 4: 50.0}.items()
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "collections_calibration_segments",
        "columns": {"segment": "Int64", "anchor": "Float64", "scale": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "eq", "variable": "product_family_code", "value_column": "segment"},
        "outputs": ["anchor", "scale", "cell_id"],
        "default": [600.0, 60.0, None],
    }).relabel(writes={"cell_id": "calibration_cell_id"})


def roll_probability_unadjusted(collections_score: float, anchor: float, scale: float) -> float:
    return _pd_curve(collections_score, anchor, scale)


roll_probability_unadjusted_step = step(roll_probability_unadjusted, output="roll_probability_before_overlay")


def scaling_change(
    scale: float, decision_date: date,
    product_family_code: int = missing_as(-1),
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, list[str]]:
    """"Scaling change" (§5.3): "points-to-double-the-odds from 20 to 23 for the revolving
    segment" -- a multiplicative change to the calibration curve's own `scale`, applied
    *before* the curve runs (this step's output overwrites the table's `scale` column;
    decider's "later write wins" ordering is what makes that safe -- see pipeline.py)."""
    provenance = {"product_family_code": product_family_code}
    result = SCORE_ADJUSTMENTS.apply_stack(
        "calibration_scale", scale, provenance, decision_date,
        SCORE_ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    return result.adjusted_value, list(result.adjustments_applied)


scaling_change_step = step(scaling_change, outputs=("scale", "scaling_change_adjustments_applied"))


# --- Bands (§5.3 item 2: 6 collections bands, keyed on roll probability) -------------------

def _band_table_rows() -> list[dict]:
    """1 = most likely to cure (lowest roll probability) .. 6 = most likely to roll."""
    edges = [
        (float("-inf"), 0.10), (0.10, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 0.85), (0.85, float("inf")),
    ]
    return [
        {"lo": lo, "hi": hi, "band": i + 1, "cell_id": _cell_id("collections_band", BANDING_VERSION, i)}
        for i, (lo, hi) in enumerate(edges)
    ]


def _build_band_table(*, read_from: str, code_output: str, cell_output: str) -> DecisionTableConfig:
    """One table definition, parameterised on its own boundary names -- built twice (§5.3
    "unadjusted band ... carried beside the adjusted one") without a second definition and
    without chaining `.relabel()` (chaining doesn't compose: a second `.relabel(writes=...)`
    call keys on the step's *original* write names, not whatever an earlier relabel already
    renamed them to, so `build_band_table().relabel(writes={"collections_band_code": ...})`
    silently no-ops and both copies collide on the same final name -- `decider.exceptions.
    WiringError: ... both write 'collections_band_cell_id'`. See NOTES.md."""
    return DecisionTableConfig.load({
        "type": "decision_table", "name": f"collections_band_{code_output}",
        "columns": {"lo": "Float64", "hi": "Float64", "band": "Int64", "cell_id": "String"},
        "rows": _band_table_rows(),
        "expression": {"type": "between", "variable": read_from,
                        "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band", "cell_id"],
        "default": [6, None],
    }).relabel(writes={"band": code_output, "cell_id": cell_output})


def build_band_table() -> DecisionTableConfig:
    return _build_band_table(
        read_from="roll_probability_for_banding",
        code_output="collections_band_code", cell_output="collections_band_cell_id",
    )


def build_unadjusted_band_table() -> DecisionTableConfig:
    return _build_band_table(
        read_from="roll_probability_before_overlay",
        code_output="collections_band_code_unadjusted", cell_output="collections_band_cell_id_unadjusted",
    )


# --- Contact responsiveness band (§5.3 item 5) ---------------------------------------------

def build_contact_band_table() -> DecisionTableConfig:
    edges = [(float("-inf"), 0.05), (0.05, 0.2), (0.2, 0.5)]
    rows = [
        {"lo": lo, "hi": hi, "band": i + 1, "cell_id": _cell_id("contact_band", CONTACT_BAND_VERSION, i)}
        for i, (lo, hi) in enumerate(edges)
    ] + [{"lo": 0.5, "hi": float("inf"), "band": 1,
          "cell_id": _cell_id("contact_band", CONTACT_BAND_VERSION, 3)}]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "contact_band",
        "columns": {"lo": "Float64", "hi": "Float64", "band": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "between", "variable": "right_party_contact_rate_90d",
                        "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band", "cell_id"],
        "default": [4, None],
    }).relabel(writes={"band": "contact_band_code", "cell_id": "contact_band_cell_id"})


# --- Recovery estimate and cost to collect (§5.3 items 3-4) ---------------------------------
# Working depth: bucket x band, not the spec's bucket x band x family x balance x 24 months.

def build_recovery_curve_table() -> DecisionTableConfig:
    rows = []
    for bucket in range(1, 9):
        for band in range(1, 7):
            cure_odds = max(0.02, 1.0 - (bucket - 1) * 0.10 - (band - 1) * 0.08)
            rows.append({
                "bucket": bucket, "band": band, "recovery_rate": round(cure_odds, 4),
                "cell_id": _cell_id("recovery_curve", RECOVERY_CURVE_VERSION, bucket, band),
            })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "recovery_curve",
        "columns": {"bucket": "Int64", "band": "Int64", "recovery_rate": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "arrears_bucket_code", "value_column": "bucket"},
            {"type": "eq", "variable": "collections_band_code", "value_column": "band"},
        ]},
        "outputs": ["recovery_rate", "cell_id"],
        "default": [0.05, None],
    }).relabel(writes={"cell_id": "recovery_curve_cell_id"})


def recovery_and_cost(
    outstanding_balance: float, recovery_rate: float,
    cost_per_treatment: float = param(15.0, ge=0.0),
    expected_treatments: float = param(3.0, ge=0.0),
) -> tuple[float, float]:
    recovery_estimate = round(outstanding_balance * recovery_rate, 2)
    cost_to_collect = round(cost_per_treatment * expected_treatments, 2)
    return recovery_estimate, cost_to_collect


recovery_and_cost_step = step(recovery_and_cost, outputs=("recovery_estimate", "cost_to_collect"))


# --- The overlay register (§5.3: four kinds, one register) ---------------------------------

SCORE_ADJUSTMENT_SET_ID = "AS-08-SCORE-2026.09"
SCORE_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-08-SCORE-001", kind="score_shift", target="collections_score",
        effect=AdjustmentEffect("add", -22.0), scope={"agency_placed_12m": True}, stack_position=1,
        owner="Collections Strategy", approval_reference="MRV-2026-014",
        rationale="Accounts placed with an agency in the last 12 months score systematically "
                  "optimistic on the base curve", effective_from=date(2026, 1, 1),
        effective_to=date(2026, 12, 31), review_date=date(2026, 10, 1),
    ),
    Adjustment(
        adjustment_id="ADJ-08-SCORE-002", kind="odds_multiplier", target="roll_probability",
        effect=AdjustmentEffect("multiply", 1.28), scope={"product_family_code": 1}, stack_position=2,
        owner="Model Risk and Validation", approval_reference="MRV-2026-019",
        rationale="Deteriorating quarter on the unsecured book", effective_from=date(2026, 6, 1),
        effective_to=date(2027, 3, 1), review_date=date(2026, 12, 1),
    ),
    Adjustment(
        adjustment_id="ADJ-08-SCORE-003", kind="scaling_change", target="calibration_scale",
        effect=AdjustmentEffect("multiply", 23.0 / 20.0), scope={"product_family_code": 2}, stack_position=1,
        owner="Model team", approval_reference="MRV-2026-021",
        rationale="Points-to-double-the-odds moved from 20 to 23 for the revolving segment",
        effective_from=date(2026, 3, 1), effective_to=date(2027, 1, 1), review_date=date(2026, 12, 1),
    ),
    Adjustment(
        adjustment_id="ADJ-08-SCORE-004", kind="band_boundary_shift", target="collections_band_boundary",
        effect=AdjustmentEffect("add", 0.03), scope={}, stack_position=1,
        owner="Collections Strategy", approval_reference="CCF-2026-028",
        rationale="Move the band 3/4 edge by 0.03 of roll probability", effective_from=date(2026, 5, 1),
        effective_to=date(2026, 11, 1), review_date=date(2026, 10, 1),
    ),
])
assert_no_statutory_target(SCORE_ADJUSTMENTS)


def _score_shift(
    collections_score_raw: float, decision_date: date,
    agency_placed_12m: bool = missing_as(False),
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, float, list[str]]:
    provenance = {"agency_placed_12m": bool(agency_placed_12m)}
    result = SCORE_ADJUSTMENTS.apply_stack(
        "collections_score", collections_score_raw, provenance, decision_date,
        SCORE_ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    return result.adjusted_value, result.unadjusted_value, list(result.adjustments_applied)


score_shift_step = step(_score_shift, outputs=(
    "collections_score", "collections_score_unadjusted", "score_shift_adjustments_applied",
))


def _odds_multiplier(
    roll_probability_before_overlay: float, decision_date: date,
    product_family_code: int = missing_as(-1),
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, list[str]]:
    provenance = {"product_family_code": product_family_code}
    result = SCORE_ADJUSTMENTS.apply_stack(
        "roll_probability", roll_probability_before_overlay, provenance, decision_date,
        SCORE_ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    value = min(max(result.adjusted_value, 0.0), 1.0)
    return value, list(result.adjustments_applied)


odds_multiplier_step = step(_odds_multiplier, outputs=("roll_probability", "odds_multiplier_adjustments_applied"))


def band_boundary_shift(
    roll_probability: float, decision_date: date,
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, list[str]]:
    """"Move the band 3/4 edge by 0.03 of roll probability" (§5.3): the shift amount
    itself is computed through the same `apply_stack` mechanism as every other overlay
    here (base value 0.0, target `collections_band_boundary`), then subtracted from the
    probability the band table reads. Moving the probability a fixed table reads is
    exactly equivalent to moving the edge, and it lets the six-row band table stay a
    single, unmodified, diffable artefact across every overlay."""
    result = SCORE_ADJUSTMENTS.apply_stack(
        "collections_band_boundary", 0.0, {}, decision_date,
        SCORE_ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    shifted = min(max(roll_probability - result.adjusted_value, 0.0), 1.0)
    return shifted, list(result.adjustments_applied)


band_boundary_shift_step = step(
    band_boundary_shift, outputs=("roll_probability_for_banding", "band_boundary_shift_adjustments_applied"),
)
