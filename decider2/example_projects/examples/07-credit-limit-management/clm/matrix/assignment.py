"""The 1 152-cell limit assignment matrix (s5.4) and the cycle dial over it.

Three things this file insists on that doc 03's provisional `Table` sketch does
not provide, each forced by a stated requirement:

  1. The lookup emits `matrix_cell_id` AS A VALUE IN THE GRAPH. s6.3.4 requires
     every decision to name the one cell it read. A lookup that returns only a
     number cannot do that, and a `tap` cannot either, because a tap is optional
     and this is not.

  2. The artefact carries a version, an approval reference, an effective range
     and a state (live | candidate | superseded). s6.3.5 requires a version that
     has been simulated but not deployed to exist.

  3. Validation is part of the artefact declaration, including a check that
     WARNS rather than forbids -- the monotonicity check in s6.3.2. A validator
     that can only reject cannot express "flag this and let a human decide".
"""

from decider2 import bounds, module, param, shadow, table, warn
from decider2.credit import overlay_point
from decider2.validate import dense, monotone_in

LimitMatrix = table(
    "clm.limit_matrix",
    keys=("behaviour_grade", "utilisation_band", "mob_band", "product_code"),
    values={"multiplier": float, "max_increase_c": int, "min_increment_c": int},
    # 12 x 8 x 6 x 2 = 1 152 cells, 3 456 values.
    domain={"behaviour_grade": range(1, 13), "utilisation_band": range(1, 9),
            "mob_band": range(1, 7), "product_code": [20, 21]},
    unit={"max_increase_c": "ZAR", "min_increment_c": "ZAR"},   # authored in rand,
    scale={"max_increase_c": 100, "min_increment_c": 100},      # scaled once at load
    validate=[
        dense(),                                     # every cell present
        bounds("multiplier", ge=1.00, le=2.00),
        bounds("max_increase_c", ge=0),
        bounds("min_increment_c", gt=0),
        warn(monotone_in("behaviour_grade", "multiplier", direction="non_increasing",
                         holding=("utilisation_band", "mob_band", "product_code")),
             code="MTX-W01",
             message="cell {cell}: grade {a} receives a larger multiplier than grade {b}"),
    ],
    effective_dated=True,
    states=("candidate", "live", "superseded"),
    cell_id="matrix_cell_id",
    source="artefacts/limit_matrix/",
    authored_in="spreadsheet",
    owner="Credit Risk Policy",
    approval_required="Credit Committee",
    evidence=["matrix_cell_id", "artefact_version"],
)


def matrix_cell_id(behaviour_grade: int, utilisation_band: int, mob_band: int,
                   product_code: int, tables) -> int:
    """Exactly one cell is read. Which one is a value, not a side effect."""
    return tables.limit_matrix.cell(behaviour_grade, utilisation_band, mob_band,
                                    product_code)


def matrix_multiplier(matrix_cell_id: int, tables) -> float:
    """The cell's limit multiplier as authored."""
    return tables.limit_matrix.multiplier[matrix_cell_id]


def matrix_max_increase_c(matrix_cell_id: int, tables) -> int:
    """The cell's ceiling on the rand increase."""
    return tables.limit_matrix.max_increase_c[matrix_cell_id]


def matrix_min_increment_c(matrix_cell_id: int, tables) -> int:
    """Below this the cell yields no increase at all."""
    return tables.limit_matrix.min_increment_c[matrix_cell_id]


def matrix_target_limit_c(current_limit_c: int, matrix_multiplier: float) -> int:
    """The uncapped target. Caps and affordability come next (s5.5, s5.6)."""
    pass  # round_half_up(current_limit_c * matrix_multiplier)


# --- the cycle dial --------------------------------------------------------
# "Run the programme at 80% of the matrix this month", applied to the
# multiplier's EXCESS OVER 1.00 -- so 1.50 becomes 1.40, not 1.20. The kind is
# named in the register, not implied by the code, because getting this wrong is
# a 96 000-client difference (s5.8).

ApplyMatrixOverlays = overlay_point(
    "matrix_overlays",
    register="overlay_set",
    adjusts={"matrix_multiplier": "excess_over_one",
             "matrix_max_increase_c": "cap",
             "matrix_min_increment_c": "replace"},
    scope_keys=["product_code", "behaviour_grade", "utilisation_band", "mob_band",
                "channel_code", "path_code"],
    order="declared",              # each overlay carries its own apply_order
    writes={"applied": "adjustments_applied",
            "set": "adjustment_set_id",
            "cycle_dial": "cycle_dial_id"},
    evidence=["adjustments_applied", "adjustment_set_id", "cycle_dial_id"],
)

MatrixLookup = module(matrix_cell_id, matrix_multiplier, matrix_max_increase_c,
                      matrix_min_increment_c, name="matrix_lookup",
                      evidence=["matrix_cell_id"])
Target = module(matrix_target_limit_c, name="matrix_target")

# offer/construct.py's `meets_minimum_unadjusted` reads `matrix_min_increment_
# unadjusted_c` and `matrix_multiplier_unadjusted` -- the values this cell
# would have produced with the cycle dial and every other matrix-scoped
# overlay disabled. Same mechanism as scoring/behavioural.py and
# decrease/triggers.py: one authored chain, `shadow` re-emits it with
# `overlay_set` neutralised, so the adjusted and unadjusted readings cannot
# drift apart. This is the whole reason `shadow` was imported above.
Matrix = (
    MatrixLookup | ApplyMatrixOverlays | Target
) | shadow(
    MatrixLookup | Target,
    neutralise={"overlay_set": "off"},
    suffix="_unadjusted",
    keep=["matrix_multiplier", "matrix_min_increment_c", "matrix_target_limit_c"],
)
