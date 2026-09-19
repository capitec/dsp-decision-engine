"""Project vocabulary. One declaration, per doc 03 5.2 layer 2.

Two jobs, and only the first is one doc 03 anticipated.

1. NAME differences between `credit-core`'s published vocabulary and the names
   used inside this project.

2. UNIT differences. This is the collision. `credit-core` publishes every money
   value as `float64` rand (doc 00 4). Doc 03 1.2 forbids exactly that: "money
   is a scaled int64 of cents, never float and never Decimal", because int64
   overflow and njit-vs-CPython `round()` are one-cent wrong-answer bugs, and
   this project must reconcile to the cent seven years later.

   Both cannot hold. The interior is cents; the published interface is the
   library's. The projection is declared here, once, with its rounding named --
   `round_half_up`, never bare `round()` -- rather than being an unwritten
   convention that 79 passthrough steps eventually encode.

   `Vocabulary` as doc 03 defines it maps names to names. It has no unit axis
   and no dtype axis, so the second block below is an invention. See
   FRAMEWORK-DEMANDS #4.
"""

from decider2 import Projection, Vocabulary, round_half_up

CREDIT_CORE = Vocabulary(
    # --- names -------------------------------------------------------------
    {
        # `credit-core` name            # local name
        "bureau_as_of_date": "bureau_as_of_date",
        "risk_grade": "risk_grade",
        "adjustment_set_id": "adjustment_set_id",
        "adjustments_applied": "adjustments_applied",
    },
    prefixes={"applicant_": "app_"},
    # --- units -------------------------------------------------------------
    # Declared per published name. The framework generates the projection step,
    # gives it an audit identity, and puts it in the version chain, so the
    # rand-denominated value an adjudicator reads has a producer like any other
    # value. Losing a cent here is a reconciliation break, so the direction is
    # declared and is not `nearest`.
    projections={
        "gross_monthly_income": Projection(
            "gross_monthly_income_cents", to="float64_rand",
            divide=100, rounding=round_half_up, places=2,
        ),
        "net_monthly_income": Projection(
            "net_monthly_income_cents", to="float64_rand",
            divide=100, rounding=round_half_up, places=2,
        ),
        "living_expenses": Projection(
            "living_expenses_cents", to="float64_rand",
            divide=100, rounding=round_half_up, places=2,
        ),
        "existing_obligations": Projection(
            "existing_obligations_cents", to="float64_rand",
            divide=100, rounding=round_half_up, places=2,
        ),
        "discretionary_income": Projection(
            "discretionary_income_cents", to="float64_rand",
            divide=100, rounding=round_half_up, places=2,
        ),
        "max_affordable_instalment": Projection(
            "max_affordable_instalment_cents", to="float64_rand",
            divide=100, rounding=round_half_up, places=2,
        ),
    },
)

# The projection is one-way at the published boundary and the framework refuses
# the inverse. A caller handing this project a float rand `proposed_instalment`
# gets:
#
#   input 'proposed_instalment' arrives as float64 rand; this pipeline consumes
#   'proposed_instalment_cents' (int64). A float->cents conversion at an input
#   boundary is lossy in a direction nobody declared. Convert at the caller and
#   state the rounding, or supply cents.
#
# That error is deliberate. The alternative -- a silent `int(x * 100)` -- is
# how a cent goes missing, and in seven years nobody can say where.
