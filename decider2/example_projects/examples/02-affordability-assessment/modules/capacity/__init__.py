"""Stage 6 -- discretionary income and capacity.

Two things happen here and they are deliberately not one thing.

The LADDER is the prescribed arithmetic in the prescribed order, and its order
matters for presentation rather than for arithmetic -- which means presentation
is the point. A regulator reads it against the regulation line by line. So each
rung is its own step with its own `@rung` narration, and the order in the `|`
expression is the order on the page.

The BUFFER is the Bank's own conservatism, and it is separated from the ladder
because spec 9.2 says the adjudicator's sharper question is whether the Bank
declined someone the statutory calculation would have approved. Answering it
requires `max_affordable_instalment_unadjusted_cents` beside
`max_affordable_instalment_cents`, and both are emitted always.
"""

from decider2 import (
    DECREASES,
    INCREASES,
    LOWEST,
    contest,
    dated_table,
    module,
    overlay_target,
    policy,
    rung,
    step,
    tightens_when,
)

# --------------------------------------------------------------------------
# 6.1 The ladder. Every intermediate is retained because the adjudicator asks
#     for the ladder and not the answer.
# --------------------------------------------------------------------------

@rung(
    section="ladder",
    order=100,
    says=(
        "Discretionary income of {discretionary_income_cents:money}:\n"
        "    gross monthly income          {gross_monthly_income_cents:money}\n"
        "  - statutory deductions          {statutory_deductions_cents:money}\n"
        "  = net monthly income            {net_monthly_income_cents:money}\n"
        "  - living expenses               {living_expenses_cents:money}\n"
        "  - court-ordered deductions      {court_ordered_deductions_cents:money}\n"
        "  - existing obligations          {existing_obligations_cents:money}\n"
        "  = discretionary income          {discretionary_income_cents:money}"
    ),
)
@step(
    output="discretionary_income_cents",
    description=(
        "The prescribed chain. May be negative, and a negative value is "
        "meaningful -- it is the finding that the applicant is already "
        "over-committed, and its magnitude tells project 06 how much relief a "
        "consolidation must produce. It is never floored at zero."
    ),
)
def discretionary_income_cents(
    net_monthly_income_cents: int,
    living_expenses_cents: int,
    court_ordered_deductions_cents: int,
    existing_obligations_cents: int,
) -> int:
    pass  # net - expenses - court orders - obligations. No max(0, ...) anywhere.


# --------------------------------------------------------------------------
# 6.2 The buffer. Two constraints; the binding one wins. Third use of
#     `contest`, and the one that makes monotonicity delicate.
# --------------------------------------------------------------------------

BUFFER_GRID = dated_table(
    "buffer_grid",
    key=("risk_grade", "product_code", "grid_name"),
    columns={"retained_pct": "float64"},
    owner=policy,
    versions="tables/policy/buffer_grid/",
    declares_edges=("risk_grade", "product_code"),
)

RESIDUAL_FLOOR = dated_table(
    "residual_floor",
    key=("dependants_bucket", "grid_name"),
    columns={"floor_cents": "int64"},
    owner=policy,
    versions="tables/policy/residual_floor/",
    declares_edges=("dependants_bucket",),
)
# `grid_name` is a key column rather than a separate table, because spec change
# scenario 8 is collections wanting a distinct grid for arrangement mode. A
# second table would be a second lookup site and a second thing to forget on
# the next gazette. A key column is a profile-set param
# (policy/modes.py: capacity.buffer_grid_name) and reaches the existing lookup.
#
# Spec change scenario 3 -- the buffer varies by channel where today it varies
# by grade and product -- is the same move: add `channel_code` to the key and
# re-issue the table. It is a table change plus one line, not a rewrite, and
# `decider2.impact(active, candidate, sample)` prices it before it ships.


@step(description="Constraint 1: retain a percentage of discretionary income, from the grade x product grid.")
def proportional_capacity_cents(
    discretionary_income_cents: int,
    risk_grade: int | None,
    product_code: int,
    grid_name: str,
    grid = BUFFER_GRID.asof,
) -> int:
    pass  # discretionary * (1 - retained_pct); risk_grade is null in some modes and
          # a null grade resolves to the worst grade, never to the best


@step(
    description=(
        "Constraint 2: a minimum rand amount must remain after the proposed "
        "instalment, which protects low-income applicants for whom a "
        "percentage is too small to matter."
    ),
)
def residual_floor_capacity_cents(
    discretionary_income_cents: int,
    dependants_bucket: int,
    grid_name: str,
    floors = RESIDUAL_FLOOR.asof,
) -> int:
    pass  # discretionary - floor_cents


BufferContest = contest(
    "max_affordable_instalment_unadjusted_cents",
    candidates={
        "proportional_buffer": proportional_capacity_cents,
        "residual_floor": residual_floor_capacity_cents,
    },
    select=LOWEST,
    tie_break=("residual_floor", "proportional_buffer"),
    emits_basis="affordability_buffer_basis_code",
    retain_losers=True,
    # Both candidates are non-decreasing in discretionary income and neither
    # depends on the proposed instalment, so their minimum is too. That is the
    # part of pipeline monotonicity a reader can check by eye; the part they
    # cannot is the interaction across band edges, which is why the property is
    # declared on the pipeline and discharged by an edge-seeded search rather
    # than argued for here.
    monotone=True,
)


@rung(
    section="capacity",
    order=110,
    says=(
        "The maximum affordable instalment before any policy overlay is "
        "{max_affordable_instalment_unadjusted_cents:money}. Two constraints "
        "were computed and the {affordability_buffer_basis_code:basis} bound: "
        "a proportional buffer retaining {retained_pct:pct} of discretionary "
        "income (grade {risk_grade}, product {product_code:product}, grid "
        "'{grid_name}', version {buffer_grid@version}) gave "
        "{proportional_capacity_cents:money}; an absolute residual floor of "
        "{floor_cents:money} at {dependants_count} "
        "dependant{dependants_count:plural} gave "
        "{residual_floor_capacity_cents:money}."
    ),
)
@step(output="affordability_buffer_applied", description="The amount of discretionary income retained by the binding constraint.")
def affordability_buffer_applied(
    discretionary_income_cents: int,
    max_affordable_instalment_unadjusted_cents: int,
) -> int:
    pass  # discretionary - unadjusted capacity


# --------------------------------------------------------------------------
# 6.3 The overlay target. Everything about conservative-only asymmetry is in
#     this declaration and in policy/overlays.py; nothing about it is at
#     runtime.
# --------------------------------------------------------------------------
max_affordable_instalment_cents = overlay_target(
    "max_affordable_instalment_cents",
    base="max_affordable_instalment_unadjusted_cents",
    direction=tightens_when(DECREASES),
    floor=None,                       # zero is a legitimate answer here
    scope_keys=("product_code", "channel_code", "risk_grade", "segment_code"),
    rung_section="overlays",
)

# The same declaration on the inputs to capacity, so an overlay may tighten the
# buffer or the floor rather than the answer -- which is what spec 5.6.2
# requires ("an overlay may raise the buffer, raise the residual floor, raise
# the expense stringency or lower the maximum instalment").
retained_pct = overlay_target(
    "retained_pct",
    base="retained_pct_unadjusted",
    direction=tightens_when(INCREASES),
    ceiling=policy(0.60, ge=0.0, le=1.0),
    scope_keys=("product_code", "channel_code", "risk_grade", "segment_code"),
    rung_section="overlays",
)

floor_cents = overlay_target(
    "floor_cents",
    base="floor_cents_unadjusted",
    direction=tightens_when(INCREASES),
    scope_keys=("product_code", "channel_code", "dependants_bucket"),
    rung_section="overlays",
)


Capacity = module(
    discretionary_income_cents,
    retained_pct,
    floor_cents,
    proportional_capacity_cents,
    residual_floor_capacity_cents,
    BufferContest,
    affordability_buffer_applied,
    max_affordable_instalment_cents,
    name="capacity",
    contract="contracts/capacity.json",
    taps=[
        "discretionary_income_cents",
        "max_affordable_instalment_unadjusted_cents",
        "affordability_buffer_basis_code",
        "max_affordable_instalment_cents@overlay:*",
    ],
)
