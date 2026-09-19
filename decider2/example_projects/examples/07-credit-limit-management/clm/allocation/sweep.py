"""The population-level budget allocation (s5.8). This file is the centre of
the project and the reason it is in the set.

THE PROBLEM. Every stage before this one is per-account and embarrassingly
parallel. This one needs every account's result before it can produce any
account's answer: 708 000 eligible accounts proposing R5.11 bn against an ALCO
budget of R2.4 bn. No property of an account determines whether it is funded;
what determines it is where the account sits relative to 707 999 others and
where the money runs out.

WHAT IT IS NOT. It is not a frame operation. Polars can compute a cumulative
sum of `additional_limit_c` over a sorted frame and cut where it crosses the
budget, and that gets about 80% of the way -- but it cannot express the tail
rule (skip an account that does not fit, keep going, stop after 2 000 skips),
it cannot express three envelopes binding independently, and it cannot say
which envelope bound. Bolting those on produces a chain of window functions
that nobody can read and lineage cannot follow.

WHAT IT IS. An ordered fold: a serial pass over a sorted frame in which the
body is ordinary record-tier scalar logic with declared `carries`. It is `Loop`
with the rows as the iteration space instead of a counter. That is the whole
invention, and it costs the framework one combinator rather than a third tier:

    Sort (frame tier, set-shaped)  |  Sweep (record tier, ordered, serial)

Three properties fall out of the shape and they are the ones that matter:

  * THE RANK IS AN ORDINARY OUTPUT. It is produced by the same mechanism as
    `behaviour_grade`, has a producer, a version chain, a tap and a place in the
    evidence set. It is not attached afterwards. This answers s13.2: an account
    learns its own non-selection from its own record.

  * THE CARRIED STATE AT THE MOMENT OF CONSIDERATION IS RECORDABLE. The budget
    remaining when this account was looked at is a per-account column, so "when
    we reached you, R0 of R2.4 bn remained" is on the record for all 708 000
    accounts and nobody re-runs a cycle to answer a complaint.

  * IT IS DETERMINISTIC AND REPLAYABLE. The carry at row i is a pure function of
    rows 0..i-1 in the declared order. Given the same snapshot, the same
    artefact set and the same `decision_date`, the funded set is identical --
    including ranks -- which is s8's determinism row and s10.13's 2031 replay.

THE ACCUMULATOR MUST BE INTEGER. A float64 budget accumulator makes the funding
line depend on summation order, because float addition is not associative. The
carries are int64 cents. This is not fastidiousness: it is the difference
between a reproducible funded set and one that moves when the frame is
re-partitioned.
"""

from decider2 import Sweep, module, param
from decider2.frame import Sort
from clm.vocabulary import ALLOCATION_OUTCOMES, ENVELOPES


# --- the seed --------------------------------------------------------------

def seed_carries(shared) -> tuple:
    """Initial carries from the ALCO instruction, at the offer envelope rather
    than the applied budget: offers are despatched at an over-allocation factor
    (1.38, from a 72.4% trailing take-up rate), so R2.4 bn of applied budget is
    R3.312 bn of offers. The factor is ALCO's parameter, not a quantity the
    programme may derive for itself."""
    pass  # (budget_c * over_allocation_factor, rwa_envelope_c, el_envelope_c, 0, 0)


BudgetSeed = module(seed_carries, name="budget_seed",
                    evidence=["budget_instruction_id", "over_allocation_factor"])


# --- the body: one account, with the population's state carried in ---------
# Every parameter below is either a column (this account) or a carry (the
# population so far). There is no third thing, which is what keeps the body
# plain scalar Python that numba compiles like any other step.

def fits_envelopes(
    additional_limit_c: int,
    expected_rwa_c: int,
    expected_incremental_loss_c: int,
    limit_budget_remaining_c: int,
    rwa_remaining_c: int,
    el_remaining_c: int,
) -> bool:
    """Whether this account's increase fits inside ALL THREE remaining envelopes.
    Any of them may bind before the limit budget does."""
    return (additional_limit_c <= limit_budget_remaining_c
            and expected_rwa_c <= rwa_remaining_c
            and expected_incremental_loss_c <= el_remaining_c)


def binding_envelope_code(
    additional_limit_c: int, expected_rwa_c: int, expected_incremental_loss_c: int,
    limit_budget_remaining_c: int, rwa_remaining_c: int, el_remaining_c: int,
) -> int:
    """Which envelope stopped this account, recorded per account and rolled up
    per cycle (s5.8 constraint 2)."""
    pass  # first of ENVELOPES.tie_order whose headroom is insufficient


def take(
    fits_envelopes: bool,
    tail_skips_used: int,
    max_tail_skips: int = param(2000, ge=0, le=100_000,
                                description="s5.8 constraint 4; after this the cycle stops"),
) -> bool:
    """s5.8 constraint 3: NO INCREASE MAY BE TRIMMED TO FIT. An account is funded
    at the amount s5.5 produced or not funded at all -- a part-funded increase is
    an amount no cell of the matrix produced and no audit could re-derive.

    Constraint 4: at the point where the remaining budget is smaller than the
    next account's increase, skip it, mark it tail-skipped, and continue down
    the ranking taking any account whose increase fits, to a maximum of 2 000
    skips, after which the cycle stops. Back-filling and hard-stopping are both
    defensible; picking one silently is not, so the rule is a param and the
    chosen value is on the record."""
    return fits_envelopes and tail_skips_used < max_tail_skips


# --- carries out: same names as carries in, one version per row ------------
# A sweep body is a scope in which a declared carry may be overwritten exactly
# once, which is doc 03 s3's loop-body rule with the rows as the iteration space.

def limit_budget_remaining_c(limit_budget_remaining_c: int, additional_limit_c: int,
                             take: bool) -> int:
    """Budget after this account. Integer cents; see the module docstring."""
    return limit_budget_remaining_c - additional_limit_c if take else limit_budget_remaining_c


def rwa_remaining_c(rwa_remaining_c: int, expected_rwa_c: int, take: bool) -> int:
    """RWA envelope after this account."""
    pass


def el_remaining_c(el_remaining_c: int, expected_incremental_loss_c: int,
                   take: bool) -> int:
    """Expected-loss envelope after this account."""
    pass


def funded_count(funded_count: int, take: bool) -> int:
    """How many have been funded when this account is considered."""
    return funded_count + 1 if take else funded_count


def tail_skips_used(tail_skips_used: int, fits_envelopes: bool, take: bool) -> int:
    """How many accounts have been skipped for not fitting."""
    return tail_skips_used + 1 if (not fits_envelopes and not take) else tail_skips_used


# --- per-account outputs ---------------------------------------------------

def allocation_outcome_code(take: bool, is_conditional: bool, fits_envelopes: bool,
                            tail_skips_used: int, reserve_exhausted: bool) -> int:
    """funded | conditional_funded | tail_skipped | fairness_capped | below_line.

    `overlay_suppressed` and `below_minimum` are NOT set here: those accounts
    never enter the ranking, and the distinction between them is made in
    offer/construct.py where `proposed_limit_c` and `proposed_limit_unadjusted_c`
    are both in scope. s5.8 requires the two to be distinguishable because they
    are different answers to the client -- one is reversible by withdrawing an
    overlay and the other is not."""
    pass


def funded_amount_c(take: bool, additional_limit_c: int) -> int:
    """The amount funded, or zero. Never a trimmed amount."""
    return additional_limit_c if take else 0


AllocateOne = module(
    fits_envelopes, binding_envelope_code, take,
    limit_budget_remaining_c, rwa_remaining_c, el_remaining_c,
    funded_count, tail_skips_used,
    allocation_outcome_code, funded_amount_c,
    name="allocate_one",
)


Allocate = Sweep(
    AllocateOne,
    seed=BudgetSeed,
    carries=["limit_budget_remaining_c", "rwa_remaining_c", "el_remaining_c",
             "funded_count", "tail_skips_used"],
    requires_order=["rank_key asc", "account_id asc"],   # checked at build, not at run
    emits_position="allocation_rank",
    halts_when="tail_skips_used >= params.max_tail_skips",
    name="allocation",
    evidence=[
        "allocation_rank", "allocation_outcome_code", "funded_amount_c",
        "binding_envelope_code",
        # the carried state AT CONSIDERATION -- the whole non-selection answer
        "limit_budget_remaining_c@entry", "funded_count@entry",
    ],
)

# The pipeline fragment. `Sort` is frame tier and genuinely set-shaped; the
# Sweep declares the order it requires and the build fails if the preceding
# frame operation does not establish it. Ordering is part of the schema
# (FRAMEWORK-DEMANDS #3) -- a `Join` between the Sort and the Sweep would
# silently destroy it, and silently is the problem.
Allocation = Sort(by=["rank_key", "account_id"], descending=[False, False]) | Allocate
