"""The candidate plan: which <=400 of 9 437 148 scenarios get evaluated, in what order.

THIS IS THE CENTRAL CLAIM OF THE WHOLE SKETCH, so it is stated before any code.

    The plan is FRAME-SHAPED. The evaluation is RECORD-SHAPED.
    The batch is not clients. The batch is hypotheses.

Doc 02 1 splits the world into a record tier (per-record decision logic, numba)
and a frame tier (things inherently about SETS of rows, polars). Candidate
enumeration is inherently about sets: it is a cross product, a sort, a dedupe
and a truncation over eighteen accounts, four products and four terms. Every one
of those is a frame operation with a known schema transform. So the plan is a
polars pipeline that emits a 400-row frame, one row per scenario, and it runs
ONCE.

What that buys, and it is the reason the 900ms budget is achievable at all:

    "400 affordability recomputations" is not 400 calls to core.affordability.
    It is ONE call, over a 400-row frame.

core.affordability is a record-tier module. `apply()`ing it to a frame of 400
candidate rows compiles to one kernel invocation over 400 rows. At doc 01's
measured 0.20 ns/step for split kernels, the arithmetic is not the problem and
never was; the problem was only ever an implementation that made 400 scalar
calls because it thought of the search as a loop.

And the invariant-income requirement (spec 5.6.1) stops being a discipline
somebody has to remember and becomes A PROPERTY OF THE FRAME:

    gross_monthly_income, net_monthly_income, living_expenses, statutory_
    deductions, income_source_code, income_haircut_applied, expense_basis_code
    and dependants_count are CONSTANT COLUMNS in the candidate frame, broadcast
    from a one-row assessment frame by the cross join that builds it.

They cannot vary per scenario because there is nothing in the plan that could
vary them. `Search(invariant=[...])` asserts it anyway - a runtime check that
each named column has exactly one distinct value across the frame - because
acceptance criterion 8 demands the property be PROVABLE, not merely true. The
check is one `n_unique()` per column per assessment and it is the cheapest
acceptance criterion in the whole spec.

See FRAMEWORK-DEMANDS D2 for what this asks of the framework that doc 02 does
not currently offer.
"""

from decider2 import Branch, Loop, module, param, step
from decider2.frame import Aggregate, Cross, Explode, Fanout, Filter, Join
from decider2.search import Plan, candidate_key
from decider2.values import na

from products.routing import ProductRouting
from search.orderings import RankAccounts


# --- settlement sets are a uint64 bitmask -----------------------------------
#
# Opinionated and load-bearing. A settlement set is a subset of the settleable
# accounts, so index the settleable accounts 0..63 by ordering_rank and let a
# set be a uint64. Four consequences, all of them wanted:
#
#   - dedupe is `unique()` on one integer column, not a set comparison;
#   - membership is a bit test, which compiles;
#   - the candidate key is small enough that the full plan of 400 rows fits in
#     a few kilobytes, which is what makes persisting the plan affordable at
#     6 000 assessments a day;
#   - two assessments explored the same space iff their sorted mask lists match,
#     which gives project 09 a replay check that costs nothing.
#
# The bound is declared, not hoped for: above `max_settleable_indexed` (64) the
# plan keeps the top 64 by H2 and records `index_truncated` with the dropped
# account_refs. Observed maximum inventory is 61 accounts total, of which far
# fewer are settleable, and CON-INT-01 caps accounts SETTLED at 8 regardless.


@step(output="settleable_index")
def index_settleable(
    account_ref: int,
    settleability_code: int,
    ordering_rank: int,
    max_settleable_indexed: int = param(64, ge=8, le=64),
) -> int:
    """Assign each settleable account a bit position 0..63, by H2 rank, stably."""
    pass  # dense rank over settleable accounts ordered by relief_per_rand desc


@step(output="mandatory_mask")
def mandatory_mask(settleable_index: int, is_client_nominated: bool) -> int:
    """Bits the client insists on. A scenario omitting any of them is not generated.

    Spec 5.2: client nominations and exclusions are applied HERE, not in the
    search. An exclusion removes the account from indexing entirely; a
    nomination sets a bit every generated mask must contain. A mandatory account
    that is not settleable is a HARD CONFLICT surfaced to the consultant, and it
    is detected here because here is where the two facts first meet.
    """
    pass  # OR of bits for nominated accounts; conflicts raise CON-CONF-01


# --- generating the settlement sets ------------------------------------------
#
# Spec 5.5 constrains what must be present among the candidates rather than how
# they are enumerated. Each constraint is one frame operation.


GenerateSettlementSets = (
    # the empty set - the baseline, so the chosen outcome is always compared to
    # doing nothing. SEED-EMPTY.
    Plan.seed("empty", mask=0)
    # the client's nominated set exactly as nominated. SEED-NOM.
    | Plan.seed("nominated", mask="mandatory_mask")
    # the full settleable set. Frequently unaffordable; it is the client's mental
    # model and must be priced so it can be shown to have been. SEED-FULL.
    | Plan.seed("full", mask="all_settleable_mask")
    # for each ENABLED ordering rule, the prefixes of that ordering: top 1, top
    # 2, ... top k. Spec 5.5: "a business-authored ordering that never produces a
    # candidate is a rule nobody can validate."
    | Plan.prefixes(of=RankAccounts, depth="prefix_depth")
    # and each of those prefixes UNIONED with the mandatory mask. SEED-NOM-EXT.
    | Plan.extend(by="mandatory_mask")
    # a mandatory account means only sets that contain it.
    | Filter.masks(contains="mandatory_mask")
    # H4 is a constraint rule, not an ordering: secured accounts whose security
    # does not release or transfer are removed from every candidate pool.
    | Filter.masks(excludes="h4_blocked_mask")
    # dedupe. Eight orderings over eighteen accounts overlap heavily - typically
    # 60-70% of generated masks are duplicates, which is the single biggest
    # saving in the whole plan and the reason the bitmask representation pays.
    | Plan.distinct(on="settlement_set_mask")
)


# --- routing: one settlement set, up to four priced outcomes -----------------
#
# Spec 5.6.3. Where two or more products can carry a set, ALL of them are
# candidate scenarios and all are evaluated.
#
# This is a one-to-many expansion, and doc 03's Branch is one-to-one: it picks an
# arm. So routing is NOT a Branch. It is a frame-tier JOIN against a declared
# routing table - `Fanout` - which has a known schema transform and therefore
# keeps lineage (doc 02 5). The heterogeneity is handled later, by a Branch over
# product_code inside the kernel, where each ROW takes exactly one arm.
#
#   routing   = set-shaped   -> Fanout, frame tier
#   pricing   = record-shaped -> Branch, record tier
#
# That is two existing constructs used correctly rather than a new one, and it
# is the part of this sketch that needed the least invention.

RouteToProducts = Fanout(
    on="settlement_set_mask",
    table=ProductRouting,
    produces=["product_code", "may_carry_reason_code"],
    # A set that only ONE product can carry is not thereby preferred. It is
    # simply cheaper to evaluate. Nothing here encodes a preference and that is
    # deliberate: preference is the objective's job, and mixing the two is how a
    # search quietly acquires a second objective nobody approved.
)


# --- terms, bounded -----------------------------------------------------------
#
# Nine candidate terms per (set, product) would spend the whole budget on term
# variation. The plan takes four per product, DETERMINISTICALLY:
#
#   product minimum | affordability bracket low | bracket high | CON-INT-05 max
#
# The bracket is a closed-form first estimate of the term at which the instalment
# crosses max_affordable_instalment, at the band-centre rate. It is an estimate,
# it is allowed to be wrong, and when it is wrong the scenario is simply rejected
# by affordability like any other. What matters is that it is a pure function of
# recorded inputs, so a replay brackets identically.
#
# Product 20 has no term. It does not get a term axis; it gets a PROMOTIONAL
# DURATION axis, supplied by the product arm. The plan asks the product for its
# own candidate axis rather than assuming every product has a term - which is
# FRAMEWORK-DEMANDS D4's not-applicable problem showing up two stages before
# anyone expects it.


@step(output="candidate_terms")
def bracket_terms(
    product_code: int,
    product_min_term: int,
    con_int_05_max_term: int,
    settlement_total: float,
    max_affordable_instalment: float,
    band_centre_rate: float,
) -> list[int]:
    """Four terms per (set, product): min, two bracketing affordability, CON-INT-05 max.

    Returns na() for product 20, whose axis is promotional duration.
    """
    pass  # closed-form term-at-instalment solve, snapped to the product's term set


ExpandTerms = Explode(
    column="candidate_terms",
    into="term_months",
    when_absent=na(),  # product 20: one row, term_months = na()
) | Fanout(
    on="product_code",
    table="promotional_duration_axis",
    produces=["promotional_months"],
    only_for=[20],
)


# --- ordering, truncation, tiering -------------------------------------------
#
# THE PLAN IS ORDERED BEFORE IT IS TRUNCATED, and truncation is by count.
#
# Interleaving matters and is a policy decision, not an engineering one: taking
# the first 400 rows of "all of H1's candidates, then all of H2's" means H8 never
# gets evaluated for a client with many accounts. The plan therefore ROUND-ROBINS
# across enabled orderings by their declared `sequence`, so every enabled rule
# contributes before any rule contributes twice. That is what makes acceptance
# criterion 14 possible - an analyst can see their rule produced candidates.

OrderAndTruncate = Plan.interleave(
    by="ordering_sequence",
    tie_break=candidate_key("settlement_set_mask", "product_code", "term_months"),
    stable=True,
) | Plan.truncate(to="budget.candidates") | Plan.tier(by="budget.tiers")


# --- the whole plan -----------------------------------------------------------

ConsolidationPlan = (
    RankAccounts
    | module(index_settleable, mandatory_mask, name="settleable_index")
    | GenerateSettlementSets
    | RouteToProducts
    | module(bracket_terms, name="term_brackets")
    | ExpandTerms
    | OrderAndTruncate
    # The invariant assessment values are attached HERE, by a cross join from a
    # one-row frame. After this line, income is constant across the plan because
    # there is no operation left that could vary it.
    | Cross(source="assessment_invariants", broadcast=True)
)


# --- the restructure plan shares everything after generation ------------------
#
# Spec 5.9: "the machinery of 5.5 to 5.8 is reused unchanged". The only thing
# that differs is what a candidate IS - a settlement subset, or a combination of
# concessions over existing agreements. So the two plans differ in one stage and
# share the other six, and the flow selects between them with a Branch on
# assessment_mode_code (pipelines/restructure.py).
#
# Combinations are options: a payment holiday followed by a term extension is a
# distinct option from either alone. That multiplies the space again, and it is
# why the restructure budget declares 260 rather than 400 - the per-candidate
# evaluation is more expensive, not less numerous.
