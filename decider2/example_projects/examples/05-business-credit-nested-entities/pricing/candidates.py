"""Stage 5.12a -- enumerate the pricing candidate space.

**This is the second-biggest departure from doc 03 in the sketch, after the
grain itself.** doc 03 s8.3 offers `Loop(should_continue, Body, carries=[...],
max_iterations=N)` for iteration. This project does not use it, anywhere.

s5.12 states four requirements that a carried-accumulator loop cannot satisfy
together:

    "Whatever search is used must return the same answer as exhaustive
     evaluation of the admissible candidate set"
    "bounded and terminating: a declared maximum number of candidate
     evaluations, a declared ordering, and a declared outcome when the maximum
     is reached"
    "Records: every candidate evaluated, in order, with its rate cell,
     instalment, DSCR and rejection reason"
    "because the instalment is not monotone in the amount, bisection on the
     amount is INVALID, and an implementation that assumes monotonicity will
     quietly return a smaller offer than the Bank would have made"

A `Loop` gives you the third only if the body writes a side record (which breaks
purity), and it gives you the first only by proof over 5 000 historical
applications -- which is exactly what the spec asks for, and exactly what a
design should make unnecessary.

Enumerating the space as a grain gives all four by construction:

    exhaustive      the fold sees every candidate; there is no traversal to get
                    wrong and nothing to bisect
    bounded         `Candidate.capacity` is 2 200, declared in grains.py, and a
                    breach raises `search_truncated` with the declared outcome
    recorded        the candidates ARE a frame; "why not R2 000 000?" is a row
    ordered         `Candidate.order` is the canonical materialisation order, so
                    the persisted ledger is byte-identical on replay

And the monotonicity question never arises, because nothing anywhere assumes it.
"""

import polars as pl
from decider2 import module, param, step, table, Enumerate, cross
from grains import Application, Candidate

RATE_CARD_BANDS = table(
    "business_rate_card_amount_bands",
    key="amount_band_index",
    columns=("band_floor", "band_ceiling"),
    source="tables/rate_card_bands.csv",
    effective_dated=True,
)

PERMITTED_TERMS = table(
    "permitted_terms",
    key=("product_code", "facility_purpose_code"),
    columns=("min_term_months", "max_term_months"),
    source="tables/permitted_terms.csv",
    effective_dated=True,
)


# --------------------------------------------------------------------------
# Generation. Frame tier: a declared cross product, filtered by the ceilings
# that do NOT depend on the amount, so the enumerated set is 200-600 rows rather
# than 2 200 in the typical case (s5.12). The filter is declared rather than
# imperative so that lineage survives and so that the *reason* a candidate was
# never enumerated is as recoverable as the reason one was rejected.
#
# Note `sector_max_term` and `asset_life_term_cap`: s5.12 says "the sector's
# maximum term may be lower -- asset life caps apply to equipment purposes".
# That is a per-application ceiling on a per-candidate coordinate, so it belongs
# in generation and not in evaluation.
# --------------------------------------------------------------------------
EnumerateCandidates = Enumerate(
    Candidate,
    from_=cross(RATE_CARD_BANDS, PERMITTED_TERMS),
    where=[
        pl.col("band_floor") <= pl.col("requested_amount"),
        pl.col("band_floor") >= pl.col("product_minimum_amount"),
        pl.col("band_ceiling") <= pl.col("product_maximum_amount"),
        pl.col("term_months") <= pl.col("sector_max_term"),
        pl.col("term_months") <= pl.col("asset_life_term_cap"),
    ],
    emit=["amount_band_index", "band_floor", "band_ceiling", "term_months"],
    name="enumerate_candidates",
    on_empty="flag:no_admissible_candidate",     # -> decline, reason 5501
)


def product_notional_term(
    product_code: int,
    revolving_amortisation_months: float = param(
        36.0, ge=1.0, le=120.0,
        description="s5.12: for product 51 the term is notional; the coverage "
                    "test uses an amortisation of the full limit over 36 months"),
) -> float:
    """Product 51 has no contractual term. It still needs one to price against,
    and the number used is a policy parameter rather than a constant buried in
    the coverage test."""
    pass


CandidateContext = module(
    product_notional_term, name="candidate_context", grain=Application)
