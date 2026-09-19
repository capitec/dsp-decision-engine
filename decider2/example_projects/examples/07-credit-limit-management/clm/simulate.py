"""Simulation (s5.10). There is no simulation pipeline.

s10.3 requires that simulation and production share one implementation and that
there be no second expression of the matrix, the caps, the affordability logic,
the overlays or the allocation anywhere in the system. The only way to keep
that promise is to make simulation an INVOCATION rather than a MODE: the same
`programme` object, applied twice, with a different artefact set.

    a = programme.apply(book, params=p, shared=s, tables=live,      origin=...)
    b = programme.apply(book, params=p, shared=s, tables=candidate, origin=...)
    report = swap_set(a, b, on="account_id")

Everything below is the reporting around those three lines. None of it contains
a threshold, a cap, a multiplier or a rank.

WHY THIS WORKS AT ALL: because `tables=` is an argument. If the matrix were
imported, or read from a path baked into a module, or resolved from ambient
state, a candidate run would require a deployment and the second implementation
would appear within a quarter. FRAMEWORK-DEMANDS #2.
"""

from datetime import date

import polars as pl
from decider2 import artefacts, diff, impact
from decider2.frame import Join, declares

from clm.pipelines.programme import programme


def resolve_candidate(decision_date: date, overrides: dict, overlays: str = "live"):
    """The artefact set a policy analyst names on the command line.

    `overlays="off"` runs the whole book with the stack disabled, which is how
    Model Risk observes the scorecards' own behaviour (s5.10 requirement 2) and
    how Credit Committee answers "what would we have done without it".
    """
    return artefacts.resolve(decision_date=decision_date, selector="live",
                             override=overrides, overlays=overlays)


def self_check(snapshot_id: str, decision_date: date) -> dict:
    """s5.10 requirement 3. Simulation run over the CURRENT production artefact
    set and the LAST production snapshot must reproduce the last production
    cycle account for account, to the rand, including ranks and the funded set.

    Runs before every simulation session; its result is part of the output. A
    simulation that cannot reproduce production is not evidence about
    production.
    """
    pass  # apply(live) vs the stored cycle record; assert_frame_equal, no tolerance


@declares(outputs={"swap_set": pl.DataFrame})
def swap_set(base: pl.LazyFrame, candidate: pl.LazyFrame) -> pl.LazyFrame:
    """Gainers, losers and unaffected, with counts, rand amounts and mean change
    per segment. NOT a before-total and an after-total: a per-account
    classification, because that is what Credit Committee approves.

    A full-outer join on `account_id` of two runs of the same pipeline. It is a
    frame operation over two decision records, so it carries no policy and it
    cannot drift from the thing it is comparing.
    """
    pass


REPORTS = {
    # s5.10's required outputs, each a frame operation over the two records.
    "bucket_migration": ("matrix_cell_id", "matrix_cell_id"),      # ~40 material transitions
    "additional_limit": ("proposed", "offered", "funded"),
    "expected_exposure": ("additional_limit_c", "credit_conversion_factor"),
    "expected_loss_and_rwa": ("expected_incremental_loss_c", "expected_rwa_c"),
    "distribution": ("behaviour_grade", "utilisation_band", "mob_band", "product_code",
                     "income_decile", "months_on_book", "region_code"),
    "swap_set": ("per_account",),
    "cap_incidence": ("binding_cap_code",),
    "funding_line": ("rank_key_at_line", "funded_count", "binding_envelope_code"),
    "overlay_attribution": ("adjustments_applied",),
}
# The last one is the question a policy analyst actually asks most often, and
# it is not "what does the new matrix do" -- it is "what does THIS MONTH'S DIAL
# cost us". Answering that with a separate calculation is unacceptable for the
# same reasons as everything else here, with the added hazard that a dial is set
# in days rather than quarters and will never get a spreadsheet of its own. It
# is `resolve_candidate(overlays="off")` against `overlays="live"`, and it is
# the same three lines.


def backtest(snapshot_id: str, decision_date: date, overrides: dict) -> pl.DataFrame:
    """s5.10 requirement 4. Run a candidate set over a 12-month-old snapshot
    with the subsequent 12 months of realised performance attached, so
    "accounts this matrix would have funded" can be scored against what those
    accounts actually did.

    Note what makes this possible and what it costs: artefact resolution is
    keyed on `decision_date`, so a backtest is an ordinary run at an old date.
    The catch is `PriorCycle` in sources/book.py -- the hysteresis inputs come
    from the cycle before the backtest date, so a backtest is a CHAIN of
    resolutions, not a point. FRAMEWORK-DEMANDS #17.
    """
    pass


# --- the CLI a policy analyst runs, without an engineer (s10.1) ------------
#
#   clm simulate \
#       --candidate clm.limit_matrix=2026Q4-candidate-c \
#       --snapshot 2026-08 --decision-date 2026-09-01 \
#       --against live --report swap-set,cap-incidence,funding-line
#
#   clm simulate --overlays off --snapshot 2026-08        # the unadjusted book
#   clm matrix diff 2026Q3 2026Q4-candidate-c             # cell by cell, rand impact
#   clm matrix validate artefacts/limit_matrix/2026Q4_candidate_c.csv
#   clm overlays expiring --within 30d                    # the monthly control report
#
# Budget: the pipeline over 4.1 M accounts is minutes of compute. The cost is
# reading the book, so the snapshot is a column-pruned Parquet artefact that
# both production and simulation read, and the twenty-minute ceiling is an I/O
# budget rather than a compute one.
