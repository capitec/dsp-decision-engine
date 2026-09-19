"""The five matrix dimensions, and the fact that their edges are parameters.

Spec §4.3: "Bands are parameters, not constants, and their edges move."

So a dimension is declared as a `Band` — a named, ordered set of edges that is
resolved from params at invocation, not baked in. Three consequences, all of
which the spec asks for somewhere:

  * Moving the balance band 3/4 edge is a VALUE change (doc 08 §2 class 1).
    Free, no recompile, and the impact report says how many accounts moved cell.

  * Adding a ninth arrears bucket (change scenario 1) is an INTERIOR change:
    the band gains an edge, the grid gains 672 cells, the validator names every
    unpopulated one, and `matrix_cell_id` is renumbered. Which is why cell ids
    carry a `banding_version` — see `cell_id()` below. History recorded under
    the old banding stays interpretable because the old banding_version still
    resolves.

  * A band overlay (overlays/register.json, kind `band_edge_shift`) moves an
    edge for a scoped population WITHOUT changing the band definition. The
    unadjusted band survives beside the adjusted one, per spec §5.3.
"""

from decider2 import Band, param, step
from decider2.types import cents, f8, i1, i2

ARREARS_BUCKET = Band(
    name="arrears_bucket_code",
    over="days_past_due",
    edges=param([1, 15, 30, 60, 90, 120, 180, 365],
                owner="collections_strategy", cadence="semi_annual"),
    labels=[1, 2, 3, 4, 5, 6, 7, 8],
    # change scenario 1 splits 180-364: edges becomes [1,15,30,60,90,120,180,270,365]
    # and labels [1..9]. One line, then 672 cells to populate and a validator
    # that will not let the release ship until they are.
)

COLLECTIONS_BAND = Band(
    name="collections_band_code",
    over="collections_score",
    edges=param([0.04, 0.09, 0.17, 0.31, 0.52], owner="model_team", cadence="on_release"),
    labels=[1, 2, 3, 4, 5, 6],
    overlayable=True,      # band_edge_shift overlays may move these. See overlays/surfaces.py.
)

BALANCE_BAND = Band(
    name="balance_band_code",
    over="outstanding_balance",
    edges=param([250_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000, 25_000_000],
                owner="collections_strategy", cadence="semi_annual"),
    labels=[1, 2, 3, 4, 5, 6, 7],
    units="cents",         # R2 500 = 250 000 cents. Money is int64 cents (doc 03 §1.2).
)

CONTACT_BAND = Band(
    name="contact_band_code",
    over="contact_responsiveness_index",
    edges=param([0.12, 0.35, 0.72], owner="collections_strategy", cadence="quarterly"),
    labels=[1, 2, 3, 4],   # 1 responsive, 2 intermittent, 3 silent, 4 unreachable
)

PRODUCT_FAMILY = Band.categorical(
    name="product_family_code",
    labels=[1, 2, 3, 4],   # unsecured term, revolving, secured asset, business
)

MATRIX_KEY = (ARREARS_BUCKET, COLLECTIONS_BAND, BALANCE_BAND, CONTACT_BAND, PRODUCT_FAMILY)
# 8 x 6 x 7 x 4 x 4 = 5 376


@step(output="matrix_cell_id")
def cell_id(
    arrears_bucket_code: i1,
    collections_band_code: i1,
    balance_band_code: i1,
    contact_band_code: i1,
    product_family_code: i1,
    banding_version: i2,
) -> i2:
    """Dense linear index over MATRIX_KEY, namespaced by banding_version.

    `banding_version` is not decoration. Cell 3417 under banding 4 and cell 3417
    under banding 5 are different populations, and eighteen months of outcome
    attribution joined on a bare cell id would silently mix them. Change
    scenario 1 is exactly this hazard.
    """
    pass


@step(output="on_band_edge")
def band_edge_flag(
    days_past_due: i2, collections_score: f8, outstanding_balance: cents,
    contact_responsiveness_index: f8,
) -> bool:
    """Spec §5.4 requires recording "where a key value fell on a band edge or
    outside the banded range". One bool, always on, and it is what turns a
    disputed cell assignment from an argument into a query."""
    pass
