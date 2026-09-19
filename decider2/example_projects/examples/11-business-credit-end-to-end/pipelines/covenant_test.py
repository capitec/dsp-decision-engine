"""EP-4 -- L2. 2.4 M tests a year. The pipeline that runs almost none of the flow.

Spec 5.1's table: EP-4 runs L2 in full, O7 partially (spreading of the delivered
figures only, ON THE COVENANT'S OWN DEFINITIONS), and NOTHING ELSE. Sixteen of
the seventeen origination phases do not run.

That is why an entry point cannot be a wrapper that calls the flow and
post-processes. This pipeline is three lines and it is 148 decision points.
"""

from decider2 import parallel, batch_pin
from decider2.frame import Join
from modules.covenants.schedule import DueTests
from modules.covenants.test import CovenantTest
from modules.covenants.waiver import Waivers
from modules.watchlist.signals import Watchlist
from time.bitemporal import ACTUAL

# --------------------------------------------------------------------------
# Frame tier: the schedule expands instances into due test rows, carrying the
# binding. Record tier: an ordinary module over those rows.
#
# The expansion is where the 13x seasonal peak lives and where it is visible
# BEFORE the batch runs (modules/covenants/schedule.py), which is what lets the
# scheduler smooth 310 000 tests in the peak month instead of discovering them.
# --------------------------------------------------------------------------
CovenantTesting = (
    DueTests
    | Join("certificates", on="covenant_instance_id", how="left")
    | CovenantTest              # contractual=True: no Dated[T] read permitted
    | Waivers
    | Watchlist.signals_only()  # covenant proximity and breach: 17 of the 186
)

# An ownership-change covenant is the one test that reads the ACTUAL view.
# Spec 5.13.3: it tests the register as at the test date against a baseline
# PINNED AT ORIGINATION in the instance. Two temporal reads in one test, from
# two different sources, and the namespace rule is what keeps them apart:
#   actual.entities@test_date   vs   instance.ownership_baseline
OwnershipChangeTest = CovenantTesting.as_true_at("test_date").reading(ACTUAL)

# Spec 5.13.3(3). A late-arriving fact that retrospectively breaches a covenant
# already tested and passed produces a RE-TEST, not a silent correction. The
# original test stands as the test that was performed; the re-test is a new
# record with a new determination date; and the difference between them is the
# Bank's knowledge, WHICH IS ITSELF A FINDING.
#
# `re_test_on` is a declared consequence edge, the same construct as
# `Allocation.reopens`: a late fact re-opens a closed test the way a security
# change re-opens a sibling facility. One mechanism, two uses, which is the
# argument for it being a framework construct at all.
CovenantTesting = CovenantTesting.re_test_on(
    "late_effective_dated_fact",
    supersedes=False,           # the original test record is never touched
    emits="knowledge_gap_finding",
)

# Profile D (spec 8.1). No external calls -- the certificate IS the input. The
# peak month is 310 000 tests against a mean of 195 000, so the nightly run is
# budgeted at 4 hours against the peak, not the mean.
#
# `parallel()` is NOT applied here despite the volume: each test is cheap and
# heterogeneous (a DSCR test and a negative-pledge test share almost no work),
# so per-row cost is a wide distribution and doc 02 3.3's rationale applies --
# "real credit logic short-circuits, so per-row work is a distribution rather
# than a constant". The batching is at the frame tier instead.
CovenantTesting = batch_pin(CovenantTesting, at="run_start",
                            covers=["generation", "component_manifest"])
# NOTE: `dated_tables` is deliberately absent from `covers` here, because this
# pipeline reads none. That absence is checked (time/dating.py R2), and it is
# the shortest proof in the project that the contractual discipline holds.
