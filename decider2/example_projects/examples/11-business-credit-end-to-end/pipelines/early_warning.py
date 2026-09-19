"""EP-5 -- L3. 240 000 facilities, daily, 3 hours, with fan-out. Spec 5.6.3.

The largest single workload in the estate, and the one that consumes components
sized for a different world: `core.adverse_events` and project 05's
classification and roll-up were written for a 900-a-day origination path, and
this calls them ~9 400 times a day as singletons plus 48 000 entity re-rolls.

Spec 8.3 tension 2: "making them fast for E without forking is the requirement;
forking them is the failure."
"""

from decider2 import parallel, fuse, incremental
from decider2.frame import Join, Aggregate
from modules.watchlist.signals import Watchlist, EscalationAsymmetry, signal_catalogue
from consumed.p05_origination import EntityAssessment, PartialReBlend
from consumed.core_library import adverse_events
from roles import SIGNAL_SCORE

# --------------------------------------------------------------------------
# The fan-out is a frame-tier join, not a record-tier loop.
#
# entity -> business -> facility. Mean 1.4 businesses per entity, mean 1.3
# facilities each; p99 entity touches 31 facilities and 340 entities touch more
# than 100. A record-tier loop over "every facility this entity touches" would
# put a variable-length collection inside a kernel for the sake of a join the
# frame tier does in one pass.
#
# Declared frame ops keep lineage (doc 02 5). "What can affect watchlist_grade"
# still answers statically across the join, which spec 5.16's N2 requires at a
# five-minute pass mark.
# --------------------------------------------------------------------------
FanOut = (
    Join("entity_attachments", on="entity_key")       # 1 116 000 attachments
    | Join("business_facilities", on="client_id")     # -> 240 000 facilities
)

DailyPass = (
    adverse_events.singleton()        # 9 400/day, same implementation as WHOLE
    | EntityAssessment.partial(unit="entity")          # 48 000/day
    | PartialReBlend                                   # 13 200/day
    | FanOut
    | fuse(Watchlist | EscalationAsymmetry)
)

# --------------------------------------------------------------------------
# Incremental, and PROVABLY EQUIVALENT to a full pass. Spec 5.6.3 requirement 2.
#
# Evaluating 240 000 facilities x 186 signals nightly is ~45 M signal
# evaluations and does not fit the window at the fan-out involved. So only
# subjects with a changed signal set, or a signal crossing a DECAY BAND
# BOUNDARY, are re-evaluated -- and the decay boundary is the one people forget,
# because nothing changed and the answer still moves.
#
# `incremental()` takes the dirty-set predicate and the reconciliation cadence,
# and generates the weekly full pass AND the comparison. Any facility whose
# grade differs between incremental and full is a defect, reported. Spec 8.2
# "cross-run consistency" and spec 10 acceptance 33.
#
# This is the same proof obligation as `EntityAssessment.partial`'s
# `equivalent_to="whole"` -- one construct, two scales -- which is 05 13-Q8
# ("what is the smallest re-run that is provably equivalent to a full one, and
# how is that proof obtained?") turned from a design question into an
# operational one.
# --------------------------------------------------------------------------
EarlyWarning = incremental(
    DailyPass,
    dirty=["changed_signal_set", "decay_band_boundary_crossed"],
    reconcile="weekly full pass",
    on_disagreement="defect, reported and investigated, facility named",
    expected={"could_change": 31_000, "actually_change": 2_400},
)

# The only `parallel()` in the project. Spec 8.1 profile E is the one workload
# whose per-row cost is close to uniform: 186 signals evaluated over a subject,
# of which ~22 have any data, with no early exit worth short-circuiting for.
# Doc 02 3.3's objection to automatic prange -- that real credit logic
# short-circuits, so per-row work is a distribution -- does not apply here, and
# saying so at the one site it does not apply is the point of `parallel()` being
# authored rather than inferred.
EarlyWarning = parallel(EarlyWarning, over="facilities")

# Spec 5.6.2 rule 3. Three signal scopes, one grade. The roll-up is declared in
# roles.py (SIGNAL_SCORE) rather than here, so the fan-out semantics are in one
# place and both EP-5 and EP-8 read the same declaration -- a cascade that
# aggregated signals differently from the daily pass would produce two watchlist
# grades for one facility on one night.
EarlyWarning = EarlyWarning.rolling_up(SIGNAL_SCORE)
