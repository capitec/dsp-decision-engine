"""The whole flow, in one expression. Products 50 and 51.

Read this file second, after `grains.py`. The thirteen stages of spec s5 are the
thirteen groups below, and the *grain shifts* -- the thing the whole project is
about -- are the `Each(...)` and `Gather(...)` lines. There are nine of them and
they are the only places the nesting is visible.

Everything between two grain shifts is ordinary `decider2`: pure scalar steps,
`|`, `Branch`, per-module params. No step in this project loops over a
collection. Not one.
"""

from decider2 import Each, Gather, Branch, Shadow, fuse, parallel
from decider2.frame import Join

from grains import Application, Entity, Event, Candidate
from vocabulary import vocabulary

# stage 5.1 -- frame tier: the disclosed graph -> a bounded, de-duplicated tree
from structure.resolve import ResolveStructure, ReconcileOwnership

# stage 5.2 -- 5.3, application grain
from business.regime import RegulatoryRegime
from business.disqualification import BusinessDisqualification

# stage 5.4, entity grain
from entities.criticality import Criticality
from entities.disqualification import EntityDisqualification, SyntheticEvents

# stage 5.5, event grain
from entities.adverse.classify import ClassifyEvent
from entities.adverse.flags import EventFlags
from entities.adverse.gather import EntityAdverseFacts
from entities.adverse.verdict import EntityAdverseVerdict

# stage 5.7, entity grain
from entities.scoring.families import EntityScoring

# stage 5.8, entity -> application
from people.weights import BaseWeight, PeopleCoverage, NormaliseWeight, WeightTotals
from people.blend import PeopleBlendFacts, PeopleBlend
from people.caps import PeopleCaps
from people.surety import SuretyCoverFacts, GuaranteeCoverFacts, SecurityPosition

# stages 5.9 -- 5.11, application grain
from financial.measures import FinancialAssessment
from grade.combine import BusinessGrade
from grade.overlay_stack import STACK, UnadjustedSpine
from people.surety import AppetiteAndExposure

# stage 5.12, candidate grain
from pricing.candidates import EnumerateCandidates
from pricing.price_one import PriceCandidate
from pricing.select import SelectOffer, SelectedOffer

# stage 5.13, application grain
from validation.consistency import FinalValidation, ConditionsAndCovenants
from evidence.attribution import AttributionSpine
from evidence.disclosure import ReasonSets
from evidence.counterfactual import DisputeShadow


# ==========================================================================
# 5.1  Structure resolution. Frame tier, because collapsing a graph is
#      set-shaped work: three bounded join-expand-antijoin rounds.
#      Emits the Entity and Event frames the rest of the pipeline is written
#      against. See structure/resolve.py for why this is not `@breaks_lineage`.
# ==========================================================================
_structure = ResolveStructure | ReconcileOwnership


# ==========================================================================
# 5.2 -- 5.3  Business level. Every rule evaluated even after the first fails
#      (s5.3: "the outcome short-circuits the remainder of the flow, not the
#      rule set"), which is `collect="all"` on the verdict -- see
#      business/disqualification.py.
# ==========================================================================
_business = RegulatoryRegime | BusinessDisqualification


# ==========================================================================
# 5.4 -- 5.6  The nested core. Five grain shifts, in this order, and the order
#      is the dependency: criticality is computed at the Entity grain and
#      *broadcast down* to the Event grain, because the event thresholds depend
#      on the criticality of the entity the event hangs off (s5.5). That is
#      spec s13 Q3, and the answer is "a broadcast value, not a wrapped
#      capability".
# ==========================================================================
_nested = (
      Each(Entity, Criticality | EntityDisqualification)
    | SyntheticEvents                              # s5.4: "material adverse event"
                                                   # dispositions are injected into
                                                   # the Event frame, marked synthetic,
                                                   # so they participate in the
                                                   # count- and amount-based rules
                                                   # rather than sitting outside them
    | Each(Event, fuse(ClassifyEvent | EventFlags))
    | EntityAdverseFacts                           # Gather(Event -> Entity)
    | Each(Entity, EntityAdverseVerdict | EntityScoring)
)


# ==========================================================================
# 5.8  The people component. Four separate Gathers over the *same* Entity
#      collection, because three consumers roll it up three different ways and
#      the coverage test rolls it up a fourth. None of them is worst-of and
#      none of them is an average.
#
#      Note the two-pass shape: BaseWeight (entity) -> WeightTotals (gather) ->
#      NormaliseWeight (entity, reading the total broadcast back down) ->
#      PeopleBlendFacts (gather). PP-04's control weighting needs the total
#      before it can renormalise, and this is what that looks like without an
#      accumulator and therefore without an order dependency.
# ==========================================================================
_people = (
      Each(Entity, BaseWeight)
    | WeightTotals                                 # Gather(Entity -> Application)
    | PeopleCoverage                               # PP-02, its own Gather
    | Each(Entity, NormaliseWeight)                # PP-03/PP-04, reads the total
    | PeopleBlendFacts                             # PP-05, Gather on log-odds
    | PeopleBlend
    | PeopleCaps                                   # PP-06, worst-of *after* the blend
    | SuretyCoverFacts                             # PP-08, a different Gather
    | GuaranteeCoverFacts                          # PP-09, a fourth Gather
)


# ==========================================================================
# 5.9 -- 5.11  Financials, the combined grade, the ceilings.
#      `UnadjustedSpine` is a Shadow of everything from 5.5 to 5.10 with the
#      overlay stack off. It is not a second implementation -- it is the same
#      modules, declared once, evaluated twice. spec s5.8 PP-11 requires
#      people_pd, people_pd_unadjusted and the decomposition between them.
# ==========================================================================
_grade = (
      FinancialAssessment
    | BusinessGrade
    | UnadjustedSpine                              # Shadow(..., overlays=Off)
    | SecurityPosition
    | AppetiteAndExposure
)


# ==========================================================================
# 5.12  Pricing. The candidate space is a grain, so there is no search: there
#      is an enumeration and a fold. Non-monotonicity stops being a hazard
#      because nothing assumes monotonicity; exhaustiveness is by construction
#      rather than by a 5 000-application proof; and every candidate is recorded
#      because the candidates *are* a frame.
# ==========================================================================
_pricing = (
      EnumerateCandidates                          # Application -> Candidate
    | Each(Candidate, parallel(PriceCandidate))
    | SelectOffer                                  # Gather(Candidate -> Application)
    | SelectedOffer                                # offered_amount, offer_outcome_code,
                                                   # search_truncated -- the named outputs
                                                   # FinalValidation and s7.1 read; the
                                                   # Gather alone only lifts the chosen
                                                   # candidate's raw fields
)


# ==========================================================================
# 5.13  Validation, evidence, disclosure.
# ==========================================================================
_close = (
      FinalValidation
    | ConditionsAndCovenants
    | DisputeShadow                                # AE-C-22: a disputed event may
                                                   # never be the sole cause of a
                                                   # decline, so the no-disputes
                                                   # outcome is computed always,
                                                   # not on request
    | AttributionSpine
    | ReasonSets
)


business_facility = (
    _structure | _business | _nested | _people | _grade | _pricing | _close
).with_vocabulary(vocabulary).with_overlay_stack(STACK)


# --------------------------------------------------------------------------
# What this expression is asserting, and where each claim is checked:
#
#   * The grain of every module is derivable statically, so `lineage()` answers
#     "which inputs and logic can affect people_pd" across grain shifts without
#     running anything. A Gather is a real node; the witness edges are real
#     edges. (doc 04 s3.)
#
#   * There is no `Loop` anywhere in this pipeline. Every iteration in spec s5
#     -- over entities, over events, over pricing candidates -- became a grain.
#     The only construct that remains iterative is `structure/resolve.py`'s
#     bounded frame-tier expansion, which is three rounds and declared as such.
#
#   * `fuse(ClassifyEvent | EventFlags)` is the one fusion annotation. The Event
#     grain is the only place in this pipeline where row counts are large enough
#     for it to matter (6.7 M rows in the monthly batch) and the two modules are
#     small. doc 02 s1.1: fusion is non-monotone, so this is the size where it
#     wins and nothing else is fused.
#
#   * `parallel(PriceCandidate)` is authored, not inferred (doc 02 s3.3). The
#     candidate grain is the one uniform body in the flow -- every candidate
#     does the same arithmetic -- which is exactly the shape `prange`'s static
#     schedule suits and exactly what credit logic normally is not.
# --------------------------------------------------------------------------
