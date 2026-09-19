"""O1 - O17. Seventeen phases, 1 280 decision points, 78% of them consumed.

Read this file to see what consuming looks like. Twelve of the seventeen lines
below are `from consumed`; five are `from modules`. That ratio is the project's
headline number (spec 4.9) and it is legible from the import block without
reading a single phase.

Nothing in this file is entry-point specific. The nine entry points are
pipelines/entry_points.py, and each is this pipeline `scoped_by` the governance
matrix that Credit Governance owns. There is no second expression of which
phases run.
"""

from decider2 import fuse, parallel
from vocabulary import CREDIT_CORE
from time.bitemporal import KNOWN

# Consumed -- project 05's flow, referenced not restated (spec 5.3).
from consumed.p05_origination import (
    StructureResolution,      # O2   68 dp   05 5.1
    AbsoluteRules,            # O4   88 dp   05 5.3, dispositioned per entry point
    EntityAssessment,         # O5  196 dp   05 5.4-5.7, under five roles
    PeopleBlend,              # O6   62 dp   05 5.8
    Spreading,                # O7  132 dp   05 5.9, 1..6 periods, basis-aware
    CombinedGrade,            # O9   54 dp   05 5.10, emits master_scale_version
    GroupExposure,            # O10  58 dp   05 5.11 + core.exposure
    PricingSearch,            # O13  72 dp   05 5.12, nine candidate spaces
    Conditions,               # O16  42 dp   05 5.13, extended to 61
)
from consumed.p02_affordability import Affordability          # O14  64 dp
from consumed.core_library import eligibility, consent, scorecard  # O3, O8

# Local -- what this project writes. Five phases, 620-ish of the 1 900 with L1-L6.
from modules.request.relationship import Relationship          # O1   34 dp
from modules.appetite.facility_types import Appetite, Constraints  # O11  96 dp
from modules.security.allocation import Security               # O12  88 dp
from modules.covenants.setting import CovenantSetting          # O15  74 dp
from modules.authority.routing import Authority, ApprovalBinding  # O17  38 dp

# --------------------------------------------------------------------------
# The phases, in order. `|` is a sequence and written order is execution order
# (doc 03 8.1), which is what lets a policy analyst read the ladder here and
# nowhere else.
#
# Three things are visible at a glance and are meant to be:
#   1. which phases are consumed (their names came from `consumed`);
#   2. which temporal view each structure-reading phase opened (`view=KNOWN`);
#   3. which roles each entity-level phase ran under (declared in roles.py, and
#      printed at the use site in the rendered artefact).
# --------------------------------------------------------------------------
Origination = (
    Relationship                                   # O1  request, four dates, state
    | StructureResolution.as_known_at("knowledge_date")   # O2  bi-temporal, named
    | eligibility | consent                        # O3  KYB, screening, regime
    | AbsoluteRules                                # O4  20 rules, 9 dispositions each
    | EntityAssessment                             # O5  the nested core, five roles
    | PeopleBlend                                  # O6  PP-01..PP-11
    | Spreading                                    # O7  48 lines, 1..6 periods
    | scorecard.behavioural                        # O8  BUS-BEH-01
    | CombinedGrade                                # O9  + master_scale_version
    | GroupExposure                                # O10 as at a date, contingent
    | fuse(Appetite | Constraints)                 # O11 hot: every entry point runs it
    | Security                                     # O12 cross-facility allocation
    | PricingSearch                                # O13 the bounded search
    | Affordability                                # O14 pinned major, role SOLE_PROPRIETOR
    | CovenantSetting                              # O15 the lifecycle begins here
    | Conditions                                   # O16 61, precedent + subsequent
    | Authority                                    # O17 seven levels, moving
).with_vocabulary(CREDIT_CORE)

# --------------------------------------------------------------------------
# Declared properties. Each generates a test; none is a comment.
# --------------------------------------------------------------------------

# Spec 5.14.3(3). An approval is bound to the structure it approved. Any later
# write inside that fingerprint's lineage invalidates it, automatically, with
# reason 5894. This is one line because the framework already computes lineage.
Origination = Origination.binding(ApprovalBinding)

# Spec 8.2 ordering independence, "extended here to the cross-facility
# allocation, which is the case most likely to violate it". Shuffle entities,
# events, facilities in a pool, signals and covenant instances; the answer must
# not move. The allocation is listed explicitly because it is the one that will.
Origination = Origination.assert_order_independent(
    over=["entities", "events", "pool_facilities", "covenant_instances"],
)

# Spec 5.13.4 / 10 acceptance 17: every phase that reads structure declares
# which temporal view it uses. This is belt-and-braces over the fact that
# `entities` has no unqualified name (time/bitemporal.py) -- the namespace rule
# makes an undeclared read impossible, and this assertion makes a *new* phase
# that adds a structure read fail loudly rather than picking up `known.` by
# habit because that is what the phase above it did.
Origination = Origination.assert_temporal_views_declared()

# Doc 02 1.1: fusion is non-monotone and loses past ~3 modules with cheap arms.
# Exactly one `fuse()` in seventeen phases, on the two cheapest modules that
# every one of the nine entry points runs. Everything else is one kernel per
# module, which is the safe default and costs almost nothing because boundary
# stores are near-free.
#
# `parallel()` appears nowhere in this file. It appears once, in
# pipelines/early_warning.py, over the daily pass -- the only workload in the
# project whose per-row cost is close to uniform. Spec 8.1 profile A is 3 020
# interactive assessments a day at p95 4s, where prange buys nothing and costs
# 1.2-2.6x compile time on 17 kernels.
