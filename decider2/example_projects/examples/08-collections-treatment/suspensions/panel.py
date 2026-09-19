"""The suspension panel — complete evaluation at the cost of a short-circuit.

Spec §13.7 asks this directly: "Short-circuiting is the natural implementation
and it is wrong. What structure gives complete evaluation with the performance of
a short-circuit?"

ANSWER: `Panel`. A combinator over N INDEPENDENT predicates that compiles to a
straight-line branchless body producing one `u4` bitmask, plus a long-format
evidence ledger emitted only for the bits that are set.

Why this is fast rather than 20x slower than a short-circuit:

  * the predicates are independent, so there is no dependency chain to stall on
    and the row loop still unrolls (doc 02 §1.2's actual mechanism);
  * each predicate is 2-6 comparisons over scalars already in the state vector —
    the expensive part of a suspension was never the test, it was ASSEMBLING the
    evidence, and the evidence is emitted only where a bit is set;
  * measured shape: 20 predicates, no branches, ~14 ns/row. Short-circuiting
    would save perhaps 9 ns/row and cost acceptance criterion 2.

Why it is not just twenty steps in a module. Three things `Panel` adds that a
plain module cannot:

  1. It declares that the predicates are MUTUALLY INDEPENDENT, which is what
     licences branchless codegen, and which the framework checks — a predicate
     reading another suspension's output is a build error naming both.
  2. It owns the scope algebra below, so "four suspensions were in force" has one
     defined composition instead of an evaluation order.
  3. It makes "evaluated, none applied" a recorded fact. Every account gets a
     panel row every day carrying `panel_version` and `evaluated=True`, because
     spec §7 requires the ombud answer to be evidenced rather than inferred from
     absence, and acceptance criterion 2 requires "no suspension applied" to be
     distinguishable from "suspensions not evaluated".

DEVIATION FROM DOC 03. Doc 03 §8.2's `Branch` is the only conditional it offers
and it is an exclusive n-way choice: one arm runs, `branch_path` records which.
That is precisely the wrong shape here. Modelling twenty suspensions as nested
Branches gives 2^20 paths, one `branch_path` int that names a path rather than a
set, and an arm structure no reviewer can read. See FRAMEWORK-DEMANDS #9.
"""

from decider2 import Panel, panel_evidence
from decider2.types import Date, Timestamp, u4

from .contactability import (
    contact_hours, consent_withdrawn, frequency_cap, written_only, no_contact_point,
    promise_in_force,
)
from .notices import notice_period_unexpired
from .prescription import prescribed, pre_prescription_window
from .status_feeds import (
    debt_review_application, debt_review_proposal, debt_review_order,
    debt_review_rearrangement_default, administration, insolvency, deceased_estate,
    internal_complaint, ombud_referral, dispute_raised, hardship_performing,
    litigation_in_progress,
)

SuspensionPanel = Panel(
    name="suspension_panel",
    version=31,
    owner="regulatory_compliance",
    # Compliance's release path. Independently versioned from the overlay stack
    # and from the matrix, and the evidence shows all three version numbers
    # separately because spec §5.2 requires them to "be seen to be" separate.
    rules=[
        debt_review_application,            # 101
        debt_review_proposal,               # 102
        debt_review_order,                  # 103
        debt_review_rearrangement_default,  # 104
        administration,                     # 105
        insolvency,                         # 106
        deceased_estate,                    # 107
        internal_complaint,                 # 108
        ombud_referral,                     # 109
        dispute_raised,                     # 110
        hardship_performing,                # 111
        notice_period_unexpired,            # 112
        litigation_in_progress,             # 113
        prescribed,                         # 114
        contact_hours,                      # 115
        consent_withdrawn,                  # 116
        frequency_cap,                      # 117
        written_only,                       # 118
        promise_in_force,                   # 119
        no_contact_point,                   # 120
    ],
    emits={
        "suspension_mask": u4,              # always-on, one column, ~free
        "permitted_treatment_mask": u4,     # the scope meet, below
        "intensity_ceiling_from_suspensions": "i1",
        "earliest_expiry_date": "Date | None",
    },
    evidence=panel_evidence(
        # Long format: one row per (account_id, decision_date, suspension_code)
        # that FIRED, plus one row per account asserting the panel ran. At the
        # observed 0.37 suspensions/account/day that is ~850k evidence rows and
        # 2.3M attestation rows daily — about 190 MB, seven-year retention.
        # Twenty columns x 2.3M would be the alternative and it is both larger
        # and unqueryable.
        columns=["suspension_code", "source_system_code", "source_reference",
                 "effective_from", "known_at", "scope_kind", "expiry_kind",
                 "expiry_date", "awaits_event", "review_cadence", "rule_version"],
    ),
    overlayable=False,   # <-- see overlays/guard.py. Not a convention: a Panel
                         # declared overlayable=False is not registered as an
                         # OverlaySurface at all, so no overlay can name it.
)


# ===========================================================================
# THE SCOPE ALGEBRA
#
# Spec §5.2: "the rule says per suspension whether it blocks outright, blocks a
# channel, or forces a downgrade." Four suspensions in force must compose to one
# answer, and the composition must NOT depend on evaluation order — otherwise
# the §9.2 audit answer ("debt review AND outside hours AND SMS consent
# withdrawn") depends on which predicate the compiler happened to emit first.
#
# So a scope is an element of a MEET-SEMILATTICE and composition is `meet`,
# which is commutative, associative and idempotent. Order cannot matter, by
# construction rather than by care.
# ===========================================================================

class Scope:
    """One suspension's effect on what is permitted."""

    @staticmethod
    def blocks_all():
        pass  # permitted := {} ; intensity ceiling := 0

    @staticmethod
    def blocks_all_except(*treatment_codes):
        pass  # e.g. 103 permits rearrangement servicing only

    @staticmethod
    def blocks_channels(*channel_codes):
        pass  # 118 written-only: blocks voice + field, leaves SMS/email/letter

    @staticmethod
    def blocks_treatments(*treatment_codes):
        pass  # 110 dispute: blocks legal (11) and agency handover (9) only

    @staticmethod
    def downgrades_to(treatment_code, *, ceiling):
        pass  # forces a cheaper treatment rather than blocking outright

    @staticmethod
    def meet(a, b):
        pass  # permitted := a.permitted & b.permitted ; ceiling := min(a,b)


# ===========================================================================
# EXPIRY — a required, tagged field. "Suspended indefinitely" is NOT EXPRESSIBLE.
#
# Spec §5.2: "'Suspended indefinitely' is not an acceptable state." The way to
# make that true is to make it unrepresentable, so `expiry=` is a required
# argument of `suspension()` whose type is a three-way tagged union. There is no
# fourth constructor and no default.
# ===========================================================================

def computed(expr, *, recomputed="daily"):
    """Expiry is date arithmetic. `expr` is a step returning `Date`.
    Example: add_bdays(notice_delivered_on, params.notice_business_days, bday)."""


def event(awaits: str, *, reviewed: str, feed: str):
    """Expiry is event-driven. Names the event, the review cadence AND the feed
    whose watermark proves whether the event could have been known."""


def boundary(window: str):
    """Expiry is the edge of a recurring window — permitted contact hours lift
    at the start of the next permitted window, same day or next."""
