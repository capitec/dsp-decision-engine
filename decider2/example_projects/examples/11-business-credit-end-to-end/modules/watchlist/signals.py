"""L3 -- the 186-signal catalogue and the six-grade watchlist. Spec 5.6.

240 000 facilities, daily, in a 3-hour window completing by 05:00, with fan-out:
a signal on one entity propagates to every facility of every business that
entity touches. The p99 entity touches 31 facilities and 340 entities in the
book touch more than 100.

---------------------------------------------------------------------------
Why the signal catalogue is a `ruleset` interior and the grade is not
---------------------------------------------------------------------------
Doc 08 3.4's test: "can one compiled loop evaluate every instance of this kind,
with the instance supplied as arrays?"

  186 signals x 11 attributes, each a threshold comparison over a named feature
  with a weight and a decay profile -> YES. A `decision_table`-shaped generic
  kernel over a 186-row array. Early Warning edits it MONTHLY and the edit must
  not recompile anything (spec 10 acceptance 34: "a policy owner changes a
  signal weight... without an engineer, producing a reviewable diff and a
  re-scored sample over the live book").

  The GRADE is six boundaries plus seven trigger overrides plus the
  escalate-automatically / de-escalate-by-human asymmetry -> NO. Codegen.

So the catalogue is `values` (free, no compile) and the grade is `skeleton`
(Python, reviewed). That split is doc 08 2 working exactly as designed, and it
is the one place in this project where the three change classes land cleanly on
a real artefact without argument.
"""

from decider2 import decision_table, module, step, param, Branch
from roles import SIGNAL_SCORE
from consumed.core_library import adverse_events

# 186 signals x 11 attributes (spec table 48). A generic kernel; free to edit.
signal_catalogue = decision_table(
    "early_warning_signals",
    rows=186,
    attributes=["signal_id", "family", "source", "window", "trigger_level",
                "weight", "decay_profile", "is_trigger", "trigger_grade",
                "subject_scope", "owner"],
    owner="early_warning",
    cadence="monthly",
    generic_kernel=True,          # interior changes are free (doc 08 3.4)
)

# 186 x 6 age bands = 1 116 cells. The mechanism by which a facility comes OFF
# the watchlist without anybody doing anything.
signal_decay = decision_table("signal_decay", rows=186, cols=6,
                              owner="early_warning", cadence="quarterly")


def signal_set(subject_key: int, subject_scope: int) -> list:
    """Evaluate the catalogue for one subject. 22 of 186 have any data.

    `subject_scope` is spec 5.6.2 rule 3: the grade is per FACILITY, the signals
    are per SUBJECT, and there are three scopes -- entity, client, facility.
    A signal on an entity applies to every facility of every business that
    entity touches. The aggregation across scopes is where the fan-out lives,
    and it is a frame-tier join, not a record-tier loop. pipelines/early_warning.py.
    """
    pass  # generic-kernel scan of the catalogue for this subject


def decayed_weight(signal_age_days: int, signal_id: int) -> float:
    """A returned debit from fourteen months ago carries a fraction of its
    weight; a confirmed fraud marker carries all of it, forever (floor = initial).
    """
    pass  # signal_decay[signal_id, age_band]


def change_cause(prev_set: list, signal_set: list) -> int:
    """NEW SIGNAL or DECAY. Spec 5.6.3 requirement 3.

    "The first is news; the second is arithmetic; and a watchlist committee told
     they are the same thing will stop reading the report."

    This is the single cheapest requirement in the project to satisfy and the
    one most likely to be dropped, because both cases produce the same delta. It
    is a required output of EP-5, not a diagnostic, so it is a step with a
    reason code and not a tap.
    """
    pass  # set-difference the two signal sets; classify


def watchlist_grade(score: float, triggers: list,
                    boundaries: tuple = param((15, 30, 50, 75),
                        description="W1/W2/W3/W4 score boundaries")) -> int:
    """W0..W5. Trigger overrides dominate the aggregate.

    Spec 5.15.1 collision 3 is a live hazard on this function: Early Warning
    adds 14 signals and the boundaries were calibrated on the previous 172.
    Every facility's aggregate score shifts, thousands move a grade for no
    credit reason, and the mandated actions fire -- 2 800 site visits against a
    capacity of 180 a week (change scenario 12).

    The defence is not in this function. It is that a catalogue edit is a
    `values` change and therefore goes through `decider2.impact(active,
    candidate, sample)` (doc 08 5) over the LIVE BOOK before activation, and the
    impact report reads in the terms Early Warning cares about: "2 800
    facilities change grade, of which 2 640 are scale artefacts". The framework
    already has that mechanism. What it does not have is a way to say the impact
    review is MANDATORY for this artefact -- doc 08 5's own last line is "not
    enforced: that anyone runs it". FRAMEWORK-DEMANDS D15.
    """
    pass  # max(boundary lookup, trigger floor)


EscalationAsymmetry = Branch(
    lambda proposed, current: proposed > current,
    [module(watchlist_grade, name="escalate_automatically"),
     module(watchlist_grade, name="de_escalate_requires_human")],
    modifies=["watchlist_grade"],
)
"""Spec 5.6.2 rule 1: escalation is automatic; de-escalation above W2 is not.

A facility moves up on the daily pass. Moving down from W3, W4 or W5 requires a
recorded human decision with a rationale and an authority, because the decay
table will otherwise quietly rehabilitate a business that has not recovered.

Expressed as a Branch rather than an `if` inside the grade function so that
`branch_path` records which arm fired on all 240 000 facilities, every night,
for free (doc 04 4.1: a compile-time immediate). "How many facilities did the
decay table rehabilitate this month" is then a group-by over a column.
"""


def mandated_actions(watchlist_grade: int, facility_type: int) -> list:
    """270-cell matrix. Spec 5.6.2 rule 2: an action is TRACKED TO COMPLETION,
    and an overdue site visit is itself a signal. So the output is not a list of
    strings -- it is rows with owners and deadlines that feed back into the
    catalogue as signals, which makes the watchlist a closed loop and "an action
    list with a 40% completion rate" a measurable defect rather than a fact
    nobody computes.
    """
    pass  # matrix lookup; emit (action, owner, deadline) rows


Watchlist = module(signal_set, decayed_weight, change_cause, mandated_actions,
                   name="watchlist", owner="early_warning",
                   co_owners=["portfolio_management", "credit_risk_policy"],
                   taps=["watchlist_score", "change_cause", "branch_path"])
