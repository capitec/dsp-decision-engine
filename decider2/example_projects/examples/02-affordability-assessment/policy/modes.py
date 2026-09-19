"""The four assessment modes, as profiles.

A mode may not change the arithmetic (spec 5.8). Doc 03 offers `Branch` for
this, which is exactly wrong: four mode-arms over seven stages is four
near-copies with a shared name, and the observable failure -- the tax table
fixed in three of them -- is a matter of time.

A `profile` is not a branch. It is a validated binding over three surfaces and
nothing else:

  * `params`   -- values, per owner class (modules/*/params surfaces)
  * `switches` -- `policy()` fields: closed enums that select an already-compiled
                  behaviour arm
  * `emits`    -- which declared outputs are materialised

`profile()` is validated against the pipeline at build time. Binding a name that
is not a param, a switch or an emit is an error. There is therefore no spelling
of a mode that introduces a step, removes a step, or rewires two steps -- which
is the structural enforcement spec question 2 asks for, rather than the
documentation of it.

The test that makes it real is in tests/test_properties.py:

    assert len({Assessment.under(m).kernel_fingerprint() for m in ALL_MODES}) == 1

All four modes are the same machine code. A mode that needed different machine
code would fail that assertion and would have to be justified as a different
calculation.
"""

from decider2 import profile

from modules.income.waterfall import StaleEvidence
from modules.obligations.treatments import QuoteHandling
from modules.verdict import ToleranceBasis

# --------------------------------------------------------------------------
# 1. New application. ~38 000/day, peak 140/s. The reference mode.
# --------------------------------------------------------------------------
NEW_APPLICATION = profile(
    "new_application",
    code=1,
    switches={
        "income.on_stale_evidence": StaleEvidence.INDETERMINATE,
        "obligations.quote_handling": QuoteHandling.IGNORE_QUOTES,
        "verdict.tolerance_basis": ToleranceBasis.APPETITE,
    },
    params={
        "income.min_evidence_tier_source": "product",
        "capacity.buffer_grid_name": "standard",
    },
    emits=["*always", "discretionary_income_after_cents"],
)

# --------------------------------------------------------------------------
# 2. Limit increase (project 07). 14 M records monthly, batch.
#    Stale income does not fail here -- it routes to a conditional outcome.
#    That is a *switch*, not a different stage.
# --------------------------------------------------------------------------
LIMIT_INCREASE = profile(
    "limit_increase",
    code=2,
    switches={
        "income.on_stale_evidence": StaleEvidence.CONDITIONAL,
        "obligations.quote_handling": QuoteHandling.IGNORE_QUOTES,
        "verdict.tolerance_basis": ToleranceBasis.APPETITE,
    },
    params={
        "income.min_evidence_tier_source": "product",
        "capacity.buffer_grid_name": "standard",
    },
    # Shape (b): capacity, no proposed instalment. Asking this mode for a
    # verdict against an instalment is an error, not a silent `pass`.
    emits=["*always"],
    forbids=["proposed_instalment_cents"],
)

# --------------------------------------------------------------------------
# 3. Arrangement (project 08). ~26 000/day. The client is in arrears and the
#    question is sustainability, not appetite. Different grid, larger floor,
#    different `marginal` tolerance -- all values and switches.
# --------------------------------------------------------------------------
ARRANGEMENT = profile(
    "arrangement",
    code=3,
    switches={
        "income.on_stale_evidence": StaleEvidence.PERMIT_WEAKER_TIER,
        "obligations.quote_handling": QuoteHandling.IGNORE_QUOTES,
        "verdict.tolerance_basis": ToleranceBasis.SUSTAINABILITY,
    },
    params={
        "income.min_evidence_tier_source": "arrangement",   # a weaker floor
        "capacity.buffer_grid_name": "arrangement",
        "expenses.prefer_statement_basis": True,
    },
    emits=["*always", "discretionary_income_after_cents"],
)

# --------------------------------------------------------------------------
# 4. Scenario (project 06). Up to 400 per application, always through
#    `hold()`/`resume()`. The only mode in which a live settlement quotation
#    zeroes an account -- and the switch is what makes that impossible to leak
#    into new-application mode, because the constant is an immediate in the
#    compiled arm and the profile is the only thing that can set it.
# --------------------------------------------------------------------------
SCENARIO = profile(
    "scenario",
    code=4,
    inherits=NEW_APPLICATION,           # evidence policy is the parent's, by construction
    switches={
        "obligations.quote_handling": QuoteHandling.EXCLUDE_ON_QUOTE,
    },
    emits=["*always", "discretionary_income_after_cents", "accounts_annotated"],
)

ALL_MODES = (NEW_APPLICATION, LIMIT_INCREASE, ARRANGEMENT, SCENARIO)

# `inherits=` is not sugar. It is what stops mode 4 from acquiring its own
# evidence policy: SCENARIO cannot set `income.on_stale_evidence` at all,
# because a profile may override an inherited switch only where the parent
# declares it `overridable`. Project 06's scenarios are *the same applicant*
# as the parent assessment; an evidence rule that differed between them would
# make the search's answer un-re-checkable against the real assessment.
