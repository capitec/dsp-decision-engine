"""Names this project declares locally (spec 10 §4.6).

The canonical vocabulary (project 00 §4, `credit_core.vocab`) is used
unchanged everywhere it applies. These fifteen names do not exist in it,
because no isolated flow needs them -- they exist only because this flow
has eight entry points, eighteen phases and a cross-phase loop. That an
end-to-end flow needs fifteen names the shared library does not have is
itself one of this project's findings (see NOTES.md "Spec problems").
"""
from __future__ import annotations

from enum import IntEnum

# entry_point_code: 1..8, set at admission (P01), immutable, on every record.
EP_NEW_APPLICATION = 1
EP_LIMIT_CHANGE = 2
EP_LIMIT_PROGRAMME = 3
EP_CAMPAIGN_PRE_APPROVAL = 4
EP_CONSOLIDATION = 5
EP_REPRICE = 6
EP_QUOTATION = 7
EP_WHAT_IF = 8


class RecordCompletenessCode(IntEnum):
    """§5.31.1 -- the four absences that must never collapse into one null."""

    COMPLETE = 1
    PHASE_ABSENT_BY_ENTRY_POINT = 2       # structural: this entry point never runs this phase
    PHASE_ABANDONED_ON_BUDGET = 3          # operational: overrun, and the phase was abandonable
    PHASE_DEGRADED = 4                     # degraded: the phase ran but could not produce the value
    EMISSION_PARTIAL = 5


class LoopTerminationCode(IntEnum):
    """§4.6, §5.23.1 -- why a cross-phase loop (L1) stopped."""

    CONVERGED = 1          # a shippable offer was found and validated
    NO_IMPROVEMENT = 2     # the next pass could not strictly improve; not run
    BOUND_REACHED = 3      # the 4-pass bound was reached
    NO_CANDIDATE = 4       # no settlement scenario was available at all
    ABANDONED_ON_BUDGET = 5


class ValueBasisCode(IntEnum):
    """§5.21.1 -- which version of a shared concept a recorded value is."""

    ACTUAL = 1
    HYPOTHETICAL = 2
    STRESSED = 3
    ENTRY_POINT_SUBSTITUTED = 4


# degraded_mode_code -- the composite mode in force, declared in advance (§5.25), never
# computed after the fact. This project implements exactly one of the twelve (per
# SCOPE.md): "bureau down" (10 §5.25, code 11).
DEGRADED_NONE = 0
DEGRADED_BUREAU_DOWN = 11

# Binding-constraint codes for the solve (P13, §5.14 item 6).
BIND_AFF = "BIND-AFF"
BIND_CAP = "BIND-CAP"
BIND_REQ = "BIND-REQ"
BIND_MIN = "BIND-MIN"
BIND_MAX = "BIND-MAX"
BIND_EXH = "BIND-EXH"

# outcome_code (10 §7.1, entry point 1 shape).
OUTCOME_APPROVE = 1
OUTCOME_APPROVE_WITH_CONDITIONS = 2
OUTCOME_REFER = 3
OUTCOME_DECLINE = 4
