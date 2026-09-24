"""P04 -- Data acquisition orchestration, and this project's one degraded mode (spec 10 §5.5, §5.25).

This project makes no real external calls (there is nothing to call from a
single decider process), so "orchestration" here is the part that survives
without a network: deciding, from the request and a caller-supplied
`bureau_available` flag (standing in for "the bureau call succeeded"),
which sources are in play and what `degraded_mode_code` that implies.

**The one declared degraded mode this project implements** (SCOPE.md):
bureau down (10 §5.25, code 11). "Approve only within a reduced envelope:
<= R25 000, term <= 36, segments 3-5 only, internal tenure >= 24 months, no
internal arrears in 24. Everyone else refers." P09's cap register
(`retail_credit.cap_waterfall`) reads `degraded_mode_code` and applies
CAP-DEGRADED-BUREAU accordingly; P07's scoring reads it too (falls back to
the thin-file scorecard with a -45 point shift, per §5.25) -- see
`retail_credit.scoring`.

Left as a declared stub, not implemented: the other eleven degraded modes
in §5.25 (statement aggregator down, fraud service timing out, ...) and
the real source-orchestration sequencing (concurrent waves, the arrival
contract for late/partial/schema-invalid responses). None of those need a
second mechanism once one degraded mode is threaded end to end through
P04 -> P06 -> P07 -> P09 -> P18 -- they need the same mechanism, exercised
eleven more times, which is repetition, not new difficulty.
"""
from __future__ import annotations

from decider import missing_as, param, step

from retail_credit.vocab import DEGRADED_BUREAU_DOWN, DEGRADED_NONE


def orchestrate_acquisition(
    bureau_available: bool = missing_as(True),
) -> tuple[int, list[int]]:
    """(degraded_mode_code, source_degradation_codes). `source_degradation_codes` (10 §4.6)
    is every source that was degraded, with its degradation kind, as at read time -- here
    just the one source this project models.
    """
    if not bureau_available:
        return DEGRADED_BUREAU_DOWN, [DEGRADED_BUREAU_DOWN]
    return DEGRADED_NONE, []


orchestrate_acquisition_step = step(orchestrate_acquisition, outputs=("degraded_mode_code", "source_degradation_codes"))


def bureau_down_reduced_envelope_ok(
    degraded_mode_code: int,
    segment_code: int,
    internal_tenure_months: float = missing_as(0.0),
    worst_arrears_months: float = missing_as(0.0),
    requested_amount: float = missing_as(0.0),
    term_months: int = missing_as(0),
    max_reduced_amount: float = param(25_000.0),
    max_reduced_term: int = param(36),
) -> bool:
    """The bureau-down reduced envelope (10 §5.25): True only where every one of the five
    conditions holds. False means "refer", never "decline" and never "approve on an
    assumption" -- degradation narrows what can be approved, it never invents evidence.
    """
    if degraded_mode_code != DEGRADED_BUREAU_DOWN:
        return True  # not degraded: this gate does not apply
    return (
        requested_amount <= max_reduced_amount
        and term_months <= max_reduced_term
        and segment_code in (3, 4, 5)
        and internal_tenure_months >= 24.0
        and worst_arrears_months <= 0
    )


bureau_down_reduced_envelope_ok_step = step(bureau_down_reduced_envelope_ok, output="bureau_down_envelope_ok")
