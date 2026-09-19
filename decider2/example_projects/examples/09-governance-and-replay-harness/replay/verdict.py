"""Verdicts, and the discipline of not saying `reproduced` when you mean
`nothing obviously disagreed`.

Spec §5.1's acceptance standard is bit-identical on every declared output,
INCLUDING reason code lists and their order. So the comparison is over the
flow's declared output surface from the manifest - `pipeline.schema()`'s
produced set - not over whatever fields happen to be in both dicts. A
field-by-field comparison that iterates the intersection will report
`reproduced` when the replay produced six of seven outputs, and that is the
defect class this module exists to prevent.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from replay.engines import Engine
from replay.seal import ReadCoverage


class Verdict(Enum):
    REPRODUCED = "reproduced"                                  # bit-identical, IMAGE engine only
    REPRODUCED_WITHIN_TOLERANCE = "reproduced_within_tolerance"
    NOT_REPRODUCED = "not_reproduced"
    NOT_REPLAYABLE = "not_replayable"          # the pin could not be assembled; a contract finding


class FieldComparison:
    name: str
    recorded: object
    rederived: object
    tolerance_class: str        # from harness/tolerances.py, versioned and effective-dated
    agrees: bool


class ReplayVerdict:
    decision_id: str
    flow: str
    verdict: Verdict
    engine: Engine
    comparisons: tuple[FieldComparison, ...]        # EVERY declared output, not the intersection
    first_divergence: "Divergence | None"           # step, value name, version index, both values
    coverage: ReadCoverage
    ran_by: str                                     # spec §5.1: replays are themselves evidence
    ran_at: datetime
    purpose: str                                    # required; logged; reviewed monthly (§5.13.2)


def compare(recorded: dict, rederived: dict, *, declared_outputs: tuple[str, ...]) -> tuple[FieldComparison, ...]:
    pass  # iterate declared_outputs; a name missing from either side is a comparison that FAILS, not one that is skipped


def first_divergence(recorded_witness: "Witness", rederived_witness: "Witness") -> "Divergence | None":
    """Walk the two version chains in lockstep and return the first boundary
    crossing where they differ.

    This is doc 04 §5.1's chain doing a job the framework docs never ask of it.
    Because each value version has exactly one producer, the first divergent
    version names the module that caused it - so a failed replay reports
    "term_cap diverged at sector_cap: 48 -> 36 recorded, 48 -> 48 re-derived"
    rather than "outputs differ". That single sentence is the difference
    between a two-hour investigation and a two-day one.
    """
    pass


def monthly_non_reproducible_rate(flow: str, month: str) -> float:
    """Spec §5.1 ceiling: < 0.05% per flow per month, each one individually
    explained, reported to the Credit Committee. Computed over ALL decisions,
    not over the ones somebody happened to replay - which means a deterministic
    sample per flow per day is replayed unprompted. ~11 000/month across the
    estate at 0.5%, which fits the compute budget and is the only way the
    denominator is honest."""
    pass
