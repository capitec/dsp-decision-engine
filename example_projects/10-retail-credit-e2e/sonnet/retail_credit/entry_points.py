"""The eight entry points and phase-set resolution (spec 10 §4.1, §5.20).

`phase_set_id` is not derivable from `entry_point_code` alone (10 §4.6):
P14 is conditional on entry point 1 (only when affordability fails and the
client is consolidation-eligible) and P10 is conditional on entry point 6
(only where a re-price raises the instalment). `resolve_phase_set` is the
one place that fact is decided -- P01's job (10 §5.2), and the only place
from which "why did phase 14 not run for this client" is answerable
without inference.

Coverage in this project (SCOPE.md): the matrix below is declared in full,
for all eight entry points, at ● / ◐ / ○ granularity -- that is the
"declared stubs... with phase-set resolution" requirement. Only entry
point 1's phases carry real logic (see `pipeline.build`); entry points
2-8 are routing declarations only, proven by `tests/test_entry_points.py`
routing to the right phase set, not by running them end to end.
"""
from __future__ import annotations

from dataclasses import dataclass

RUNS = "runs"                 # ●
REDUCED = "reduced"           # ◐ -- runs, but in a modified/reduced form
CONDITIONAL = "conditional"   # ○ -- runs only if a runtime condition holds
SKIPPED = "skipped"           # blank

# spec 10 §5.20's matrix, one row per entry point, P01..P18 in order.
_MATRIX: dict[int, tuple[str, ...]] = {
    1: (RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS,
        CONDITIONAL, SKIPPED, RUNS, RUNS, RUNS),
    2: (RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, SKIPPED, REDUCED, SKIPPED,
        SKIPPED, REDUCED, RUNS, RUNS, RUNS),
    3: (RUNS, REDUCED, RUNS, REDUCED, SKIPPED, RUNS, RUNS, RUNS, RUNS, REDUCED, SKIPPED, REDUCED, SKIPPED,
        SKIPPED, RUNS, RUNS, RUNS, RUNS),
    4: (RUNS, SKIPPED, RUNS, REDUCED, SKIPPED, RUNS, RUNS, RUNS, RUNS, REDUCED, RUNS, RUNS, RUNS,
        REDUCED, REDUCED, RUNS, RUNS, RUNS),
    5: (RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS, RUNS,
        RUNS, SKIPPED, RUNS, RUNS, RUNS),
    6: (RUNS, RUNS, RUNS, SKIPPED, SKIPPED, RUNS, RUNS, RUNS, RUNS, CONDITIONAL, SKIPPED, RUNS, SKIPPED,
        SKIPPED, SKIPPED, RUNS, RUNS, RUNS),
    7: (RUNS, SKIPPED, REDUCED, SKIPPED, SKIPPED, SKIPPED, SKIPPED, SKIPPED, REDUCED, SKIPPED, RUNS, RUNS, SKIPPED,
        SKIPPED, SKIPPED, SKIPPED, REDUCED, RUNS),
    # Entry point 8 (what-if) has no fixed phase set: it is the phase set of the decision
    # it derives from (10 §4.1). Not represented as a row -- `resolve_phase_set(8, ...)`
    # requires the base decision's own `phase_set_id`.
}

_PHASE_IDS = tuple(range(1, 19))  # P01..P18


@dataclass(frozen=True)
class PhaseRun:
    phase_id: int
    status: str  # RUNS | REDUCED | CONDITIONAL | SKIPPED


def declared_matrix(entry_point_code: int) -> tuple[PhaseRun, ...]:
    """The declared (pre-conditional) row for one entry point -- §5.20's table, unresolved."""
    if entry_point_code not in _MATRIX:
        raise ValueError(f"entry point {entry_point_code} has no fixed matrix row (8 derives its base decision's)")
    return tuple(PhaseRun(pid, status) for pid, status in zip(_PHASE_IDS, _MATRIX[entry_point_code]))


def resolve_phase_set(
    entry_point_code: int,
    *,
    consolidation_loop_fires: bool = False,   # EP1's P14 condition (10 §5.2, §5.20)
    reprice_raises_instalment: bool = False,  # EP6's P10 condition (10 §5.20)
) -> list[int]:
    """The actual phase set that ran, after conditionals resolve -- what `phase_set_id`
    identifies (10 §4.6). Order is phase execution order, P01 first.
    """
    rows = declared_matrix(entry_point_code)
    resolved: list[int] = []
    for row in rows:
        if row.status in (RUNS, REDUCED):
            resolved.append(row.phase_id)
        elif row.status == CONDITIONAL:
            if row.phase_id == 14 and consolidation_loop_fires:
                resolved.append(row.phase_id)
            elif row.phase_id == 10 and entry_point_code == 6 and reprice_raises_instalment:
                resolved.append(row.phase_id)
        # SKIPPED: never resolved in.
    return resolved


def phase_set_id(entry_point_code: int, phase_ids: list[int]) -> int:
    """A stable, content-derived identity for one exact set of phases (10 §4.6
    `phase_set_id`: "the identity of the exact set of phases that ran, resolvable to the
    list"). Content-derived, not sequential, so two decisions that ran the same set always
    carry the same id regardless of when either ran.
    """
    return entry_point_code * 100_000 + sum(1 << (pid - 1) for pid in phase_ids)


def phases_for_id(entry_point_code: int, resolved_id: int) -> list[int]:
    """The inverse of `phase_set_id`: resolves an id back to its phase list (10 §4.6
    "resolvable to the list" -- required for N3 navigability, §5.27).
    """
    bits = resolved_id - entry_point_code * 100_000
    return [pid for pid in _PHASE_IDS if bits & (1 << (pid - 1))]
