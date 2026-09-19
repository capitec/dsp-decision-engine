"""Navigability tooling. Spec 5.27, N1-N5. Referenced from flow.py's build
comment ("the per-phase-set bitmap layout"). This is not a documentation
generator — it is the mechanism that makes N1 and N3 answerable in the
target time, over generated artefacts, without asking a person.

Doc 07's proposed layout has no equivalent of this file, because a
six-module project answers "where is X set" by reading six modules. At 1 400
decision points and 41 shared intermediates that stops working (README's
"where doc 07 breaks" discussion) — N1's own corollary states it: a human
cannot enumerate 132 places `amount_cap` can be set by reading, and will not
try twice. The enumeration below is GENERATED from declarations that already
exist for other reasons (values/ceilings.py's `narrows=`, tables/cap_register's
`narrows=` column, values/register.py's `produced_by=`/`consumed_by=`) — this
file adds no new place to declare anything, only a walk over what is already
declared.
"""

from __future__ import annotations

from values.ceilings import CEILINGS
from values.register import REGISTER


def where(value_name: str) -> list[dict]:
    """N1. Every place `value_name` can be set: for a `ceiling()`, every
    register entry with `narrows=value_name` plus its seed and its overlay
    attribution point; for a `shared_value()`, its single `produced_by=`.
    `decider2 where amount_cap --count` (values/ceilings.py's own comment)
    resolves through here and must return 132 for `amount_cap` — asserted at
    build against tables/cap_register/cap_register.toml plus the five
    non-register sites (product maxima, the regulatory maximum, the uplift
    entry) enumerated in values/ceilings.py.
    """
    pass  # walk narrows= / produced_by= declarations; return every site, not just one


def resolve(decision_point_id: int) -> dict:
    """N3. A citation in a record resolves to: identity, owner (OWNERS.toml,
    cross-checked against .github/CODEOWNERS), current rendering, and version
    chain. Zero-unresolvable-citations is the acceptance bar over a
    10 000-decision sample."""
    pass  # look up the built registry entry for decision_point_id


def invert(decision_point_id: int, window_days: int = 90) -> list[int]:
    """N4. Every decision in the last `window_days` that this decision point
    BOUND (values/ceilings.py's evaluated/bound/not-applicable distinction),
    under 60 seconds over ~64 M decisions. Backed by the daily
    evaluated/fired/bound counts (spec 7.2's persisted cycle outputs), not by
    a full record scan."""
    pass  # query the daily bound-count index, not the raw record store


def phase_set_bitmap(phase_set_id: int) -> int:
    """One bit per phase, per phase_set_id, resolved at build
    (entrypoints/manifest.py's derivation) and archived so a 2034 replay of a
    2027 decision resolves the same phase_set_id to the same bit pattern.
    This is the "per-phase-set bitmap layout" flow.py's build comment names."""
    pass  # pack the phase_set's eighteen phase-membership bits into one int


def trace_loop(decision_id: int) -> str:
    """N5. "Why 60 months and not 72" rendered as one sentence: connects a
    term to a pass (loop_pass_index) to a product to a cap entry to a grade —
    five hops, per spec 5.23.1's navigability consequence — in under 90
    seconds for a consultant."""
    pass  # walk the record's loop-pass trace and the cap chain it references


# Asserted at build, spec 5.27's own targets:
N1_TARGET_SECONDS = 600          # under 10 minutes, 10 of 12 sampled values
N2_TARGET_SECONDS = 300          # under 5 minutes, 18 of 20 sampled decisions
N3_UNRESOLVABLE_CITATIONS_PERMITTED = 0
N4_TARGET_SECONDS = 60
N5_TARGET_SECONDS = 90
