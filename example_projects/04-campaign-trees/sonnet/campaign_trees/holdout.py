"""Stage 7 -- holdout, control and challenger (spec 04 §5.7). Deterministic assignment from
a stable hash of identifiers alone (§5.7 requirement 1: "no randomness at run time, no
stored assignment table that can drift"), re-derivable from `client_id`, `campaign_id` and
a design version alone (§5.7 requirement 6, acceptance §10 item 8) -- and, because the
window is a plain float comparison against a versioned percentage, a change to that
percentage is the *only* thing that moves anyone between groups (§5.7 requirement 2), which
`tests/test_holdout.py` proves directly by diffing two design versions' assignment sets.
"""
from __future__ import annotations

import hashlib

from decider import missing_as, param, step


def _stable_unit_interval(*parts: object) -> float:
    """A value in `[0, 1)`, deterministic in `parts` alone. `sha256` rather than Python's
    `hash()`: the latter is salted per-process (`PYTHONHASHSEED`) and would make "the same
    client lands in the same group in March, April and May" (§5.7 requirement 2) false the
    moment the cycle runs in a new process."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:12], 16) / 16**12


def is_control(
    client_id: str, campaign_id: int, holdout_design_version: str = missing_as("v1"),
    control_percentage: float = param(0.05, ge=0.0, le=1.0),
) -> bool:
    """§5.7 requirement 3: 5% per campaign by default (12 run 10% while new -- set via
    `control_percentage` per campaign, a `param()`, not here)."""
    return _stable_unit_interval("control", client_id, campaign_id, holdout_design_version) < control_percentage


def is_universal_holdout(
    client_id: str, universal_holdout_design_version: str = missing_as("v1"),
    universal_holdout_percentage: float = param(0.01, ge=0.0, le=1.0),
) -> bool:
    """§5.7 requirement 3: a 1% universal holdout excluded from *every* campaign, to measure
    the whole programme's effect rather than any one campaign's."""
    return _stable_unit_interval("universal", client_id, universal_holdout_design_version) < universal_holdout_percentage


def tree_variant(client_id: str, campaign_id: int, variant_design_version: str,
                  variant_split: tuple[float, ...]) -> int:
    """§5.7 requirement 5: champion/challenger on a percentage split (90/10, or 70/20/10 for
    three variants). Not wired into `pipeline.py`'s demo (campaign 23 runs one variant) --
    exercised directly in `tests/test_holdout.py`, matching 00/03's own "capability
    standalone-testable" precedent for the parts a representative demo pipeline leaves out."""
    u = _stable_unit_interval("variant", client_id, campaign_id, variant_design_version)
    cum = 0.0
    for i, p in enumerate(variant_split):
        cum += p
        if u < cum:
            return i
    return len(variant_split) - 1


is_control_step = step(is_control)
is_universal_holdout_step = step(is_universal_holdout)
